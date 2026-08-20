#!/usr/bin/env python3
"""SAM 3.1 Object-Multiplex video tracking: LiteRT-only host loop.

The full tracker state machine (detection, NMS, det<->track association, hotstart,
occlusion suppression, memory bank + temporal pos-enc, recondition every 16 frames,
masklet confirmation, bucketized multiplex decode) re-implemented in numpy on top of the
seven LiteRT CompiledModel graphs. NO torch, NO sam3 package at runtime — this file is
the executable spec for the Kotlin/Swift port.

Graphs (Mac GPU via ai_edge_litert):
  models/out/sam3_vision_tri.tflite   image -> 9 feature maps (detector + tracker necks)
  models/out/sam3_text.tflite         token embeddings -> text memory
  models/out/sam3_head.tflite         features + text -> 200 detections
  models/tracker_precheck/trk_memattn_n7.tflite  memory attention (7 slots + 16 ptr frames)
  models/tracker_precheck/trk_maskdec.tflite     multiplex mask decoder (16 obj x 3 masks)
  models/tracker_precheck/trk_memenc.tflite      memory encoder
  models/tracker_precheck/trk_initdec.tflite     interactive decoder (mask-as-output init)

Host constants come from models/tracker_host/{consts.npz,flags.json}
(dump_tracker_host_assets.py). Verification vs the all-PyTorch reference:

  tracker_host_loop.py --clip models/clip8  --ref models/tracker_ref/ref_person.npy
  tracker_host_loop.py --clip models/clip24 --ref models/tracker_ref24/ref_person.npy

Known deliberate deviation from the python reference: torch's UNSTABLE bool argsort
orders simultaneously-appearing new detections in an algorithm-dependent way; this loop
uses the deterministic rule "first (lowest) kept row, then remaining kept rows in
descending row order", verified to match torch on every frame of both reference clips.
In general it may permute the ID assignment among new objects appearing in the same
frame — a labeling artifact, not a tracking difference.
"""
import argparse
import json
import math
import os
import re
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "models", "out")
PRE = os.path.join(ROOT, "models", "tracker_precheck")
HOSTD = os.path.join(ROOT, "models", "tracker_host")

C = 256
HW72 = 72 * 72
L = 5184
MASK = 288
IMG = 1008
INMASK = 1152                 # tracker input_mask_size
NO_OBJ_SCORE = -1024.0

CONSTS = dict(np.load(os.path.join(HOSTD, "consts.npz")))
FLAGS = json.load(open(os.path.join(HOSTD, "flags.json")))

VIS_LAYOUT = [("sam3_fpn288", 256, 288), ("sam3_fpn144", 256, 144), ("sam3_fpn72", 256, 72),
              ("inter_h0", 32, 288), ("inter_h1", 64, 144), ("inter_f2", 256, 72),
              ("prop_h0", 32, 288), ("prop_h1", 64, 144), ("prop_f2", 256, 72)]


# ============================================================ numeric primitives
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def bf16(x):
    """float32 -> bfloat16 -> float32 (round-to-nearest-even), like torch .to(bfloat16)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((b >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    return ((b + r) & np.uint32(0xFFFF0000)).view(np.float32)


def interp_bilinear(x, oh, ow):
    """torch F.interpolate(mode='bilinear', align_corners=False, antialias=False), NCHW."""
    n, c, ih, iw = x.shape
    if (ih, iw) == (oh, ow):
        return x.astype(np.float32).copy()

    def axis_idx(i_sz, o_sz):
        src = (np.arange(o_sz, dtype=np.float64) + 0.5) * (i_sz / o_sz) - 0.5
        src = np.maximum(src, 0.0)
        i0 = np.minimum(np.floor(src).astype(np.int64), i_sz - 1)
        i1 = np.minimum(i0 + 1, i_sz - 1)
        lam = (src - i0).astype(np.float32)
        return i0, i1, lam

    y0, y1, ly = axis_idx(ih, oh)
    x0, x1, lx = axis_idx(iw, ow)
    x = x.astype(np.float32)
    top = x[:, :, y0, :] * (1 - ly)[None, None, :, None] + x[:, :, y1, :] * ly[None, None, :, None]
    return top[:, :, :, x0] * (1 - lx)[None, None, None, :] + top[:, :, :, x1] * lx[None, None, None, :]


_AA_CACHE = {}


def _aa_weights(i_sz, o_sz):
    """Triangle-filter weights of torch's antialias=True bilinear (PIL-style)."""
    key = (i_sz, o_sz)
    if key in _AA_CACHE:
        return _AA_CACHE[key]
    scale = i_sz / o_sz
    support = max(scale, 1.0)
    centers = (np.arange(o_sz, dtype=np.float64) + 0.5) * scale
    xmin = np.maximum(0, (centers - support + 0.5).astype(np.int64))
    xmax = np.minimum(i_sz, (centers + support + 0.5).astype(np.int64))
    ws = []
    for o in range(o_sz):
        k = np.arange(xmin[o], xmax[o])
        w = np.clip(1.0 - np.abs((k + 0.5 - centers[o]) / support), 0.0, None)
        s = w.sum()
        ws.append((w / s if s > 0 else w).astype(np.float32))
    _AA_CACHE[key] = (xmin, xmax, ws)
    return _AA_CACHE[key]


def interp_bilinear_aa(x, oh, ow):
    """torch F.interpolate(mode='bilinear', align_corners=False, antialias=True), NCHW."""
    n, c, ih, iw = x.shape
    x = x.astype(np.float32)
    if ih != oh:
        xmin, xmax, ws = _aa_weights(ih, oh)
        out = np.empty((n, c, oh, iw), dtype=np.float32)
        for o in range(oh):
            out[:, :, o, :] = np.einsum("nchw,h->ncw", x[:, :, xmin[o]:xmax[o], :], ws[o])
        x = out
    if iw != ow:
        xmin, xmax, ws = _aa_weights(iw, ow)
        out = np.empty((n, c, x.shape[2], ow), dtype=np.float32)
        for o in range(ow):
            out[:, :, :, o] = np.einsum("nchw,w->nch", x[:, :, :, xmin[o]:xmax[o]], ws[o])
        x = out
    return x


_ERF = np.frompyfunc(math.erf, 1, 1)


def gelu(x):
    return (0.5 * x * (1.0 + _ERF(x / np.sqrt(2.0)).astype(np.float32))).astype(np.float32)


def conv_stride_eq_kernel(x, w, b, k):
    """Conv2d with stride == kernel (non-overlapping blocks), NCHW, exact."""
    n, ci, h, wd = x.shape
    xb = x.reshape(n, ci, h // k, k, wd // k, k)
    y = np.einsum("ncykxl,ockl->noyx", xb, w, optimize=True).astype(np.float32)
    return y + b[None, :, None, None]


def layernorm2d(x, w, b, eps=1e-6):
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w[None, :, None, None] + b[None, :, None, None]


def linear(x, w, b):
    return (x @ w.T + b).astype(np.float32)


def mlp3(x, prefix):
    for i in range(3):
        x = linear(x, CONSTS[f"{prefix}.{i}.w"], CONSTS[f"{prefix}.{i}.b"])
        if i < 2:
            x = np.maximum(x, 0.0)
    return x


def get_1d_sine_pe(pos, dim=256, temperature=10000.0):
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pe = pos[:, None] / dim_t[None, :]
    return np.concatenate([np.sin(pe), np.cos(pe)], axis=-1).astype(np.float32)


# ============================================================ LiteRT graphs
class GpuGraph:
    def __init__(self, path, f32=True, n_out=None):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import Options
        t0 = time.time()
        if f32:
            o = Options.create()
            o.hardware_accelerators = HardwareAccelerator.GPU
            o.gpu_options.enforce_f32 = True
            self.m = CompiledModel.from_file(path, options=o)
        else:
            self.m = CompiledModel.from_file(path, HardwareAccelerator.GPU)
        self.compile_s = time.time() - t0
        self.ib = self.m.create_input_buffers(0)
        self.ob = self.m.create_output_buffers(0)
        self.n_out = n_out
        self.name = os.path.basename(path)
        self.calls = 0
        self.ms = 0.0

    def __call__(self, x):
        self.ib[0].write(np.ascontiguousarray(x, dtype=np.float32).ravel())
        t0 = time.time()
        self.m.run_by_index(0, self.ib, self.ob)
        y = np.array(self.ob[0].read(self.n_out, np.float32))
        self.ms += (time.time() - t0) * 1000
        self.calls += 1
        return y


# ============================================================ CLIP BPE tokenizer
class BpeTokenizer:
    BOS, EOT, MAX_LEN = 49406, 49407, 32

    def __init__(self, vocab_path, merges_path):
        self.encoder = json.load(open(vocab_path))
        self.ranks = {}
        with open(merges_path) as f:
            for i, ln in enumerate(f.read().split("\n")[1:]):
                p = ln.strip().split(" ")
                if len(p) == 2:
                    self.ranks[(p[0], p[1])] = i
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_to_unicode = {b: chr(c) for b, c in zip(bs, cs)}
        # letters+ | single digit | punctuation runs, like the Kotlin/open_clip regex
        self.piece = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+", re.UNICODE)

    def _bpe(self, token):
        word = list(token[:-4])
        if not word:
            return [token]
        word[-1] = word[-1] + "</w>"
        while len(word) > 1:
            best, best_rank = None, 1 << 30
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and r < best_rank:
                    best_rank, best = r, (word[i], word[i + 1])
            if best is None:
                break
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best[0] and word[i + 1] == best[1]:
                    merged.append(best[0] + best[1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        return word

    def encode(self, text):
        clean = " ".join(text.lower().strip().split())
        ids = []
        for m in self.piece.findall(clean):
            mapped = "".join(self.byte_to_unicode[b] for b in m.encode("utf-8")) + "</w>"
            for t in self._bpe(mapped):
                if t in self.encoder:
                    ids.append(self.encoder[t])
                else:
                    ids.extend(self.encoder[ch] for ch in t if ch in self.encoder)
        out = [0] * self.MAX_LEN
        out[0] = self.BOS
        n = min(len(ids), self.MAX_LEN - 2)
        out[1:1 + n] = ids[:n]
        out[n + 1] = self.EOT
        return out


# ============================================================ multiplex state
PAD, REMOVED = -1, -1116


class MultiplexState:
    def __init__(self, assignments, capacity, object_ids):
        self.capacity = capacity
        self._init(assignments, object_ids)

    def _init(self, assignments, object_ids):
        self.assignments = assignments
        self.num_buckets = len(assignments)
        self.mux_count = len(assignments[0])
        self.total_valid = sum(1 for b in assignments for x in b if x >= 0)
        self.total_non_padding = sum(1 for b in assignments for x in b if x != PAD)
        self.object_ids = object_ids
        self.slot_of = {}
        for bi, b in enumerate(assignments):
            for si, o in enumerate(b):
                if o >= 0:
                    self.slot_of[o] = (bi, si)

    @property
    def available_slots(self):
        return self.num_buckets * self.capacity - self.total_non_padding

    def mux(self, x):
        out = np.zeros((self.num_buckets, self.mux_count) + x.shape[1:], dtype=np.float32)
        for o, (bi, si) in self.slot_of.items():
            out[bi, si] = x[o]
        return out

    def demux(self, x):
        out = np.zeros((self.total_valid,) + x.shape[2:], dtype=np.float32)
        for o, (bi, si) in self.slot_of.items():
            out[o] = x[bi, si]
        return out

    def valid_mask(self):
        m = np.zeros((self.num_buckets, self.mux_count), dtype=np.float32)
        for o, (bi, si) in self.slot_of.items():
            m[bi, si] = 1.0
        return m

    def add_objects(self, object_indices, object_ids):
        rem_idx = list(object_indices)
        rem_ids = list(object_ids)
        assert rem_idx == sorted(rem_idx)
        for b in self.assignments:
            for i in range(self.capacity):
                if not rem_idx:
                    break
                if b[i] == PAD:
                    b[i] = rem_idx.pop(0)
                    self.object_ids.append(rem_ids.pop(0))
            if not rem_idx:
                break
        while rem_idx:
            nb = [PAD] * self.mux_count
            for i in range(self.capacity):
                if not rem_idx:
                    break
                nb[i] = rem_idx.pop(0)
                self.object_ids.append(rem_ids.pop(0))
            self.assignments.append(nb)
        self._init(self.assignments, self.object_ids)

    def remove_objects(self, object_indices):
        object_indices = list(object_indices)
        for b in self.assignments:
            for si, o in enumerate(b):
                if o in object_indices:
                    b[si] = REMOVED
                    object_indices.remove(o)
        self.assignments = [b for b in self.assignments
                            if not all(o in (PAD, REMOVED) for o in b)]
        if not self.assignments:
            self.object_ids = []
            return
        pos = sorted({o for b in self.assignments for o in b if o >= 0})
        remap = {old: new for new, old in enumerate(pos)}
        for b in self.assignments:
            for i, o in enumerate(b):
                if o >= 0:
                    b[i] = remap[o]
        new_ids = [None] * len(pos)
        for old, new in remap.items():
            new_ids[new] = self.object_ids[old]
        self.object_ids = new_ids
        self._init(self.assignments, self.object_ids)


# ============================================================ detection helpers
def load_frames(clip_dir):
    names = [p for p in os.listdir(clip_dir) if os.path.splitext(p)[-1].lower()
             in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frames, H, W = [], None, None
    for nme in names:
        img = Image.open(os.path.join(clip_dir, nme)).convert("RGB")
        W, H = img.width, img.height
        img = img.resize((IMG, IMG), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1).astype(np.float16)
        x = ((x - np.float16(0.5)) / np.float16(0.5)).astype(np.float16)
        frames.append(x)
    return frames, H, W


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def pairwise_iom_rowarea(masks_bool):
    """perflib pairwise_iom quirk: with identical mask sets, min(area_i, area_j) broadcasts
    elementwise (N,1)-vs-(N,1) == the ROW mask's own area."""
    n = masks_bool.shape[0]
    flat = masks_bool.reshape(n, -1).astype(np.float32)
    inter = flat @ flat.T
    area = flat.sum(axis=1, keepdims=True)
    return inter / (area + 1e-8)


def generic_nms_mask(ious, scores, is_valid, iou_threshold):
    """Vectorized NMS replica; already-suppressed rows still suppress lower-scored rows."""
    order = np.argsort(-scores, kind="stable")
    ious_sorted = ious[order][:, order]
    thr = ious_sorted > iou_threshold
    keep = is_valid[order].copy()
    n = len(scores)
    tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    for i in range(n):
        keep = np.where(thr[i] & tri[i], False, keep)
    out = np.zeros_like(keep)
    out[order] = keep
    return out


def sort_perm_first_then_desc(keep_bool):
    """Deterministic stand-in for torch's unstable bool argsort(descending=True):
    kept rows [lowest, then remaining descending], then the rest ascending.
    Verified against torch on every frame of clip8/clip24."""
    t = np.where(keep_bool)[0]
    f = np.where(~keep_bool)[0]
    if len(t) > 1:
        t = np.concatenate([[t[0]], t[1:][::-1]])
    return np.concatenate([t, f]).astype(np.int64)


def mask_iom_true(a_bool, b_bool):
    fa = a_bool.reshape(a_bool.shape[0], -1).astype(np.float32)
    fb = b_bool.reshape(b_bool.shape[0], -1).astype(np.float32)
    inter = fa @ fb.T
    min_area = np.minimum(fa.sum(axis=1)[:, None], fb.sum(axis=1)[None, :])
    return (inter / (min_area + 1e-8)).astype(np.float32)


def mask_iou_mat(a_bool, b_bool):
    fa = a_bool.reshape(a_bool.shape[0], -1).astype(np.float32)
    fb = b_bool.reshape(b_bool.shape[0], -1).astype(np.float32)
    inter = fa @ fb.T
    union = fa.sum(axis=1)[:, None] + fb.sum(axis=1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


def apply_non_overlapping(masks):
    """Keep only the argmax object per pixel (float mask logits, (N,1,H,W))."""
    if masks.shape[0] <= 1:
        return masks
    arg = np.argmax(masks[:, 0], axis=0)
    keep = arg[None] == np.arange(masks.shape[0])[:, None, None]
    return np.where(keep[:, None], masks, np.minimum(masks, -10.0))


def suppress_pw_area_shrinkage(masks):
    if masks.shape[0] <= 1:
        return masks
    arg = np.argmax(masks[:, 0], axis=0)
    keep = arg[None] == np.arange(masks.shape[0])[:, None, None]
    pw = np.where(keep[:, None], masks, np.minimum(masks, -10.0))
    area_before = np.maximum((masks > 0).sum(axis=(2, 3)), 1.0)
    ratio = (pw > 0).sum(axis=(2, 3)) / area_before
    keep_obj = ratio >= 0.3
    return np.where(keep_obj[:, :, None, None], masks, np.minimum(masks, -10.0))


def obj_wise_non_overlap(masks_bool, scores):
    if masks_bool.shape[0] <= 1:
        return masks_bool
    single = np.where(masks_bool, scores[:, None, None], 0.0).astype(np.float32)
    arg = np.argmax(single, axis=0)
    keep = arg[None] == np.arange(masks_bool.shape[0])[:, None, None]
    pixel = np.where(keep, single, np.minimum(single, -10.0))
    return np.where(pixel > 0, masks_bool, False)


# ============================================================ graph bundle
class Graphs:
    def __init__(self, f32=True):
        n_vis = sum(c * hw * hw for _, c, hw in VIS_LAYOUT)
        self.offs = {}
        o = 0
        for name, c, hw in VIS_LAYOUT:
            self.offs[name] = (o, c, hw)
            o += c * hw * hw
        self.vis = GpuGraph(os.path.join(OUT, "sam3_vision_tri.tflite"), f32, n_vis)
        self.text = GpuGraph(os.path.join(OUT, "sam3_text.tflite"), f32, 32 * 256)
        self.head = GpuGraph(os.path.join(OUT, "sam3_head.tflite"), f32, 1001 + 200 * MASK * MASK)
        self.memattn = GpuGraph(os.path.join(PRE, "trk_memattn_n7.tflite"), f32, L * C)
        self.maskdec = GpuGraph(os.path.join(PRE, "trk_maskdec.tflite"), f32,
                                16 * 3 * MASK * MASK + 16 * 3 + 16 + 16 * 3 * 256)
        self.memenc = GpuGraph(os.path.join(PRE, "trk_memenc.tflite"), f32, C * HW72)
        self.initdec = GpuGraph(os.path.join(PRE, "trk_initdec.tflite"), f32,
                                MASK * MASK + 1 + 256 + 1)

    def all(self):
        return [self.vis, self.text, self.head, self.memattn, self.maskdec,
                self.memenc, self.initdec]


# ============================================================ the loop
class Loop:
    def __init__(self, g, num_frames, H, W):
        self.g = g
        self.num_frames = num_frames
        self.H, self.W = H, W
        self.states = []
        self.pos72_flat = CONSTS["pos_72"].reshape(C, HW72).T.copy()      # (5184,256)
        self.memenc_pos_flat = CONSTS["memenc_pos_72"].reshape(C, HW72).T.copy()
        self.tpos = CONSTS["maskmem_tpos_enc"][:, 0, 0, :]                 # (7,256)
        self.meta = {
            "obj_ids_all": np.array([], np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "sam2_score_frame": {},
            "gpu": {"N": 0},
            "removed_obj_ids": set(),
            "conf_status": np.array([], np.int64),
            "conf_cnt": np.array([], np.int64),
        }
        self.cur_feats = None
        self.cur_image_features = None

    # ---------------- frame-level graph runs
    def run_vision(self, frame_fp16):
        y = self.g.vis(frame_fp16.astype(np.float32)[None])
        feats = {}
        for name, (o, c, hw) in self.g.offs.items():
            feats[name] = y[o:o + c * hw * hw].reshape(1, c, hw, hw)
        return feats

    def run_detection(self, feats, text_mem, pad):
        head_in = np.concatenate([feats["sam3_fpn288"].ravel(), feats["sam3_fpn144"].ravel(),
                                  feats["sam3_fpn72"].ravel(), text_mem.ravel(), pad])
        y = self.g.head(head_in)
        logits = y[:200].astype(np.float32)
        boxes_xyxy = box_cxcywh_to_xyxy(y[200:1000].reshape(200, 4).astype(np.float32))
        masks = y[1001:].reshape(200, MASK, MASK).astype(np.float32)
        probs0 = sigmoid(logits)
        is_valid = probs0 > FLAGS["score_threshold_detection"]
        ious = pairwise_iom_rowarea(masks > 0)
        keep = generic_nms_mask(ious, probs0, is_valid, FLAGS["det_nms_thresh"])
        logits = logits - 1e4 * (~keep).astype(np.float32)
        probs = sigmoid(logits)
        pos = probs > FLAGS["score_threshold_detection"]
        if FLAGS["suppress_det_close_to_boundary"]:
            xc = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
            yc = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
            pos = pos & (xc > 0.025) & (xc < 0.975) & (yc > 0.025) & (yc < 0.975)
        perm = sort_perm_first_then_desc(pos)
        return {"scores": probs[perm], "bbox": boxes_xyxy[perm], "mask": masks[perm],
                "keep": pos[perm]}

    # ---------------- interactive (mask-as-output) path
    def _dense_embed(self, mask_1152):
        x = conv_stride_eq_kernel(mask_1152, CONSTS["interactive_mask_downsample.w"],
                                  CONSTS["interactive_mask_downsample.b"], 4)
        x = conv_stride_eq_kernel(x, CONSTS["mask_downscaling.0.w"], CONSTS["mask_downscaling.0.b"], 2)
        x = gelu(layernorm2d(x, CONSTS["mask_downscaling.1.w"], CONSTS["mask_downscaling.1.b"]))
        x = conv_stride_eq_kernel(x, CONSTS["mask_downscaling.3.w"], CONSTS["mask_downscaling.3.b"], 2)
        x = gelu(layernorm2d(x, CONSTS["mask_downscaling.4.w"], CONSTS["mask_downscaling.4.b"]))
        w6 = CONSTS["mask_downscaling.6.w"][:, :, 0, 0]
        return (np.einsum("nchw,oc->nohw", x, w6, optimize=True)
                + CONSTS["mask_downscaling.6.b"][None, :, None, None]).astype(np.float32)

    def use_mask_as_output(self, feats, masks_1152_bool):
        n = masks_1152_bool.shape[0]
        mask_f = masks_1152_bool.astype(np.float32)[:, None]
        high_res = mask_f * 20.0 - 10.0
        low_res = interp_bilinear_aa(high_res, MASK, MASK)
        pix = feats["inter_f2"][0] + CONSTS["interactivity_no_mem_embed"][0, 0][:, None, None]
        dense = self._dense_embed(mask_f)
        sparse = CONSTS["sparse_const"]
        tokens, osl_g = [], []
        for i in range(n):
            x = np.concatenate([pix.ravel(), feats["inter_h0"].ravel(), feats["inter_h1"].ravel(),
                                sparse.ravel(), dense[i].ravel()])
            y = self.g.initdec(x)
            o = MASK * MASK
            tokens.append(y[o + 1:o + 257].astype(np.float32))
            osl_g.append(float(y[o + 257]))
        tokens = np.stack(tokens)
        osl_g = np.array(osl_g, dtype=np.float32)[:, None]
        ptr = mlp3(tokens, "interactive_obj_ptr_proj")
        lam = (osl_g > FLAGS["object_score_logit_threshold"]).astype(np.float32)
        ptr = lam * ptr + (1 - lam) * linear(ptr, CONSTS["no_obj_ptr_linear.w"],
                                             CONSTS["no_obj_ptr_linear.b"])
        lam_m = masks_1152_bool.reshape(n, -1).any(axis=1).astype(np.float32)[:, None]
        osl = 20.0 * lam_m - 10.0
        ptr = lam_m * ptr + (1 - lam_m) * linear(ptr, CONSTS["no_obj_ptr_linear.w"],
                                                 CONSTS["no_obj_ptr_linear.b"])
        return {"low_res_masks": low_res, "object_score_logits": osl.astype(np.float32),
                "obj_ptr": ptr}

    # ---------------- propagation
    def select_cond(self, cond, frame_idx):
        max_cond = FLAGS["max_cond_frames_in_attn"]
        if max_cond == -1 or len(cond) <= max_cond:
            return dict(cond), {}
        selected = {}
        before = max((t for t in cond if t < frame_idx), default=None)
        if before is not None:
            selected[before] = cond[before]
        after = min((t for t in cond if t >= frame_idx), default=None)
        if after is not None:
            selected[after] = cond[after]
        rest = sorted((t for t in cond if t not in selected), key=lambda x: abs(x - frame_idx))
        for t in rest[:max_cond - len(selected)]:
            selected[t] = cond[t]
        return selected, {t: v for t, v in cond.items() if t not in selected}

    def memory_conditioned_features(self, frame_idx, state):
        cond = state["output_cond"]
        non_cond = state["output_non_cond"]
        selected, unselected = self.select_cond(cond, frame_idx)
        t_pos_and_prevs = [((frame_idx - t), out, True) for t, out in selected.items()]
        for t_pos in range(1, 7):
            prev_idx = frame_idx - (7 - t_pos)
            out = non_cond.get(prev_idx)
            if out is None:
                out = unselected.get(prev_idx)
            t_pos_and_prevs.append((t_pos, out, False))

        mem_slots, mem_pos_slots, mi_slots, mi_pos_slots = [], [], [], []
        for t_pos, prev, _sel in t_pos_and_prevs:
            if prev is None or prev.get("maskmem_features") is None:
                continue
            tpos_enc = self.tpos[6] if (t_pos <= 0 or t_pos >= 7) else self.tpos[7 - t_pos - 1]
            f = prev["maskmem_features"]                             # (nb,256,72,72)
            mem_slots.append(f.reshape(f.shape[0], C, HW72).transpose(2, 0, 1))  # (5184,nb,256)
            mem_pos_slots.append(self.memenc_pos_flat + tpos_enc[None, :])
            mi_slots.append(prev["image_features"])                  # (5184,256)
            mi_pos_slots.append(prev["image_pos_enc"] + tpos_enc[None, :])

        nb = state["mux"].num_buckets
        if not mem_slots:
            return np.repeat(state["cur_prop_f2"], nb, axis=0)

        max_ptr = min(self.num_frames, FLAGS["max_obj_ptrs_in_encoder"])
        pos_and_outs = [((frame_idx - t), out) for t, out in selected.items()]
        for t_diff in range(1, max_ptr):
            t = frame_idx - t_diff
            if t < 0:
                break
            out = non_cond.get(t, unselected.get(t))
            if out is not None:
                pos_and_outs.append((t_diff, out))
        pos_and_outs = [(p, o) for p, o in pos_and_outs if o.get("obj_ptr") is not None]
        if pos_and_outs:
            pos_list = np.array([p for p, _ in pos_and_outs], dtype=np.float32)
            ptrs = np.concatenate([o["obj_ptr"] for _, o in pos_and_outs], axis=1)  # (nb,p16,256)
            obj_pos = get_1d_sine_pe(pos_list / (max_ptr - 1))
            obj_pos = linear(obj_pos, CONSTS["obj_ptr_tpos_proj.w"], CONSTS["obj_ptr_tpos_proj.b"])
            obj_pos = np.repeat(obj_pos, 16, axis=0)
            p16 = ptrs.shape[1]
        else:
            ptrs = np.zeros((nb, 0, C), dtype=np.float32)
            obj_pos = np.zeros((0, C), dtype=np.float32)
            p16 = 0

        n = len(mem_slots)
        n_tok = n * L
        pix_flat = state["cur_prop_f2"].reshape(C, HW72).T
        mi = np.zeros((7 * L, C), dtype=np.float32)
        mi[:n_tok] = np.concatenate(mi_slots, axis=0)
        mip = np.zeros((7 * L, C), dtype=np.float32)
        mip[:n_tok] = np.concatenate(mi_pos_slots, axis=0)
        keepv = np.zeros(7 * L + 256, dtype=np.float32)
        keepv[:n_tok] = 1.0
        keepv[7 * L:7 * L + p16] = 1.0
        ptr_pos = np.zeros((256, C), dtype=np.float32)
        ptr_pos[:p16] = obj_pos
        out_buckets = []
        for b in range(nb):
            mm = np.zeros((7 * L, C), dtype=np.float32)
            mm[:n_tok] = np.concatenate([s[:, b, :] for s in mem_slots], axis=0)
            ptr = np.zeros((256, C), dtype=np.float32)
            if p16:
                ptr[:p16] = ptrs[b]
            x = np.concatenate([pix_flat.ravel(), mi.ravel(), mip.ravel(), mm.ravel(),
                                ptr.ravel(), ptr_pos.ravel(), keepv])
            y = self.g.memattn(x)
            out_buckets.append(y.reshape(HW72, C).T.reshape(C, 72, 72))
        return np.stack(out_buckets)

    def forward_sam_heads_prop(self, pix_with_mem, feats, state):
        mux = state["mux"]
        valid = mux.valid_mask()
        merged = (valid[..., None] * CONSTS["output_valid_embed"][None]
                  + (1 - valid[..., None]) * CONSTS["output_invalid_embed"][None]).astype(np.float32)
        masks_b, ious_b, osl_b, tok_b = [], [], [], []
        for b in range(mux.num_buckets):
            x = np.concatenate([pix_with_mem[b].ravel(), feats["prop_h0"].ravel(),
                                feats["prop_h1"].ravel(), merged[b].ravel()])
            y = self.g.maskdec(x)
            o = 16 * 3 * MASK * MASK
            masks_b.append(y[:o].reshape(16, 3, MASK, MASK))
            ious_b.append(y[o:o + 48].reshape(16, 3))
            osl_b.append(y[o + 48:o + 64].reshape(16, 1))
            tok_b.append(y[o + 64:o + 64 + 16 * 3 * 256].reshape(16, 3, 256))
        low_multi = mux.demux(np.stack(masks_b))
        ious = mux.demux(np.stack(ious_b))
        osl = mux.demux(np.stack(osl_b))
        tokens = mux.demux(np.stack(tok_b))
        is_obj = osl > FLAGS["object_score_logit_threshold"]
        low_multi = np.where(is_obj[:, :, None, None], low_multi, NO_OBJ_SCORE)
        best = np.argmax(ious, axis=-1)
        rows = np.arange(low_multi.shape[0])
        low = low_multi[rows, best][:, None]
        token = tokens[rows, best]
        ptr = mlp3(token, "obj_ptr_proj")
        lam = is_obj.astype(np.float32)
        ptr = lam * ptr + (1 - lam) * linear(ptr, CONSTS["no_obj_ptr_linear.w"],
                                             CONSTS["no_obj_ptr_linear.b"])
        return {"low_res_masks": low, "ious": ious, "object_score_logits": osl, "obj_ptr": ptr}

    def encode_new_memory(self, prop_f2, masks_high, osl, cond_objs, mux):
        mask_for_mem = sigmoid(masks_high[:, 0]) * FLAGS["sigmoid_scale_for_mem_enc"] \
            + FLAGS["sigmoid_bias_for_mem_enc"]
        cond_vals = np.full(mask_for_mem.shape[0], FLAGS["condition_bg"], dtype=np.float32)
        for o in sorted(cond_objs):
            cond_vals[o] = FLAGS["condition_fg"]
        mux_mask = mux.mux(mask_for_mem)
        cond_maps = np.broadcast_to(cond_vals[:, None, None], mask_for_mem.shape)
        mux_cond = mux.mux(np.ascontiguousarray(cond_maps))
        x = np.concatenate([mux_mask, mux_cond], axis=1)
        if x.shape[-1] != IMG:
            x = interp_bilinear(x, IMG, IMG)
        feats = []
        for b in range(mux.num_buckets):
            feats.append(self.g.memenc(np.concatenate([prop_f2.ravel(), x[b].ravel()]))
                         .reshape(C, 72, 72))
        feats = np.stack(feats)
        osl_full = osl
        if osl_full.shape[0] != mux.total_valid:
            osl_full = np.concatenate(
                [osl_full, np.zeros((mux.total_valid - osl_full.shape[0], 1), np.float32)])
        osl_mux = mux.mux(osl_full[:, 0])
        is_obj = (osl_mux > FLAGS["object_score_logit_threshold"]).astype(np.float32)
        no_obj = ((1 - is_obj)[..., None] * CONSTS["no_obj_embed_spatial"][None]).sum(axis=1)
        return feats + no_obj[:, :, None, None]

    # ---------------- SAM2-state ops
    def new_state(self):
        return {"mux": None, "obj_id_to_idx": {}, "obj_ids": [],
                "output_cond": {}, "output_non_cond": {},
                "temp_cond": {}, "temp_non_cond": {},
                "frames_already_tracked": set(),
                "consolidated_cond": set(), "consolidated_non_cond": set(),
                "cur_prop_f2": None}

    def _video_res_output(self, masks_288):
        v = interp_bilinear(masks_288, self.H, self.W)
        if FLAGS["non_overlap_masks_for_output"]:
            v = apply_non_overlapping(v)
        return v

    def add_new_masks(self, state, frame_idx, obj_ids, masks_1152_bool, feats,
                      reconditioning=False):
        n = masks_1152_bool.shape[0]
        obj_idxs = []
        for oid in obj_ids:
            if oid in state["obj_id_to_idx"]:
                obj_idxs.append(state["obj_id_to_idx"][oid])
            else:
                assert not reconditioning
                idx = len(state["obj_id_to_idx"])
                state["obj_id_to_idx"][oid] = idx
                state["obj_ids"] = list(state["obj_id_to_idx"])
                obj_idxs.append(idx)
        video = interp_bilinear_aa(masks_1152_bool.astype(np.float32)[:, None],
                                   self.H, self.W) > 0.5                # (n,1,H,W)
        is_new_state = state["mux"] is None
        if not reconditioning and is_new_state:
            cap = FLAGS["multiplex_count"]
            nb = (n + cap - 1) // cap
            ids = list(range(n)) + [PAD] * (nb * cap - n)
            assignments = [ids[i * cap:(i + 1) * cap] for i in range(nb)]
            state["mux"] = MultiplexState(assignments, cap, list(map(int, obj_ids)))
        is_init_cond = frame_idx not in state["frames_already_tracked"]
        is_cond = is_init_cond
        storage = "output_cond" if is_cond else "output_non_cond"
        tstore = "temp_cond" if is_cond else "temp_non_cond"

        mask_out = self.use_mask_as_output(feats, masks_1152_bool)
        if reconditioning or not is_new_state:
            existing = state["output_cond"].get(frame_idx) or state["output_non_cond"].get(frame_idx)
            assert existing is not None
            low = mask_out["low_res_masks"]
            if low.shape[-1] != existing["pred_masks"].shape[-1]:
                low = interp_bilinear_aa(low, existing["pred_masks"].shape[-2],
                                         existing["pred_masks"].shape[-1])
            if reconditioning:
                for j, oi in enumerate(obj_idxs):
                    existing["pred_masks"][oi] = low[j]
                    existing["object_score_logits"][oi] = mask_out["object_score_logits"][j]
                ptr = state["mux"].demux(existing["obj_ptr"])
                for j, oi in enumerate(obj_idxs):
                    ptr[oi] = mask_out["obj_ptr"][j]
                existing["obj_ptr"] = state["mux"].mux(ptr)
                existing["conditioning_objects"].update(obj_idxs)
            else:
                mux = state["mux"]
                old_ptr = mux.demux(existing["obj_ptr"])
                start = mux.total_valid
                mux.add_objects(list(range(start, start + n)), list(map(int, obj_ids)))
                existing["pred_masks"] = np.concatenate([existing["pred_masks"], low])
                existing["object_score_logits"] = np.concatenate(
                    [existing["object_score_logits"], mask_out["object_score_logits"]])
                existing["obj_ptr"] = mux.mux(np.concatenate([old_ptr, mask_out["obj_ptr"]]))
                existing["conditioning_objects"].update(range(start, start + n))
            current = existing
            # pred_masks_video_res on the existing entry (for consolidation base)
            vres = self._video_res_output(existing["pred_masks"])
            for j, oi in enumerate(obj_idxs):
                vres[oi] = np.where(video[j], -NO_OBJ_SCORE, NO_OBJ_SCORE)
            current["pred_masks_video_res"] = vres
        else:
            current = {"pred_masks": mask_out["low_res_masks"],
                       "object_score_logits": mask_out["object_score_logits"],
                       "obj_ptr": state["mux"].mux(mask_out["obj_ptr"]),
                       "conditioning_objects": set(obj_idxs),
                       "maskmem_features": None,
                       "image_features": self.cur_image_features,
                       "image_pos_enc": self.pos72_flat}
            vres = self._video_res_output(current["pred_masks"])
            for j, oi in enumerate(obj_idxs):
                vres[oi] = np.where(video[j], -NO_OBJ_SCORE, NO_OBJ_SCORE)
            current["pred_masks_video_res"] = vres

        if is_cond and frame_idx in state["output_non_cond"]:
            del state["output_non_cond"][frame_idx]
            state["consolidated_non_cond"].discard(frame_idx)
        state[storage][frame_idx] = current
        state["consolidated_cond" if is_cond else "consolidated_non_cond"].add(frame_idx)

        # per-object temp entries (video res) with cross-suppression among the new masks
        video_bin = video[:, 0]                                       # (n,H,W)
        for j, oi in enumerate(obj_idxs):
            m = np.where(video_bin[j], -NO_OBJ_SCORE, NO_OBJ_SCORE).astype(np.float32)
            if n > 1:
                others = np.concatenate([video_bin[:j], video_bin[j + 1:]]).any(axis=0)
                m = np.where(others, NO_OBJ_SCORE, m)
            state[tstore].setdefault(oi, {})[frame_idx] = m
        # existing (non-new) objects at this frame suppressed by the new masks
        combined = video_bin.any(axis=0)
        for oi2, d in state[tstore].items():
            if oi2 in obj_idxs or frame_idx not in d:
                continue
            d[frame_idx] = np.where(combined, NO_OBJ_SCORE, d[frame_idx])

    def preflight(self, state, feats):
        state["tracking_has_started"] = True
        nobj = state["mux"].total_valid
        for is_cond in (False, True):
            tstore = "temp_cond" if is_cond else "temp_non_cond"
            storage = "output_cond" if is_cond else "output_non_cond"
            frames = set()
            for d in state[tstore].values():
                frames.update(d.keys())
            state["consolidated_cond" if is_cond else "consolidated_non_cond"].update(frames)
            for f in frames:
                all_out = state["output_cond"].get(f) or state["output_non_cond"].get(f)
                assert all_out is not None
                base = all_out.get("pred_masks_video_res")
                if base is not None:
                    cons = interp_bilinear_aa(base, MASK, MASK)
                else:
                    cons = all_out["pred_masks"].copy()
                for oi in range(nobj):
                    src = state[tstore].get(oi, {}).get(f)
                    if src is not None:
                        cons[oi] = interp_bilinear_aa(src[None, None], MASK, MASK)[0]
                    # else: fall back to per-object stored 288 mask (already in cons rows
                    # only when base is None); with a video-res base the official code
                    # falls back to the per-object stored pred_masks slice:
                    elif base is not None:
                        cons[oi] = all_out["pred_masks"][oi]
                high = apply_non_overlapping(interp_bilinear(cons, IMG, IMG))
                feats_mem = self.encode_new_memory(state["cur_prop_f2"], high,
                                                   all_out["object_score_logits"],
                                                   all_out["conditioning_objects"], state["mux"])
                consolidated = {"pred_masks": cons,
                                "object_score_logits": all_out["object_score_logits"],
                                "obj_ptr": all_out["obj_ptr"],
                                "conditioning_objects": all_out["conditioning_objects"],
                                "maskmem_features": bf16(feats_mem),
                                "image_features": self.cur_image_features,
                                "image_pos_enc": self.pos72_flat}
                state[storage][f] = consolidated
            for d in state[tstore].values():
                d.clear()
        for f in list(state["output_cond"]):
            state["output_non_cond"].pop(f, None)
            state["consolidated_non_cond"].discard(f)

    def propagate_state_one_frame(self, state, frame_idx, feats):
        if frame_idx in state["consolidated_cond"]:
            cur = state["output_cond"][frame_idx]
        elif frame_idx in state["consolidated_non_cond"]:
            cur = state["output_non_cond"][frame_idx]
        else:
            pix = self.memory_conditioned_features(frame_idx, state)
            out = self.forward_sam_heads_prop(pix, feats, state)
            cur = {"pred_masks": out["low_res_masks"],
                   "object_score_logits": out["object_score_logits"],
                   "obj_ptr": state["mux"].mux(out["obj_ptr"]),
                   "conditioning_objects": set(),
                   "maskmem_features": None,
                   "image_features": self.cur_image_features,
                   "image_pos_enc": self.pos72_flat}
            state["output_non_cond"][frame_idx] = cur
        state["frames_already_tracked"].add(frame_idx)
        return (list(state["obj_ids"]), cur["pred_masks"][:, 0].copy(),
                cur["object_score_logits"][:, 0].copy())

    # ---------------- planning
    def associate(self, det_out, trk_masks, trk_obj_ids):
        det_scores, det_keep, det_masks = det_out["scores"], det_out["keep"], det_out["mask"]
        if trk_masks.shape[0] == 0:
            is_new = det_scores >= FLAGS["new_det_thresh"]
            return {"trk_is_unmatched": np.zeros(0, bool), "is_new_det": is_new,
                    "im_mask": np.zeros((200, 0), bool), "hi_conf": {}, "det_matched": {}}
        det_bin = (det_masks > 0) & det_keep[:, None, None]
        trk_bin = trk_masks > 0
        metric = mask_iom_true(det_bin, trk_bin)
        trk_is_matched = (metric >= FLAGS["trk_assoc_iou_thresh"]).any(axis=0)
        trk_is_nonempty = trk_bin.any(axis=(1, 2))
        trk_is_unmatched = trk_is_nonempty & ~trk_is_matched
        is_new = (det_scores >= FLAGS["new_det_thresh"]) & det_keep \
            & ~(metric >= FLAGS["assoc_iou_thresh"]).any(axis=1)
        thr_r = FLAGS["iom_thresh_recondition"]
        det_many = (metric >= thr_r).sum(axis=1) > 1
        trk_many = (metric >= thr_r).sum(axis=0) > 1
        metric_z = np.where(trk_many[None, :], 0.0, metric)
        metric_z = np.where(det_many[:, None], 0.0, metric_z)
        det_to_max = np.argmax(metric_z, axis=1)
        det_hi = (det_scores >= 0.8) & det_keep & ~is_new & (metric_z.max(axis=1) >= thr_r)
        im_mask = metric_z >= FLAGS["assoc_iou_thresh"]
        det_matched, hi_conf = {}, {}
        ids = np.asarray(trk_obj_ids)
        for d in range(200):
            if det_keep[d]:
                det_matched[d] = ids[im_mask[d]]
                if det_hi[d]:
                    hi_conf[int(ids[det_to_max[d]])] = d
        return {"trk_is_unmatched": trk_is_unmatched, "is_new_det": is_new,
                "im_mask": im_mask, "hi_conf": hi_conf, "det_matched": det_matched}

    def process_hotstart(self, frame_idx, adt):
        g = self.meta["gpu"]
        N = adt["im_mask"].shape[1]
        if N == 0:
            return np.zeros(0, bool)
        assert g["N"] == N
        matched = adt["im_mask"].any(axis=0)
        g["keep_alive"] = np.clip(np.where(matched, g["keep_alive"] + 1, g["keep_alive"] - 1),
                                  FLAGS["min_trk_keep_alive"], FLAGS["max_trk_keep_alive"])
        g["unmatch_cnt"] = np.where(adt["trk_is_unmatched"], g["unmatch_cnt"] + 1, g["unmatch_cnt"])
        multi = adt["im_mask"] & (adt["im_mask"].sum(axis=1) > 1)[:, None]
        inc = multi.astype(np.int64).T @ multi.astype(np.int64)
        np.fill_diagonal(inc, 0)
        g["overlap"] = g["overlap"] + np.triu(inc, k=1)
        within = g["first_frame"] > (frame_idx - FLAGS["hotstart_delay"])
        rm_unmatch = within & (g["unmatch_cnt"] >= FLAGS["hotstart_unmatch_thresh"]) & ~g["removed"]
        earlier = g["first_frame"][:, None] < g["first_frame"][None, :]
        max_ov = np.where(earlier, g["overlap"], 0).max(axis=0)
        rm_overlap = within & (max_ov >= FLAGS["hotstart_dup_thresh"]) & ~g["removed"]
        to_remove = rm_unmatch | rm_overlap
        g["removed"] = g["removed"] | to_remove
        return to_remove

    def suppress_overlapping_occl(self, frame_idx, trk_masks, to_remove):
        g = self.meta["gpu"]
        binm = trk_masks > 0
        last = np.where(to_remove, 100000, g["last_occl"])
        sup = np.zeros(trk_masks.shape[0], bool)
        if trk_masks.shape[0] > 1:
            iou = mask_iou_mat(binm, binm)
            pairs = np.triu(iou >= FLAGS["suppress_overlap_recent_occl_thresh"], k=1)
            li, lj = last[:, None], last[None, :]
            sup = (pairs & (li > lj) & (lj > -1)).any(axis=1) \
                | (pairs & (lj > li) & (li > -1)).any(axis=0)
        occluded = ~binm.any(axis=(1, 2))
        new_last = last.copy()
        new_last[occluded | sup] = frame_idx
        g["last_occl"] = new_last
        trk_masks[sup] = -10.0
        return trk_masks

    def update_memories(self, frame_idx, trk_masks):
        high = suppress_pw_area_shrinkage(interp_bilinear(trk_masks[:, None], INMASK, INMASK))
        osl = np.where((high > 0).any(axis=(2, 3)), 10.0, -10.0).astype(np.float32)
        all_ids, state_of = [], {}
        for si, st in enumerate(self.states):
            for oid in st["obj_ids"]:
                state_of[oid] = si
                all_ids.append(oid)
        order = sorted(range(len(all_ids)), key=lambda i: all_ids[i])
        assign = {}
        for gpos, li in enumerate(order):
            assign.setdefault(state_of[all_ids[li]], []).append(gpos)
        for si, st in enumerate(self.states):
            if not st["obj_ids"]:
                continue
            idxs = assign[si]
            entry = st["output_cond"].get(frame_idx) or st["output_non_cond"].get(frame_idx)
            cond_objs = entry["conditioning_objects"] if entry is not None else set()
            feats_mem = self.encode_new_memory(st["cur_prop_f2"], high[idxs], osl[idxs],
                                               cond_objs, st["mux"])
            if entry is not None:
                entry["maskmem_features"] = bf16(feats_mem)
                entry["image_features"] = self.cur_image_features
                entry["image_pos_enc"] = self.pos72_flat

    def recondition(self, frame_idx, det_out, adt, trk_masks, trk_scores, feats):
        recond_ids = set()
        hi = adt["hi_conf"]
        ids_all = list(self.meta["obj_ids_all"])
        trk_ids = list(hi.keys())
        det_idx = [hi[t] for t in trk_ids]
        obj_pos = [ids_all.index(t) for t in trk_ids]
        conf = sigmoid(trk_scores[obj_pos]) > 0.8
        val = [(t, d, p) for t, d, p, c in zip(trk_ids, det_idx, obj_pos, conf) if c]
        if not val:
            return recond_ids
        val_t = [t for t, _, _ in val]
        val_d = [d for _, d, _ in val]
        val_p = [p for _, _, p in val]
        new_masks = det_out["mask"][val_d]
        new_bin_1152 = interp_bilinear(new_masks[:, None], INMASK, INMASK)[:, 0] > 0
        old = trk_masks[val_p]
        agree = (new_masks > 0) == (old > 0)
        trk_masks[val_p] = np.where(agree, old, new_masks)
        for st in self.states:
            pairs = [(t, m) for t, m in zip(val_t, new_bin_1152) if t in st["obj_id_to_idx"]]
            if not pairs:
                continue
            self.add_new_masks(st, frame_idx, [t for t, _ in pairs],
                               np.stack([m for _, m in pairs]), feats, reconditioning=True)
            recond_ids.update(st["obj_ids"])
            self.preflight(st, feats)
        return recond_ids

    def update_confirmation(self, prev_ids, new_ids_all, adt, new_det_ids):
        meta = self.meta
        status = np.full(len(new_ids_all), 1, np.int64)
        cnt = np.zeros(len(new_ids_all), np.int64)
        pos = {int(o): i for i, o in enumerate(new_ids_all)}
        for i, o in enumerate(prev_ids):
            j = pos.get(int(o))
            if j is not None:
                status[j] = meta["conf_status"][i]
                cnt[j] = meta["conf_cnt"][i]
        matched = set(int(x) for x in new_det_ids)
        for ids in adt["det_matched"].values():
            matched.update(int(x) for x in ids)
        for j, o in enumerate(new_ids_all):
            cnt[j] = cnt[j] + 1 if int(o) in matched else 0
            if cnt[j] >= FLAGS["masklet_confirmation_consecutive_det_thresh"]:
                status[j] = 2
        meta["conf_status"] = status
        meta["conf_cnt"] = cnt

    # ---------------- execution
    def add_objects_execution(self, frame_idx, det_out, new_fa, new_ids, feats):
        masks_1152 = interp_bilinear(det_out["mask"][new_fa][:, None], INMASK, INMASK)[:, 0] > 0
        best = None
        for st in self.states:
            if st["mux"] is None:
                continue
            av = st["mux"].available_slots
            if av >= len(new_fa) and (best is None or av < best["mux"].available_slots):
                best = st
        if best is None:
            best = self.new_state()
            self.states.append(best)
        best["cur_prop_f2"] = feats["prop_f2"]
        self.add_new_masks(best, frame_idx, list(map(int, new_ids)), masks_1152, feats)
        self.preflight(best, feats)

    def remove_objects_execution(self, obj_ids):
        # NOT exercised by the verification clips; simplified port of remove_objects.
        keep_states = []
        for st in self.states:
            idxs = sorted(st["obj_id_to_idx"][o] for o in obj_ids if o in st["obj_id_to_idx"])
            if idxs:
                st["mux"].remove_objects(idxs)
                remove_set = set(idxs)
                old2new, new = {}, 0
                for old in range(len(st["obj_id_to_idx"])):
                    if old not in remove_set:
                        old2new[old] = new
                        new += 1
                st["obj_id_to_idx"] = {oid: old2new[i] for oid, i in st["obj_id_to_idx"].items()
                                       if i in old2new}
                st["obj_ids"] = list(st["obj_id_to_idx"])
                keep_rows = sorted(old2new.keys())
                for key in ("output_cond", "output_non_cond"):
                    for out in st[key].values():
                        for k2 in ("pred_masks", "object_score_logits"):
                            if out.get(k2) is not None:
                                out[k2] = out[k2][keep_rows]
                        out["conditioning_objects"] = {old2new[o] for o in out["conditioning_objects"]
                                                       if o in old2new}
            if st["obj_ids"]:
                keep_states.append(st)
        self.states = keep_states

    # ---------------- one full det+track step
    def det_track_one_frame(self, frame_idx, feats, det_out):
        meta = self.meta
        # Step 2: propagation
        obj_ids_local, low_list, score_list = [], [], []
        for st in self.states:
            if not st["obj_ids"]:
                continue
            st["cur_prop_f2"] = feats["prop_f2"]
            ids, masks, scores = self.propagate_state_one_frame(st, frame_idx, feats)
            obj_ids_local.extend(ids)
            low_list.append(masks)
            score_list.append(scores)
        if low_list:
            trk_masks = np.concatenate(low_list)
            trk_scores = np.concatenate(score_list)
            if obj_ids_local != sorted(obj_ids_local):
                order = sorted(range(len(obj_ids_local)), key=lambda i: obj_ids_local[i])
                obj_ids_local = [obj_ids_local[i] for i in order]
                trk_masks = trk_masks[order]
                trk_scores = trk_scores[order]
        else:
            trk_masks = np.zeros((0, MASK, MASK), np.float32)
            trk_scores = np.zeros((0,), np.float32)
        assert list(meta["obj_ids_all"]) == obj_ids_local

        # Step 3: planning
        adt = self.associate(det_out, trk_masks, meta["obj_ids_all"])
        to_remove = self.process_hotstart(frame_idx, adt)
        recond_ids = set()
        if FLAGS["recondition_every_nth_frame"] > 0 \
                and frame_idx % FLAGS["recondition_every_nth_frame"] == 0 and adt["hi_conf"]:
            recond_ids = self.recondition(frame_idx, det_out, adt, trk_masks, trk_scores, feats)
        if trk_masks.shape[0] > 0:
            trk_masks = self.suppress_overlapping_occl(frame_idx, trk_masks, to_remove)
            self.update_memories(frame_idx, trk_masks)

        det_scores = det_out["scores"]
        new_fa = np.nonzero(adt["is_new_det"])[0]
        prev_n = len(meta["obj_ids_all"])
        if prev_n + len(new_fa) > FLAGS["max_num_objects"]:
            keep_n = max(FLAGS["max_num_objects"] - prev_n, 0)
            order = np.argsort(det_scores[new_fa], kind="stable")[::-1]
            new_fa = new_fa[order[:keep_n]] if keep_n else np.array([], np.int64)
        new_ids = meta["max_obj_id"] + 1 + np.arange(len(new_fa))
        removed_now = set(np.array(meta["obj_ids_all"])[to_remove].tolist()) if len(to_remove) else set()

        prev_ids = meta["obj_ids_all"]
        updated = [int(o) for o in prev_ids if int(o) not in removed_now] + [int(i) for i in new_ids]
        meta["obj_ids_all"] = np.array(updated, np.int64)
        for oid, fa in zip(new_ids, new_fa):
            meta["obj_id_to_score"][int(oid)] = float(det_scores[fa])
            meta["sam2_score_frame"].setdefault(frame_idx, {})[int(oid)] = float(det_scores[fa])
        if len(new_ids):
            meta["max_obj_id"] = int(max(meta["max_obj_id"], int(new_ids.max())))
        for oid in removed_now:
            meta["obj_id_to_score"][int(oid)] = -1e4
            meta["sam2_score_frame"].setdefault(frame_idx, {})[int(oid)] = -1e4
        self.update_confirmation(prev_ids, meta["obj_ids_all"], adt, new_ids)

        g = meta["gpu"]
        if g["N"] > 0:
            keep_idx = np.nonzero(~g["removed"])[0]
            for k in ("first_frame", "unmatch_cnt", "keep_alive", "removed", "last_occl"):
                g[k] = g[k][keep_idx]
            g["overlap"] = g["overlap"][np.ix_(keep_idx, keep_idx)]
            g["N"] = len(keep_idx)
        if len(new_ids):
            nn = len(new_ids)
            g["first_frame"] = np.concatenate([g.get("first_frame", np.zeros(0, np.int64)),
                                               np.full(nn, frame_idx, np.int64)])
            g["unmatch_cnt"] = np.concatenate([g.get("unmatch_cnt", np.zeros(0, np.int64)),
                                               np.zeros(nn, np.int64)])
            g["keep_alive"] = np.concatenate([g.get("keep_alive", np.zeros(0, np.int64)),
                                              np.full(nn, FLAGS["init_trk_keep_alive"], np.int64)])
            g["removed"] = np.concatenate([g.get("removed", np.zeros(0, bool)), np.zeros(nn, bool)])
            g["last_occl"] = np.concatenate([g.get("last_occl", np.zeros(0, np.int64)),
                                             np.full(nn, -1, np.int64)])
            oldN = g["N"]
            ov = np.zeros((oldN + nn, oldN + nn), np.int64)
            if oldN:
                ov[:oldN, :oldN] = g["overlap"]
            g["overlap"] = ov
            g["N"] = oldN + nn
        meta["removed_obj_ids"].update(removed_now)

        # Step 4: execution
        if len(new_fa):
            self.add_objects_execution(frame_idx, det_out, new_fa, new_ids, feats)
        if removed_now:
            self.remove_objects_execution(removed_now)

        # sam2 scores of pre-existing objects for this frame (overwrites removed entries)
        d = meta["sam2_score_frame"].setdefault(frame_idx, {})
        for oid, s in zip(prev_ids, sigmoid(trk_scores)):
            d[int(oid)] = float(s)

        # Step 5: outputs
        obj_id_to_mask = {}
        if trk_masks.shape[0]:
            vid = interp_bilinear(trk_masks[:, None], self.H, self.W) > 0
            for oid, m in zip(prev_ids, vid):
                obj_id_to_mask[int(oid)] = m
        if len(new_fa):
            newv = interp_bilinear(det_out["mask"][new_fa][:, None], self.H, self.W) > 0
            for oid, m in zip(new_ids, newv):
                obj_id_to_mask[int(oid)] = m

        unconfirmed = [int(o) for o, s in zip(meta["obj_ids_all"], meta["conf_status"]) if s == 1]
        return {"obj_id_to_mask": obj_id_to_mask, "removed_now": removed_now,
                "unconfirmed": unconfirmed,
                "sam2_scores": dict(meta["sam2_score_frame"].get(frame_idx, {}))}


# ============================================================ compare & main
def compare(results, ref, tag):
    frames = sorted(ref)
    agree = True
    lines = []
    min_iou, max_dp = 1.0, 0.0
    for f in frames:
        r, o = ref[f], results.get(f)
        if o is None:
            lines.append(f"  f{f}: MISSING")
            agree = False
            continue
        rid, oid = list(r["out_obj_ids"]), list(o["out_obj_ids"])
        same = rid == oid
        agree &= same
        ious = []
        for j, obj in enumerate(rid):
            if obj in oid:
                k = oid.index(obj)
                a, b = r["out_binary_masks"][j] > 0, o["out_binary_masks"][k] > 0
                iou = np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)
                ious.append(iou)
                min_iou = min(min_iou, iou)
        dp = (np.abs(np.asarray(r["out_probs"]) - np.asarray(o["out_probs"])).max()
              if same and len(rid) else float("nan"))
        if same and len(rid):
            max_dp = max(max_dp, dp)
        lines.append(f"  f{f}: ids ref={rid} got={oid} same={same} |dprob|={dp:.4f} "
                     f"maskIoU={'/'.join(f'{v:.3f}' for v in ious)}")
    print(f"[compare {tag}] all-ids-agree={agree} min-maskIoU={min_iou:.4f} max|dprob|={max_dp:.4f}")
    print("\n".join(lines))
    return agree, min_iou, max_dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(ROOT, "models", "clip8"))
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--ref", default=os.path.join(ROOT, "models", "tracker_ref", "ref_person.npy"))
    ap.add_argument("--fp16", action="store_true", help="graphs in delegate fp16 mode (Mali reality)")
    ap.add_argument("--dump-device", default=None,
                    help="write per-frame expected outputs (ids/probs/packed masks) for the on-device autotest")
    a = ap.parse_args()
    t_start = time.time()
    frames, H, W = load_frames(a.clip)
    num_frames = len(frames)
    g = Graphs(f32=not a.fp16)
    loop = Loop(g, num_frames, H, W)

    ids = BpeTokenizer(os.path.join(OUT, "sam3_tokenizer", "vocab.json"),
                       os.path.join(OUT, "sam3_tokenizer", "merges.txt")).encode(a.prompt)
    table = np.fromfile(os.path.join(OUT, "sam3_token_embed.bin"), dtype=np.float16).reshape(-1, 1024)
    emb = table[ids].astype(np.float32)[None]
    text_mem = g.text(emb)
    pad = np.array([1.0 if i == 0 else 0.0 for i in ids], dtype=np.float32)

    unconfirmed_per_frame = {}
    delay = FLAGS["masklet_confirmation_consecutive_det_thresh"] - 1

    def postprocess(frame_idx, out, removed_snapshot):
        m = out["obj_id_to_mask"]
        ids_ = sorted(m.keys())
        if not ids_:
            return {"out_obj_ids": np.zeros(0, np.int64), "out_probs": np.zeros(0, np.float32),
                    "out_binary_masks": np.zeros((0, H, W), bool)}
        probs = np.array([loop.meta["obj_id_to_score"][i] for i in ids_], np.float32)
        sam2 = np.array([out["sam2_scores"].get(i, 0.0) for i in ids_], np.float32)
        masks = np.concatenate([m[i] for i in ids_])
        keep = masks.any(axis=(1, 2))
        f_unc = max(0, min(frame_idx + delay, num_frames - 1))
        hide = set(removed_snapshot) | set(unconfirmed_per_frame.get(f_unc, []))
        if hide:
            keep = keep & ~np.isin(np.array(ids_), sorted(hide))
        ki = np.nonzero(keep)[0]
        masks2 = masks[ki]
        if masks2.shape[0] > 1:
            masks2 = obj_wise_non_overlap(masks2, sam2[ki])
        return {"out_obj_ids": np.array(ids_, np.int64)[ki], "out_probs": probs[ki],
                "out_binary_masks": masks2}

    def run_frame(fi):
        feats = loop.run_vision(frames[fi])
        loop.cur_feats = feats
        loop.cur_image_features = feats["prop_f2"].reshape(C, HW72).T.copy()
        det = loop.run_detection(feats, text_mem, pad)
        return loop.det_track_one_frame(fi, feats, det)

    results = {}
    out0 = run_frame(0)                                   # add_prompt(frame 0)
    results[0] = postprocess(0, out0, set())

    hot_removed = set()
    removed_snapshot_of = {}
    outs = {}
    for fi in range(num_frames):                          # propagate_in_video forward
        out = run_frame(fi)
        outs[fi] = out
        hot_removed.update(out["removed_now"])
        unconfirmed_per_frame[fi] = out["unconfirmed"]
        if fi == num_frames - 1:
            for yf in list(outs.keys()):
                removed_snapshot_of.setdefault(yf, set(hot_removed))
        elif fi >= FLAGS["hotstart_delay"] - 1:
            yf = fi - (FLAGS["hotstart_delay"] - 1)
            removed_snapshot_of[yf] = set(hot_removed)
    for fi in range(num_frames):
        results[fi] = postprocess(fi, outs[fi], removed_snapshot_of.get(fi, set(hot_removed)))

    total = time.time() - t_start
    print(f"[run] {num_frames} frames in {total:.1f}s  " +
          "  ".join(f"{gr.name}: {gr.calls}x/{gr.ms:.0f}ms" for gr in g.all()))
    if a.dump_device:
        os.makedirs(a.dump_device, exist_ok=True)
        for fi in range(num_frames):
            r = results[fi]
            r["out_obj_ids"].astype("<i4").tofile(os.path.join(a.dump_device, f"f{fi}_ids.bin"))
            r["out_probs"].astype("<f4").tofile(os.path.join(a.dump_device, f"f{fi}_probs.bin"))
            packed = np.packbits(r["out_binary_masks"].reshape(len(r["out_obj_ids"]), -1),
                                 axis=-1, bitorder="little")
            packed.tofile(os.path.join(a.dump_device, f"f{fi}_masks.bin"))
        json.dump({"frames": num_frames, "height": H, "width": W, "prompt": a.prompt,
                   "mask_packing": "per object, row-major HxW, 1 bit/px, LSB-first (np.packbits bitorder=little)"},
                  open(os.path.join(a.dump_device, "manifest.json"), "w"), indent=1)
        print(f"[dump-device] wrote {num_frames} frames to {a.dump_device}")
    ref = np.load(a.ref, allow_pickle=True).item()
    ok, min_iou, max_dp = compare(results, ref, os.path.basename(a.clip))
    return 0 if (ok and min_iou >= 0.99) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the SAM 2.1 video-path graphs against the HF PyTorch reference.

The HOST LOOP below (``track``) is the numpy specification of what the Kotlin /
Swift orchestration must do per frame: it only calls the four graph functions
and does the bank bookkeeping, best-mask pick, no-object handling and the
mask_for_mem construction. Any runner (torch wrappers, tflite Interpreter, and
later the on-device CompiledModel) plugs into the same loop.

Modes (run in THIS order; ``ref`` must run in a process that never imports
convert_sam2_video, because that import patches the HF classes):

  python verify_sam2_video.py ref              # HF streaming reference -> ref_nmm{7,2}.npz
  python verify_sam2_video.py torch [--bf16mem]# patched fp32 torch wrappers vs ref
  python verify_sam2_video.py tflite           # fp16 .tflite graphs (Interpreter) vs ref
  python verify_sam2_video.py vectors          # dump per-graph device test vectors (frame T-1)

Synthetic clip: a white disk moving diagonally on black, T frames at 1024x1024,
one positive click on the disk in frame 0. Deterministic, regenerated on demand.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.environ.get("SAM2_OUT", HERE + "/output/video"))
CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
T = int(os.environ.get("SAM2_T", "10"))
IE, F0, F1 = 1048576, 2097152, 1048576
HW, MEMCH, HD, NPTR_FRAMES, PTR_SPLIT = 4096, 64, 256, 16, 4
NPTR = NPTR_FRAMES * PTR_SPLIT
NO_OBJ_SCORE, MEM_SCALE, MEM_BIAS, MASK_NEG = -1024.0, 20.0, -10.0, -1e9
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
CLICK = (400.0, 512.0)  # frame-0 click, in 1024x1024 pixel coords (disk center)


# ----------------------------------------------------------------------------- clip
def frame(t):
    """Normalized CHW float32 frame t: white disk moving (+14,+6) px/frame in 512-space."""
    img = Image.new("RGB", (512, 512), "black")
    cx, cy, r = 200 + 14 * t, 256 + 6 * t, 100
    ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], fill="white")
    arr = np.asarray(img.resize((1024, 1024), Image.BILINEAR)).astype(np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


# ------------------------------------------------------------------- host helpers
def sine_pe_1d(pos, dim=256, temperature=10000.0):
    """get_1d_sine_pe for a scalar position (numpy)."""
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    x = pos / dim_t
    return np.concatenate([np.sin(x), np.cos(x)]).astype(np.float32)


class Consts:
    def __init__(self, d=OUT):
        p = np.fromfile(f"{d}/sam2v_prompt.bin", np.float32)
        self.gauss, self.pe1, self.pe0, self.nap = p[:256], p[256:512], p[512:768], p[768:1024]
        self.track_sparse = np.fromfile(f"{d}/sam2v_track_sparse.bin", np.float32)
        self.mtpe = np.fromfile(f"{d}/sam2v_mtpe.bin", np.float32).reshape(7, 64)
        self.no_obj_ptr = np.fromfile(f"{d}/sam2v_no_obj_ptr.bin", np.float32)
        tp = np.fromfile(f"{d}/sam2v_tpos_proj.bin", np.float32)
        self.tpos_w, self.tpos_b = tp[:64 * 256].reshape(64, 256), tp[64 * 256:]

    def click_sparse(self, x, y, label=1):
        """2-token sparse prompt: one click + the not-a-point pad (matches HF _embed_points)."""
        xn, yn = 2 * ((x + 0.5) / 1024) - 1, 2 * ((y + 0.5) / 1024) - 1
        proj = 2 * math.pi * (xn * self.gauss[:128] + yn * self.gauss[128:])
        pe = self.pe1 if label == 1 else self.pe0
        tok = np.concatenate([np.sin(proj), np.cos(proj)]) + pe
        return np.concatenate([tok, self.nap]).astype(np.float32)

    def ptr_pos(self, t_diff):
        # float64 matvec: Accelerate's sgemv raises spurious FP-flag warnings on macOS
        pe = sine_pe_1d(t_diff / (NPTR_FRAMES - 1.0)).astype(np.float64)
        return (self.tpos_w.astype(np.float64) @ pe + self.tpos_b).astype(np.float32)


def upsample_1024(low):
    """Bilinear 256->1024, align_corners=False (torch semantics), low (256,256)."""
    return F.interpolate(torch.from_numpy(low)[None, None], size=(1024, 1024),
                         mode="bilinear", align_corners=False)[0, 0].numpy()


# ------------------------------------------------------------------ THE HOST LOOP
def track(run, nmm, consts, T=T, on_frame=None):
    """Per-frame orchestration over the 4 graphs. Returns per-frame dict list.

    run: object with encode(chw)->(pix_raw, hi0, hi1) flat float32;
         memcond(nmm, pix_raw, mem_spatial(N,4096,64), slot_tpe(N,64), ptr_tok(64,64),
                 ptr_pos(64,64), key_mask(N*4096+64)) -> pix_feat flat(IE);
         decode(pix_feat, hi0, hi1, sparse(512), nomem) -> (masks(4,256,256), iou(4), ptr(4,256), obj);
         memorize(pix_raw, mask_for_mem(1024,1024), occ) -> mem(4096,64)
    """
    spatial_bank, ptr_bank, outs = {}, {}, []
    cond_frame = 0
    for t in range(T):
        pix_raw, hi0, hi1 = run.encode(frame(t))
        prompted = t == cond_frame
        if prompted:
            sparse, nomem, pix_feat = consts.click_sparse(*CLICK), 1.0, pix_raw
        else:
            # ---- assemble the fixed bank ----
            mem = np.zeros((nmm, HW, MEMCH), np.float32)
            tpe = np.zeros((nmm, MEMCH), np.float32)
            km = np.full(nmm * HW + NPTR, MASK_NEG, np.float32)
            slot = 0
            mem[slot], tpe[slot] = spatial_bank[cond_frame], consts.mtpe[6]     # cond frame
            km[slot * HW:(slot + 1) * HW] = 0
            slot += 1
            for off in range(nmm - 1, 0, -1):                                    # most distant first
                pf = t - off
                if pf in spatial_bank and pf != cond_frame:
                    mem[slot], tpe[slot] = spatial_bank[pf], consts.mtpe[off - 1]
                    km[slot * HW:(slot + 1) * HW] = 0
                    slot += 1
            ptr_tok = np.zeros((NPTR, MEMCH), np.float32)
            ptr_pos = np.zeros((NPTR, MEMCH), np.float32)
            ptrs = [(t - cond_frame, ptr_bank[cond_frame])]                      # cond pointer (past only)
            for td in range(1, NPTR_FRAMES):
                pf = t - td
                if pf < 0:
                    break
                if pf in ptr_bank and pf != cond_frame:
                    ptrs.append((td, ptr_bank[pf]))
            for i, (td, p) in enumerate(ptrs):
                pos = consts.ptr_pos(td)
                for j in range(PTR_SPLIT):
                    ptr_tok[i * PTR_SPLIT + j] = p[j * MEMCH:(j + 1) * MEMCH]
                    ptr_pos[i * PTR_SPLIT + j] = pos
                    km[nmm * HW + i * PTR_SPLIT + j] = 0
            pix_feat = run.memcond(nmm, pix_raw, mem, tpe, ptr_tok, ptr_pos, km)
            sparse, nomem = consts.track_sparse, 0.0
        masks, iou, ptr, obj = run.decode(pix_feat, hi0, hi1, sparse, nomem)
        best = 1 + int(np.argmax(iou[1:]))                                       # multimask: tokens 1..3
        appearing = obj > 0
        low = masks[best] if appearing else np.full((256, 256), NO_OBJ_SCORE, np.float32)
        obj_ptr = ptr[best] if appearing else consts.no_obj_ptr
        high = upsample_1024(low)
        if prompted:
            mfm = (high > 0).astype(np.float32) * MEM_SCALE + MEM_BIAS
        else:
            mfm = 1.0 / (1.0 + np.exp(-high)) * MEM_SCALE + MEM_BIAS
        mem_t = run.memorize(pix_raw, mfm.astype(np.float32), 0.0 if appearing else 1.0)
        spatial_bank[t], ptr_bank[t] = mem_t, obj_ptr
        rec = dict(mask=low, obj=float(obj), ptr=obj_ptr, mem=mem_t,
                   pix_feat=None if prompted else pix_feat)
        outs.append(rec)
        if on_frame:
            on_frame(t, rec)
    return outs


# ------------------------------------------------------------------------- runners
class TorchRunner:
    """The patched fp32 torch wrappers (same modules that get exported)."""

    def __init__(self, nmm_list, bf16mem=False):
        import convert_sam2_video as cv
        from transformers import Sam2VideoModel
        self.cv = cv
        model = Sam2VideoModel.from_pretrained(CKPT).eval()
        cv.base.bake_pos_embed(model)
        Encode, MemCond, Decode, Memorize = cv.build(model)
        self.enc, self.dec, self.memz = Encode().eval(), Decode().eval(), Memorize().eval()
        self.mc = {n: MemCond(n).eval() for n in nmm_list}
        self.bf16mem = bf16mem

    @torch.no_grad()
    def encode(self, chw):
        f = self.enc(torch.from_numpy(chw))[0].numpy()
        return f[:IE], f[IE:IE + F0], f[IE + F0:]

    @torch.no_grad()
    def memcond(self, n, pix_raw, mem, tpe, ptr_tok, ptr_pos, km):
        if self.bf16mem:  # HF stores maskmem_features in bfloat16
            mem = torch.from_numpy(mem).to(torch.bfloat16).float().numpy()
        flat = np.concatenate([pix_raw, mem.ravel(), tpe.ravel(), ptr_tok.ravel(),
                               ptr_pos.ravel(), km]).astype(np.float32)[None]
        return self.mc[n](torch.from_numpy(flat))[0].numpy()

    @torch.no_grad()
    def decode(self, pix_feat, hi0, hi1, sparse, nomem):
        flat = np.concatenate([pix_feat, hi0, hi1, sparse, [nomem]]).astype(np.float32)[None]
        o = self.dec(torch.from_numpy(flat))[0].numpy()
        return split_dec(o)

    @torch.no_grad()
    def memorize(self, pix_raw, mfm, occ):
        flat = np.concatenate([pix_raw, mfm.ravel(), [occ]]).astype(np.float32)[None]
        return self.memz(torch.from_numpy(flat))[0].numpy().reshape(HW, MEMCH)


def split_dec(o):
    masks = o[:4 * 65536].reshape(4, 256, 256)
    iou = o[4 * 65536:4 * 65536 + 4]
    ptr = o[4 * 65536 + 4:4 * 65536 + 4 + 1024].reshape(4, 256)
    return masks, iou, ptr, float(o[-1])


class TfliteRunner:
    """fp16 .tflite graphs through the ai_edge_litert Interpreter (Mac, CPU)."""

    def __init__(self, nmm_list, record=None):
        from ai_edge_litert.interpreter import Interpreter
        self.record = record  # optional dict to capture (name -> (in, out)) for device vectors

        def load(name):
            it = Interpreter(model_path=f"{OUT}/{name}.tflite", num_threads=8)
            it.allocate_tensors()
            return it
        self.enc, self.dec, self.memz = load("sam2v_encode"), load("sam2v_decode"), load("sam2v_memorize")
        self.mc = {n: load(f"sam2v_memcond{n}") for n in nmm_list}

    def run1(self, it, name, x):
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(i["index"], x.reshape(i["shape"]).astype(np.float32))
        it.invoke()
        y = it.get_tensor(o["index"]).ravel().copy()
        if self.record is not None:
            self.record[name] = (x.ravel().copy(), y)
        return y

    def encode(self, chw):
        f = self.run1(self.enc, "sam2v_encode", chw)
        return f[:IE], f[IE:IE + F0], f[IE + F0:]

    def memcond(self, n, pix_raw, mem, tpe, ptr_tok, ptr_pos, km):
        flat = np.concatenate([pix_raw, mem.ravel(), tpe.ravel(), ptr_tok.ravel(), ptr_pos.ravel(), km])
        return self.run1(self.mc[n], f"sam2v_memcond{n}", flat)

    def decode(self, pix_feat, hi0, hi1, sparse, nomem):
        flat = np.concatenate([pix_feat, hi0, hi1, sparse, [nomem]])
        return split_dec(self.run1(self.dec, "sam2v_decode", flat))

    def memorize(self, pix_raw, mfm, occ):
        flat = np.concatenate([pix_raw, mfm.ravel(), [occ]])
        return self.run1(self.memz, "sam2v_memorize", flat).reshape(HW, MEMCH)


# ------------------------------------------------------------------ HF reference
def reference(nmm):
    """Clean HF streaming run (num_maskmem = nmm). Never import convert_sam2_video here."""
    assert "convert_sam2_video" not in sys.modules
    from transformers import Sam2VideoModel, Sam2VideoInferenceSession
    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    model.num_maskmem = nmm
    cap = {}
    model.memory_attention.register_forward_hook(
        lambda m, a, o: cap.__setitem__("pix_feat", o.detach().clone()))
    model.memory_encoder.register_forward_hook(
        lambda m, a, o: cap.__setitem__("mem", o[0].detach().clone()))
    sess = Sam2VideoInferenceSession(video_height=1024, video_width=1024, dtype=torch.float32)
    oi = sess.obj_id_to_idx(1)
    sess.add_point_inputs(oi, 0, {"point_coords": torch.tensor([[[[CLICK[0], CLICK[1]]]]]),
                                  "point_labels": torch.tensor([[[1]]])})
    sess.obj_with_new_inputs = [1]
    masks, objs, ptrs, mems, pfs = [], [], [], [], []
    with torch.no_grad():
        for t in range(T):
            cap.clear()
            out = model(inference_session=sess, frame=torch.from_numpy(frame(t)))
            masks.append(out.pred_masks.numpy().reshape(256, 256))
            objs.append(float(out.object_score_logits.reshape(-1)[0]))
            store = "cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"
            ptrs.append(sess.output_dict_per_obj[oi][store][t]["object_pointer"].numpy().reshape(256))
            mem_t = cap["mem"].numpy().reshape(MEMCH, HW).T.copy()               # (4096,64) fp32 pre-bf16
            if objs[-1] <= 0:  # HF adds the occlusion embedding after the encoder
                mem_t = mem_t + model.occlusion_spatial_embedding_parameter.detach().numpy().reshape(1, 64)
            mems.append(mem_t)
            pfs.append(np.zeros((HW * HD,), np.float32) if t == 0 else
                       cap["pix_feat"].numpy().reshape(HW, HD).T.reshape(-1).copy())  # (256,64,64) flat
            print(f"ref nmm={nmm} frame {t}: fg={(masks[-1] > 0).sum()} obj={objs[-1]:.3f}", flush=True)
    np.savez_compressed(f"{OUT}/ref_nmm{nmm}.npz", mask=np.stack(masks), obj=np.array(objs),
                        ptr=np.stack(ptrs), mem=np.stack(mems), pix_feat=np.stack(pfs))


# ------------------------------------------------------------------------ compare
def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(tag, nmm, outs):
    ref = np.load(f"{OUT}/ref_nmm{nmm}.npz")
    worst = dict(iou=1.0, corr=1.0, dmask=0.0, dpf=0.0, dmem=0.0, dptr=0.0)
    lines = []
    for t, o in enumerate(outs):
        rm, m = ref["mask"][t], o["mask"]
        r = dict(iou=iou(rm > 0, m > 0), corr=corr(rm, m), dmask=float(np.abs(rm - m).max()),
                 dobj=abs(float(ref["obj"][t]) - o["obj"]),
                 dptr=float(np.abs(ref["ptr"][t] - o["ptr"]).max()),
                 dmem=float(np.abs(ref["mem"][t] - o["mem"]).max()),
                 cmem=corr(ref["mem"][t], o["mem"]),
                 dpf=0.0 if t == 0 else float(np.abs(ref["pix_feat"][t] - o["pix_feat"]).max()),
                 cpf=1.0 if t == 0 else corr(ref["pix_feat"][t], o["pix_feat"]))
        lines.append(f"  f{t:02d} fg={int((m > 0).sum()):6d}/{int((rm > 0).sum()):6d} IoU={r['iou']:.4f} "
                     f"corr={r['corr']:.5f} max|dmask|={r['dmask']:.3f} |dobj|={r['dobj']:.3f} "
                     f"pix_feat corr={r['cpf']:.5f} max|d|={r['dpf']:.4f} mem corr={r['cmem']:.5f} "
                     f"max|d|={r['dmem']:.4f} |dptr|={r['dptr']:.4f}")
        worst["iou"] = min(worst["iou"], r["iou"]); worst["corr"] = min(worst["corr"], r["corr"])
        for k in ("dmask", "dpf", "dmem", "dptr"):
            worst[k] = max(worst[k], r[k])
    head = (f"[{tag}] nmm={nmm} T={len(outs)}: min IoU={worst['iou']:.4f} min corr={worst['corr']:.5f} "
            f"max|dmask|={worst['dmask']:.3f} max|dpix_feat|={worst['dpf']:.4f} "
            f"max|dmem|={worst['dmem']:.4f} max|dptr|={worst['dptr']:.4f}")
    print(head)
    print("\n".join(lines))
    with open(f"{OUT}/parity_{tag}_nmm{nmm}.log", "w") as f:
        f.write(head + "\n" + "\n".join(lines) + "\n")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "torch", "tflite", "vectors"])
    ap.add_argument("--nmm", default="7,2")
    ap.add_argument("--bf16mem", action="store_true")
    a = ap.parse_args()
    nmms = [int(x) for x in a.nmm.split(",")]
    if a.mode == "ref":
        for n in nmms:
            reference(n)
        return
    consts = Consts()
    if a.mode == "torch":
        run = TorchRunner(nmms, bf16mem=a.bf16mem)
        for n in nmms:
            compare("torch" + ("_bf16mem" if a.bf16mem else ""), n, track(run, n, consts))
    elif a.mode == "tflite":
        run = TfliteRunner(nmms)
        for n in nmms:
            compare("tflite", n, track(run, n, consts))
    elif a.mode == "vectors":
        vd = f"{OUT}/vectors"
        os.makedirs(vd, exist_ok=True)
        for n in nmms:
            rec = {}
            run = TfliteRunner([n], record=rec)
            track(run, n, consts)             # last frame's I/O of every graph stays in rec
            for name, (x, y) in rec.items():
                x.astype(np.float32).tofile(f"{vd}/{name}_in.bin")
                y.astype(np.float32).tofile(f"{vd}/{name}_out.bin")
                print(f"vectors: {name} in={x.size} out={y.size}")


if __name__ == "__main__":
    main()

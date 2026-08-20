#!/usr/bin/env python3
"""SAM 3.1 Object-Multiplex tracking with the OFFICIAL orchestration (CPU) and the heavy
compute swapped, one module at a time, for the LiteRT CompiledModel graphs running on the
host-Mac GPU. The reference is tracker_reference_cpu.py's dump (all-PyTorch).

Swap points (`--swap a,b,...`; 'all' = everything):
  vision   detector.backbone.forward_image  -> sam3_vision_tri.tflite (trunk + 3 necks +
           conv_s0/s1 folded; the official conv_s0/s1 modules become Identity)
  text     detector.backbone.forward_text   -> host tokenizer + fp16 table + sam3_text.tflite
  head     detector.forward_grounding       -> sam3_head.tflite
  memattn  tracker memory attention          -> trk_memattn_n{N}.tflite (fixed slots + mask)
  maskdec  tracker multiplex mask decoder    -> trk_maskdec.tflite
  memenc   tracker memory encoder            -> trk_memenc.tflite
Everything not swapped stays stock PyTorch, so each step isolates one graph's numerics
inside the real state machine. Per-frame agreement vs the reference is printed (ids,
probs, mask IoU).

Usage: tracker_litert_loop.py --swap vision,text [--clip models/clip8] [--prompt person]
       [--vision-f32] [--out models/tracker_ref]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402,F401
from tracker_reference_cpu import cpu_stubs  # noqa: E402
from run_pipeline_mac import GpuGraph  # noqa: E402
from build_sam3_tri import layout_offsets  # noqa: E402

OUT = os.path.join(ROOT, "models", "out")
CTX = 32


# ----------------------------------------------------------------------------- vision
class LiteRTVision:
    """Replacement for SAM3VLBackboneTri.forward_image using sam3_vision_tri.tflite."""

    def __init__(self, det, f32):
        self.offs, self.total = layout_offsets()
        self.g = GpuGraph(os.path.join(OUT, "sam3_vision_tri.tflite"), f32=f32, n_out=self.total)
        pe = det.backbone.vision_backbone.position_encoding
        with torch.no_grad():
            self.pos = [pe(torch.zeros(1, 256, hw, hw)).detach().clone() for hw in (288, 144, 72)]
        self.calls = 0
        self.ms = 0.0

    def _one(self, img):
        y = self.g(img.numpy())
        self.ms += self.g.last_ms
        self.calls += 1
        t = torch.from_numpy(y)

        def take(name):
            o0, o1, c, hw = self.offs[name]
            return t[o0:o1].reshape(1, c, hw, hw)
        return take

    def forward_image(self, samples, *, need_sam3_out=True, need_interactive_out=True,
                      need_propagation_out=True):
        from sam3.model.data_misc import NestedTensor
        outs = []
        for i in range(samples.shape[0]):
            outs.append(self._one(samples[i:i + 1].float().cpu()))
        cat = lambda name: torch.cat([tk(name) for tk in outs], 0)  # noqa: E731
        output = {}
        if need_sam3_out:
            fpn = [NestedTensor(cat(n), None) for n in ("sam3_fpn288", "sam3_fpn144", "sam3_fpn72")]
            output.update({"vision_features": fpn[-1].tensors, "vision_mask": None,
                           "vision_pos_enc": [p.expand(samples.shape[0], -1, -1, -1) for p in self.pos],
                           "backbone_fpn": fpn})
        if need_interactive_out:
            fpn = [NestedTensor(cat(n), None) for n in ("inter_h0", "inter_h1", "inter_f2")]
            output["interactive"] = {"vision_features": fpn[-1].tensors, "vision_mask": None,
                                     "vision_pos_enc": [p.expand(samples.shape[0], -1, -1, -1) for p in self.pos],
                                     "backbone_fpn": fpn}
        if need_propagation_out:
            fpn = [NestedTensor(cat(n), None) for n in ("prop_h0", "prop_h1", "prop_f2")]
            output["sam2_backbone_out"] = {"vision_features": fpn[-1].tensors, "vision_mask": None,
                                           "vision_pos_enc": [p.expand(samples.shape[0], -1, -1, -1) for p in self.pos],
                                           "backbone_fpn": fpn}
        return output


# ----------------------------------------------------------------------------- text
class LiteRTText:
    def __init__(self, det, f32=True):
        self.tok = det.backbone.language_backbone.tokenizer
        self.table = np.fromfile(os.path.join(OUT, "sam3_token_embed.bin"), dtype=np.float16).reshape(-1, 1024)
        self.g = GpuGraph(os.path.join(OUT, "sam3_text.tflite"), f32=f32, n_out=CTX * 256)
        self.ms = 0.0

    def forward_text(self, captions, input_boxes=None, additional_text=None, device="cpu"):
        assert additional_text is None
        ids = self.tok(list(captions), context_length=CTX).numpy()          # (B,32)
        mems, pads, embs = [], [], []
        for b in range(ids.shape[0]):
            emb = self.table[ids[b]].astype(np.float32)[None]                 # (1,32,1024)
            y = self.g(emb)
            self.ms += self.g.last_ms
            mems.append(torch.from_numpy(y.reshape(CTX, 1, 256)))
            pads.append(torch.from_numpy(ids[b] == 0)[None])
            embs.append(torch.from_numpy(emb[0])[:, None])                    # (32,1,1024)
        return {"language_features": torch.cat(mems, 1), "language_mask": torch.cat(pads, 0),
                "language_embeds": torch.cat(embs, 1)}


# ----------------------------------------------------------------------------- head
class LiteRTHead:
    """Replacement for Sam3MultiplexDetector.forward_grounding (text prompt only)."""

    def __init__(self, det, vision, f32=True):
        self.det = det
        self.vision = vision                                   # LiteRTVision (features)
        self.g = GpuGraph(os.path.join(OUT, "sam3_head.tflite"), f32=f32, n_out=1001 + 200 * 288 * 288)
        self.ms = 0.0

    def forward_grounding(self, backbone_out, find_input, find_target, geometric_prompt, **kw):
        from sam3.model.box_ops import box_cxcywh_to_xyxy
        assert geometric_prompt.box_embeddings is None or geometric_prompt.box_embeddings.shape[0] == 0
        img_batch = backbone_out["img_batch_all_stages"]
        img_id = int(find_input.img_ids.reshape(-1)[0])
        image = img_batch[img_id] if isinstance(img_batch, torch.Tensor) else img_batch[img_id]
        image = image.unsqueeze(0).float()
        vis = self.vision.forward_image(image)                  # dict (LiteRT vision tri)
        fpn = [x.tensors for x in vis["backbone_fpn"]]
        txt_id = int(find_input.text_ids.reshape(-1)[0])
        text_mem = backbone_out["language_features"][:, txt_id].reshape(1, -1)    # (1, 32*256)
        pad = backbone_out["language_mask"][txt_id].float().reshape(1, -1)         # (1, 32)
        head_in = torch.cat([f.flatten(1) for f in fpn] + [text_mem, pad], 1).numpy()
        y = torch.from_numpy(self.g(head_in))
        self.ms += self.g.last_ms
        logits = y[:200].reshape(1, 200, 1)
        boxes = y[200:1000].reshape(1, 200, 4)
        masks = y[1001:].reshape(1, 200, 288, 288)
        bo = dict(backbone_out)
        bo.update(vis)
        return {"pred_logits": logits, "pred_boxes": boxes, "pred_boxes_xyxy": box_cxcywh_to_xyxy(boxes),
                "pred_masks": masks, "presence_logit_dec": y[1000].reshape(1, 1),
                "prev_encoder_out": {"backbone_out": bo, "encoder_out": None}}


# ----------------------------------------------------------------------------- tracker graphs
class LiteRTMemEnc:
    """Replacement for SimpleMaskEncoder.forward(pix_feat (B,256,72,72), masks (B,32,1008,1008),
    skip_mask_sigmoid=True) -> {"vision_features": (B,256,72,72), "vision_pos_enc": [pos]}"""

    def __init__(self, enc, f32=True):
        self.enc = enc
        self.g = GpuGraph(os.path.join(ROOT, "models", "tracker_precheck", "trk_memenc.tflite"),
                          f32=f32, n_out=256 * 72 * 72)
        with torch.no_grad():
            self.pos = enc.position_encoding(torch.zeros(1, 256, 72, 72)).detach().clone()
        self.ms = 0.0

    def forward(self, pix_feat, masks, skip_mask_sigmoid=False):
        assert skip_mask_sigmoid, "the graph expects the already-transformed mask"
        B = pix_feat.shape[0]
        if not hasattr(self, "_shown"):
            self._shown = True
            print(f"[memenc] call shapes pix={tuple(pix_feat.shape)} masks={tuple(masks.shape)} dtype={masks.dtype}")
        if masks.shape[1] == 16:          # no condition channels configured
            masks = torch.cat([masks, torch.zeros_like(masks)], 1)
        assert masks.shape[1] == 32, masks.shape
        if masks.shape[-1] != 1008:
            print(f"[memenc] NOTE mask input {tuple(masks.shape)} -> resized to 1008 (bilinear)")
            masks = torch.nn.functional.interpolate(masks.float(), size=(1008, 1008), mode="bilinear",
                                                    align_corners=False)
        outs = []
        for b in range(B):
            x = torch.cat([pix_feat[b:b + 1].reshape(1, -1), masks[b:b + 1].float().reshape(1, -1)], 1)
            y = self.g(x.numpy())
            self.ms += self.g.last_ms
            outs.append(torch.from_numpy(y).reshape(1, 256, 72, 72))
        feats = torch.cat(outs, 0)
        return {"vision_features": feats, "vision_pos_enc": [self.pos.expand(B, -1, -1, -1)]}


class LiteRTMaskDec:
    """Replacement for MultiplexMaskDecoder.forward (propagation path, multimask_output=True)."""

    def __init__(self, dec, f32=True):
        self.dec = dec
        self.g = GpuGraph(os.path.join(ROOT, "models", "tracker_precheck", "trk_maskdec.tflite"),
                          f32=f32, n_out=16 * 3 * 288 * 288 + 16 * 3 + 16 + 16 * 3 * 256)
        self.ms = 0.0

    def forward(self, image_embeddings, image_pe, multimask_output, high_res_features=None,
                extra_per_object_embeddings=None):
        assert multimask_output and high_res_features is not None
        B = image_embeddings.shape[0]
        if extra_per_object_embeddings is None:
            extra_per_object_embeddings = torch.zeros(B, 16, 256)
        masks, ious, scores, toks = [], [], [], []
        for b in range(B):
            x = torch.cat([image_embeddings[b:b + 1].reshape(1, -1), high_res_features[0][b:b + 1].reshape(1, -1),
                           high_res_features[1][b:b + 1].reshape(1, -1),
                           extra_per_object_embeddings[b:b + 1].reshape(1, -1)], 1)
            y = torch.from_numpy(self.g(x.numpy()))
            self.ms += self.g.last_ms
            o = 0
            n = 16 * 3 * 288 * 288; masks.append(y[o:o + n].reshape(1, 16, 3, 288, 288)); o += n
            n = 16 * 3; ious.append(y[o:o + n].reshape(1, 16, 3)); o += n
            n = 16; scores.append(y[o:o + n].reshape(1, 16, 1)); o += n
            n = 16 * 3 * 256; toks.append(y[o:o + n].reshape(1, 16, 3, 256)); o += n
        return {"masks": torch.cat(masks, 0), "iou_pred": torch.cat(ious, 0),
                "object_score_logits": torch.cat(scores, 0), "sam_tokens_out": torch.cat(toks, 0)}


class LiteRTMemAttn:
    """Replacement for TransformerEncoderDecoupledCrossAttention.forward (memory attention).
    The official call has a variable bank: memory_image (n*L,1,C), memory = [maskmem (n*L,B,C)
    | obj ptrs (p*16,B,C)], matching pos tensors. The graph is fixed at N slots / P pointer
    frames for ONE bucket: pad with zeros + key mask, loop over buckets."""

    def __init__(self, enc, N=7, Pf=16, f32=True):
        self.enc, self.N, self.Pf = enc, N, Pf
        self.L, self.C = 5184, 256
        self.g = GpuGraph(os.path.join(ROOT, "models", "tracker_precheck", f"trk_memattn_n{N}.tflite"),
                          f32=f32, n_out=self.L * self.C)
        self.ms = 0.0
        self.max_n = 0
        self.max_p = 0

    def forward(self, image, src, memory_image, memory, image_pos=None, src_pos=None,
                memory_image_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        L, C, N, Pf = self.L, self.C, self.N, self.Pf
        B = src.shape[1]
        n_mem_tok = memory_image.shape[0]
        assert n_mem_tok % L == 0
        n = n_mem_tok // L
        p16 = num_obj_ptr_tokens
        assert p16 % 16 == 0 and memory.shape[0] == n_mem_tok + p16, (memory.shape, n_mem_tok, p16)
        p = p16 // 16
        assert n <= N and p <= Pf, (n, p)
        self.max_n, self.max_p = max(self.max_n, n), max(self.max_p, p)
        assert image.shape[1] == 1 and memory_image.shape[1] == 1
        outs = []
        for b in range(B):
            pix = image[:, 0].reshape(1, -1)
            mi = torch.zeros(N * L, C); mi[:n_mem_tok] = memory_image[:, 0]
            mip = torch.zeros(N * L, C); mip[:n_mem_tok] = memory_image_pos[:, 0] if memory_image_pos.shape[1] == 1 else memory_image_pos[:, b]
            mm = torch.zeros(N * L, C); mm[:n_mem_tok] = memory[:n_mem_tok, b]
            ptr = torch.zeros(Pf * 16, C); ptr[:p16] = memory[n_mem_tok:, b]
            ptrp = torch.zeros(Pf * 16, C); ptrp[:p16] = memory_pos[n_mem_tok:, b]
            keep = torch.zeros(N * L + Pf * 16); keep[:n_mem_tok] = 1.0; keep[N * L:N * L + p16] = 1.0
            x = torch.cat([pix, mi.reshape(1, -1), mip.reshape(1, -1), mm.reshape(1, -1),
                           ptr.reshape(1, -1), ptrp.reshape(1, -1), keep.reshape(1, -1)], 1)
            y = torch.from_numpy(self.g(x.numpy()))
            self.ms += self.g.last_ms
            outs.append(y.reshape(L, 1, C))
        return {"memory": torch.cat(outs, 1)}


class LiteRTInitDec:
    """Replacement for the interactive MaskDecoder.forward (mask-as-output init of new objects,
    multimask_output=False, repeat_image=True); one graph call per object."""

    def __init__(self, dec, f32=True):
        self.dec = dec
        self.g = GpuGraph(os.path.join(ROOT, "models", "tracker_precheck", "trk_initdec.tflite"),
                          f32=f32, n_out=288 * 288 + 1 + 256 + 1)
        self.ms = 0.0

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings,
                multimask_output, repeat_image, high_res_features=None):
        assert not multimask_output and repeat_image and high_res_features is not None
        n = sparse_prompt_embeddings.shape[0]
        assert image_embeddings.shape[0] == 1 and sparse_prompt_embeddings.shape[1] == 2
        masks, ious, toks, scores = [], [], [], []
        for i in range(n):
            x = torch.cat([image_embeddings.reshape(1, -1), high_res_features[0].reshape(1, -1),
                           high_res_features[1].reshape(1, -1), sparse_prompt_embeddings[i:i + 1].reshape(1, -1),
                           dense_prompt_embeddings[i:i + 1].reshape(1, -1)], 1)
            y = torch.from_numpy(self.g(x.numpy()))
            self.ms += self.g.last_ms
            o = 288 * 288
            masks.append(y[:o].reshape(1, 1, 288, 288)); ious.append(y[o:o + 1].reshape(1, 1))
            toks.append(y[o + 1:o + 257].reshape(1, 1, 256)); scores.append(y[o + 257:o + 258].reshape(1, 1))
        return torch.cat(masks, 0), torch.cat(ious, 0), torch.cat(toks, 0), torch.cat(scores, 0)


# ----------------------------------------------------------------------------- main
def compare(results, ref, tag):
    frames = sorted(ref)
    lines = []
    agree = True
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
                ious.append(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))
        dp = (np.abs(np.asarray(r["out_probs"]) - np.asarray(o["out_probs"])).max()
              if same and len(rid) else float("nan"))
        lines.append(f"  f{f}: ids ref={rid} got={oid} same={same} |dprob|={dp:.4f} "
                     f"maskIoU={'/'.join(f'{v:.3f}' for v in ious)}")
    print(f"[compare {tag}] all-ids-agree={agree}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap", default="vision,text")
    ap.add_argument("--clip", default=os.path.join(ROOT, "models", "clip8"))
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--vision-f32", action="store_true")
    ap.add_argument("--all-fp16", action="store_true", help="run every graph in the delegate's fp16 mode (Mali reality)")
    ap.add_argument("--ref", default=os.path.join(ROOT, "models", "tracker_ref", "ref_person.npy"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "sam3.1_multiplex.pt"))
    a = ap.parse_args()
    swaps = set(a.swap.split(",")) if a.swap != "all" else {"vision", "text", "head", "memattn", "maskdec", "memenc", "initdec"}
    cpu_stubs()
    from sam3 import model_builder as mb
    pred = mb.build_sam3_multiplex_video_predictor(checkpoint_path=a.ckpt, use_fa3=False,
                                                   use_rope_real=True, compile=False,
                                                   async_loading_frames=False)
    model = pred.model
    model.to("cpu")
    _init_state = model.init_state
    model.init_state = lambda **kw: _init_state(**{k: v for k, v in kw.items() if k != "offload_state_to_cpu"})
    # per-frame (non-batched) grounding so forward_image / forward_grounding see one frame
    model.use_batched_grounding = False
    det = model.detector

    timers = {}
    F32 = not a.all_fp16
    if "vision" in swaps:
        v = LiteRTVision(det, f32=a.vision_f32 and F32)
        det.backbone.forward_image = v.forward_image
        # conv_s0/s1 are folded into the graph -> identity in the official code paths
        for dec in (model.tracker.model.interactive_sam_mask_decoder, model.tracker.model.sam_mask_decoder):
            dec.conv_s0 = nn.Identity()
            dec.conv_s1 = nn.Identity()
        timers["vision"] = v
        print(f"[swap] vision -> sam3_vision_tri.tflite ({'f32' if a.vision_f32 else 'fp16'})")
    if "text" in swaps:
        t = LiteRTText(det, f32=F32)
        det.backbone.forward_text = t.forward_text
        timers["text"] = t
        print("[swap] text -> sam3_text.tflite (f32)")
    if "memattn" in swaps:
        ma = LiteRTMemAttn(model.tracker.model.transformer.encoder, N=7, Pf=16, f32=F32)
        model.tracker.model.transformer.encoder.forward = ma.forward
        timers["memattn"] = ma
        print("[swap] memattn -> trk_memattn_n7.tflite (f32)")
    if "initdec" in swaps:
        idc = LiteRTInitDec(model.tracker.model.interactive_sam_mask_decoder, f32=F32)
        model.tracker.model.interactive_sam_mask_decoder.forward = idc.forward
        timers["initdec"] = idc
        print("[swap] initdec -> trk_initdec.tflite")
    if "memenc" in swaps:
        me = LiteRTMemEnc(model.tracker.model.maskmem_backbone, f32=F32)
        model.tracker.model.maskmem_backbone.forward = me.forward
        timers["memenc"] = me
        print("[swap] memenc -> trk_memenc.tflite (f32)")
    if "maskdec" in swaps:
        md = LiteRTMaskDec(model.tracker.model.sam_mask_decoder, f32=F32)
        model.tracker.model.sam_mask_decoder.forward = md.forward
        timers["maskdec"] = md
        print("[swap] maskdec -> trk_maskdec.tflite (f32)")
    if "head" in swaps:
        assert "vision" in swaps, "head swap needs the LiteRT vision features"
        h = LiteRTHead(det, timers["vision"], f32=F32)
        det.forward_grounding = h.forward_grounding
        timers["head"] = h
        print("[swap] head -> sam3_head.tflite (f32)")

    r = pred.handle_request(dict(type="start_session", resource_path=a.clip))
    sid = r["session_id"]
    results = {}

    def record(frame_idx, out):
        results[frame_idx] = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                              for k, v in out.items()}
    t0 = time.time()
    r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=a.prompt))
    record(0, r["outputs"])
    for ev in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                              propagation_direction="forward", start_frame_index=0)):
        record(ev["frame_index"], ev["outputs"])
    total = time.time() - t0
    print(f"[run] {len(results)} frames in {total:.1f}s  " +
          "  ".join(f"{k}: {getattr(v, 'ms', 0):.0f} ms total" for k, v in timers.items()))
    ref = np.load(a.ref, allow_pickle=True).item()
    compare(results, ref, a.swap)
    try:
        pred.handle_request(dict(type="close_session", session_id=sid))
    except AssertionError:
        pass
    sys.stdout.flush()


if __name__ == "__main__":
    main()

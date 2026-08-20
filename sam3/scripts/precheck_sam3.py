#!/usr/bin/env python3
"""SAM 3.1 (facebook/sam3.1) image-side precheck for LiteRT CompiledModel GPU.

Precheck only (2026-08-19): can the SAM 3.1 *detector* (the image side of the
Object-Multiplex checkpoint) be torch.export-ed and lowered by litert-torch,
and what does the op histogram / fp16 size look like? No GPU patches are
applied unless a sub-graph fails to export at all (the "raw convert first"
rule of the convert loop). Three sub-graphs, each a single flat float I/O:

  vision  image (1,3,1008,1008) -> [fpn288 | fpn144 | fpn72]  (SAM3 tri-neck head)
          ViT-L/14 @1008 (32 blocks, window 24, global blocks 7/15/23/31,
          real-valued 2D RoPE) + Sam3TriViTDetNeck (sam3 head only).
  text    token ids (1,32) int64 -> [text_mem(32*256) | pad_mask(32)]
          24-layer CLIP-style text transformer (width 1024) + resizer.
  head    [fpn288 | fpn144 | fpn72 | text_mem | text_pad] -> [logits(200) |
          boxes(200*4) | presence(1) | masks(200*H*W)]
          6-layer fusion encoder + 6-layer DETR decoder (200 queries, box
          refine, log-RPB, presence token) + dot-product scoring +
          UniversalSegmentationHead (pixel decoder + mask embed).

Usage:  precheck_sam3.py {vision,text,head,all} [--ckpt models/sam3.1_multiplex.pt]
        [--no-convert] [--out models/precheck]

Env: sam3/.venv (torch 2.12 CPU/MPS, litert-torch 0.9.3, ai-edge-litert,
ai-edge-quantizer, facebookresearch/sam3 installed -e).
"""
import argparse
import collections
import os
import sys
import time

import types

import numpy as np

# --- sam3.model.edt imports triton unconditionally (CUDA-only EDT kernel); stub the
# module so the CPU/MPS precheck can import the tracker utils without triton.
try:
    import triton  # noqa: F401
except ImportError:
    _edt = types.ModuleType("sam3.model.edt")
    _edt.edt_triton = None
    sys.modules["sam3.model.edt"] = _edt

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPU_BAD = {"GATHER_ND", "GATHER", "SELECT", "SELECT_V2", "NOT_EQUAL", "EQUAL",
           "GREATER", "LESS", "TOPK_V2", "CAST", "PACK", "SPLIT", "TRANSPOSE_CONV",
           "ARG_MAX", "ARG_MIN", "WHERE", "CUMSUM", "SCATTER_ND", "UNIQUE",
           "NON_MAX_SUPPRESSION_V4", "NON_MAX_SUPPRESSION_V5"}
RECIPE = [{
    "regex": ".*", "operation": "*", "algorithm_key": "float_casting",
    "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}},
}]
# fp16 only on the matmul-class weights: keeps small buffers (pos-enc, query embeds, masks)
# fp32 so no stray DEQUANTIZE feeds an elementwise op (the Metal delegate refuses those).
RECIPE_MATMUL_ONLY = [{
    "regex": ".*", "operation": op, "algorithm_key": "float_casting",
    "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}},
} for op in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D")] + [{
    # tiny MLPs applied once per decoder layer (weights shared by 6 FCs): a shared fp16
    # weight = one DEQUANTIZE with 6 consumers, which the Metal delegate refuses -> keep fp32
    "regex": ".*boxRPB_embed.*", "operation": "FULLY_CONNECTED", "algorithm_key": "no_quantize",
    "op_config": {},
}]
CONTEXT = 32


# ----------------------------------------------------------------------------- build
def build_detector(ckpt_path=None):
    """Sam3MultiplexDetector exactly as build_sam3_multiplex_video_predictor builds it,
    weights from the `detector.` prefix of sam3.1_multiplex.pt (strict on that subset)."""
    import pkg_resources
    from sam3 import model_builder as mb
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    # PositionEmbeddingSine(precompute_resolution=...) allocates on "cuda" unconditionally;
    # the lazy per-size cache computes the same constants on CPU, so drop the precompute.
    from sam3.model.position_encoding import PositionEmbeddingSine
    mb._create_position_encoding = lambda precompute_resolution=None: PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000,
        precompute_resolution=None)
    # TransformerDecoder.__init__ precomputes the log-RPB coords on "cuda"; keep them CPU.
    from sam3.model.decoder import TransformerDecoder
    _orig_get_coords = TransformerDecoder._get_coords
    TransformerDecoder._get_coords = staticmethod(
        lambda H, W, device: _orig_get_coords(H, W, "cpu" if device == "cuda" else device))
    # ViT MLP uses a CUDA bf16 fused addmm (perflib.fused.addmm_act); use the plain fp32 form.
    import sam3.model.vitdet as vitdet
    vitdet.addmm_act = lambda act, linear, x: act()(linear(x))
    # geometry encoder calls Tensor.pin_memory() (CUDA-only staging); no-op on CPU/MPS.
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    tri_neck = mb._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False,
                                                 use_rope_real=True)
    text_encoder = mb._create_text_encoder(bpe_path)
    backbone = SAM3VLBackboneTri(scalp=0, visual=tri_neck, text=text_encoder)
    transformer = mb._create_sam3_transformer(use_fa3=False)
    segmentation_head = mb._create_segmentation_head(use_fa3=False)
    geometry_encoder = mb._create_geometry_encoder()
    dot_prod_scoring = mb._create_dot_product_scoring()
    detector = Sam3MultiplexDetector(
        num_feature_levels=1, backbone=backbone, transformer=transformer,
        segmentation_head=segmentation_head, semantic_segmentation_head=None,
        input_geometry_encoder=geometry_encoder, use_early_fusion=True,
        use_dot_prod_scoring=True, dot_prod_scoring=dot_prod_scoring,
        supervise_joint_box_scores=True, is_multiplex=True)
    if ckpt_path:
        t0 = time.time()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        det = {k[len("detector."):]: v for k, v in ckpt.items() if k.startswith("detector.")}
        missing, unexpected = detector.load_state_dict(det, strict=False)
        missing = [k for k in missing if "freqs_cis" not in k and "attn_mask" not in k]
        print(f"[ckpt] detector.* keys={len(det)} missing={len(missing)} "
              f"unexpected={len(unexpected)} ({time.time()-t0:.0f}s)")
        if missing:
            print("  missing:", missing[:20])
        if unexpected:
            print("  unexpected:", unexpected[:20])
        n_tracker = sum(1 for k in ckpt if k.startswith("tracker."))
        print(f"[ckpt] tracker.* keys={n_tracker}  other="
              f"{sum(1 for k in ckpt if not k.startswith(('detector.', 'tracker.')))}")
    # --- export-only patches (numerically identical at 1008x1008) ---
    # (a) ViT attention keeps the complex freqs_cis buffer next to the real/imag pair
    #     used by use_rope_real=True; litert-torch cannot serialize complex64 -> drop it.
    n_cplx = 0
    for m in detector.modules():
        fc = getattr(m, "freqs_cis", None)
        if isinstance(fc, torch.Tensor) and fc.is_complex():
            assert getattr(m, "use_rope_real", False)
            m._buffers["freqs_cis"] = torch.zeros(1)   # keep the not-None assert happy
            n_cplx += 1
    print(f"[patch] dropped complex freqs_cis buffers: {n_cplx}")
    # (b) log-RPB feat_size arrives as a 0-d tensor pair (spatial_shapes[0]) which turns
    #     the coordinate grid into unbacked symbols under torch.export; the graph is fixed
    #     at 1008 -> 72x72, so pin it.
    _orig_rpb = TransformerDecoder._get_rpb_matrix
    TransformerDecoder._get_rpb_matrix = lambda self, rb, fs: _orig_rpb(self, rb, (72, 72))
    # (c) text-only prompting: the geometry encoder still calls torchvision.roi_align on
    #     ZERO boxes (no lowering in litert-torch); the result is an empty tensor, so
    #     short-circuit it. Box/point prompts are out of scope for this precheck.
    import sam3.model.geometry_encoders as ge
    _orig_roi = ge.torchvision.ops.roi_align

    def _roi_align_empty_ok(inp, boxes, output_size, *a, **k):
        if isinstance(boxes, (list, tuple)) and all(b.shape[0] == 0 for b in boxes):
            return inp.new_zeros(0, inp.shape[1], output_size, output_size)
        return _orig_roi(inp, boxes, output_size, *a, **k)
    ge.torchvision.ops.roi_align = _roi_align_empty_ok
    _orig_gs = ge.torch.nn.functional.grid_sample

    def _grid_sample_empty_ok(inp, grid, *a, **k):
        if grid.shape[1] == 0:   # zero points -> [bs, C, 0, 1]
            return inp.new_zeros(inp.shape[0], inp.shape[1], 0, grid.shape[2])
        return _orig_gs(inp, grid, *a, **k)
    ge.torch.nn.functional.grid_sample = _grid_sample_empty_ok
    # concat_padded_sequences uses a data-dependent scatter; with an empty side it is
    # the identity on the other side (exact), which is all the text-only prompt needs
    # (0 points + 0 boxes + cls token).
    _orig_cps = ge.concat_padded_sequences

    def _cps_empty_ok(seq1, mask1, seq2, mask2, return_index=False):
        if seq1.shape[0] == 0 or seq2.shape[0] == 0:
            seq, mask = (seq2, mask2) if seq1.shape[0] == 0 else (seq1, mask1)
            if return_index:
                idx = torch.arange(seq2.shape[0], device=seq.device)[None].repeat(seq.shape[1], 1)
                return seq, mask, idx
            return seq, mask
        return _orig_cps(seq1, mask1, seq2, mask2, return_index)
    ge.concat_padded_sequences = _cps_empty_ok
    detector.eval()
    return detector


# ----------------------------------------------------------------------------- wrappers
class VisionFlat(nn.Module):
    """image -> flat [fpn288 | fpn144 | fpn72] of the SAM3 (detector) tri-neck head."""

    def __init__(self, det):
        super().__init__()
        self.neck = det.backbone.vision_backbone

    def forward(self, x):
        sam3_out, _, _, _, _, _ = self.neck(x, need_sam3_out=True,
                                            need_interactive_out=False,
                                            need_propagation_out=False)
        return torch.cat([getattr(f, "tensors", f).flatten(1) for f in sam3_out], 1)


class TextFlat(nn.Module):
    """token ids (1,32) -> flat [text_mem(32*256) | pad_mask(32)] (mask 1.0 = padding)."""

    def __init__(self, det):
        super().__init__()
        self.te = det.backbone.language_backbone

    def forward(self, tokens):
        enc = self.te.encoder
        x = enc.token_embedding(tokens) + enc.positional_embedding[:CONTEXT]
        x = enc.transformer(x, attn_mask=enc.attn_mask[:CONTEXT, :CONTEXT])
        x = enc.ln_final(x)                       # (1, 32, 1024) tokens
        mem = self.te.resizer(x)                  # (1, 32, 256)
        pad = (tokens == 0).to(mem.dtype)         # (1, 32)
        return torch.cat([mem.flatten(1), pad], 1)


class HeadFlat(nn.Module):
    """[fpn288 | fpn144 | fpn72 | text_mem | text_pad] -> [logits | boxes | presence | masks]."""

    def __init__(self, det, sizes):
        super().__init__()
        self.det = det
        self.sizes = sizes  # [(288,288),(144,144),(72,72)]
        self.n = [256 * h * w for h, w in sizes]
        # constant sine pos-enc per level (fixed size -> constant)
        pe = det.backbone.vision_backbone.position_encoding
        self.pos = nn.ParameterList([
            nn.Parameter(pe(torch.zeros(1, 256, h, w)).detach(), requires_grad=False)
            for h, w in sizes])

    def forward(self, flat):
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt
        det = self.det
        off = 0
        fpn = []
        for (h, w), n in zip(self.sizes, self.n):
            fpn.append(flat[:, off:off + n].reshape(1, 256, h, w))
            off += n
        text_mem = flat[:, off:off + CONTEXT * 256].reshape(1, CONTEXT, 256).transpose(0, 1)
        off += CONTEXT * 256
        text_pad = flat[:, off:off + CONTEXT] > 0.5   # (1,32) bool, True = padding
        backbone_out = {
            "backbone_fpn": fpn, "vision_pos_enc": list(self.pos),
            "language_features": text_mem, "language_mask": text_pad,
        }
        dev = flat.device
        find_input = FindStage(
            img_ids=torch.tensor([0], device=dev), text_ids=torch.tensor([0], device=dev),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)
        geo = Prompt(box_embeddings=torch.zeros(0, 1, 4, device=dev),
                     box_mask=torch.zeros(1, 0, device=dev, dtype=torch.bool))
        out = det.forward_grounding(backbone_out=backbone_out, find_input=find_input,
                                    find_target=None, geometric_prompt=geo)
        logits = out["pred_logits"].reshape(1, -1)          # (1,200)
        boxes = out["pred_boxes"].reshape(1, -1)            # (1,800) cxcywh norm
        presence = out["presence_logit_dec"].reshape(1, -1)  # (1,1)
        masks = out["pred_masks"].reshape(1, -1)            # (1,200*H*W)
        return torch.cat([logits, boxes, presence, masks], 1)


# ----------------------------------------------------------------------------- helpers
def back_to_cpu(det):
    """After an MPS timing pass: move the shared detector back and drop device caches."""
    det.to("cpu")
    for m in det.modules():
        c = getattr(m, "cache", None)
        if isinstance(c, dict):
            c.clear()
    if hasattr(det, "transformer") and hasattr(det.transformer, "decoder"):
        det.transformer.decoder.compilable_cord_cache = None


def timeit(fn, x, n=3, tag=""):
    with torch.inference_mode():
        fn(x)
        ts = []
        for _ in range(n):
            t0 = time.time()
            y = fn(x)
            if x.device.type == "mps":
                torch.mps.synchronize()
            ts.append(time.time() - t0)
    print(f"[torch {tag}] {x.device.type}: {min(ts)*1000:.0f} ms (best of {n})  out={tuple(y.shape)}")
    return y


def opcheck(path, tag):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k in GPU_BAD or k.startswith("Flex")}
    over = [d for d in it.get_tensor_details() if len(d.get("shape", [])) > 4]
    total = sum(ops.values())
    print(f"[ops {tag}] total={total} kinds={len(ops)}")
    print("   histogram:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"   GPU_BAD={bad or 'NONE'}  >4D tensors={len(over)}"
          + (f" e.g. {over[0]['name']} {list(over[0]['shape'])}" if over else ""))
    return ops, bad, over


def fp16(src, dst, matmul_only=True):
    from ai_edge_quantizer import quantizer
    if os.path.exists(dst):
        os.remove(dst)
    q = quantizer.Quantizer(src)
    q.load_quantization_recipe(RECIPE_MATMUL_ONLY if matmul_only else RECIPE)
    q.quantize().export_model(dst)
    return os.path.getsize(dst) / 1e6


def convert(mod, x, name, out_dir):
    import litert_torch
    fp32 = os.path.join(out_dir, f"{name}_fp32.tflite")
    t0 = time.time()
    try:
        ep = litert_torch.convert(mod.eval(), (x,))
        ep.export(fp32)
    except Exception as e:  # noqa: BLE001
        print(f"[convert {name}] FAILED after {time.time()-t0:.0f}s: {type(e).__name__}: "
              f"{str(e)[:1500]}")
        return None
    print(f"[convert {name}] ok {time.time()-t0:.0f}s  fp32={os.path.getsize(fp32)/1e6:.1f} MB")
    opcheck(fp32, name)
    try:
        dst = os.path.join(out_dir, f"{name}.tflite")
        print(f"[fp16 {name}] {fp16(fp32, dst):.1f} MB")
    except Exception as e:  # noqa: BLE001
        print(f"[fp16 {name}] FAILED: {type(e).__name__}: {str(e)[:500]}")
    return fp32


def tflite_parity(path, x, ref, tag):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=8)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], x.cpu().numpy().astype(inp["dtype"]))
    t0 = time.time()
    it.invoke()
    dt = time.time() - t0
    y = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1).astype(np.float32)
    r = ref.reshape(-1).cpu().numpy().astype(np.float32)
    corr = float(np.corrcoef(y, r)[0, 1]) if y.size > 1 else 0.0
    print(f"[tflite-cpu {tag}] {dt*1000:.0f} ms  corr={corr:.5f}  max|diff|={np.abs(y-r).max():.4g}")


def gpu_mac(path, x, ref, tag, n=5, f32=False):
    """Host-Mac GPU (Metal accelerator of ai-edge-litert) CompiledModel run: compile time,
    fully-accelerated flag, per-run wall time (run+read), parity vs torch."""
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    t0 = time.time()
    tag = tag + (" f32" if f32 else " fp16")
    try:
        if f32:
            from ai_edge_litert.options import Options
            o = Options.create()
            o.hardware_accelerators = HardwareAccelerator.GPU
            o.gpu_options.enforce_f32 = True
            model = CompiledModel.from_file(path, options=o)
        else:
            model = CompiledModel.from_file(path, HardwareAccelerator.GPU)
    except Exception as e:  # noqa: BLE001
        print(f"[gpu-mac {tag}] COMPILE FAILED: {type(e).__name__}: {str(e)[:600]}")
        return
    tc = time.time() - t0
    try:
        fully = model.is_fully_accelerated()
    except Exception as e:  # noqa: BLE001
        fully = f"? ({type(e).__name__})"
    r = ref.reshape(-1).cpu().numpy().astype(np.float32)
    try:
        ib = model.create_input_buffers(0)
        ob = model.create_output_buffers(0)
        xin = x.cpu().numpy()
        ib[0].write(xin.ravel())
        ts = []
        for _ in range(n):
            t1 = time.time()
            model.run_by_index(0, ib, ob)
            y = np.array(ob[0].read(int(r.size), np.float32))
            ts.append(time.time() - t1)
        corr = float(np.corrcoef(y, r)[0, 1])
        print(f"[gpu-mac {tag}] compile {tc:.1f}s fully_accelerated={fully} "
              f"run+read best {min(ts)*1000:.0f} ms (first {ts[0]*1000:.0f} ms, n={n})  "
              f"corr={corr:.5f} max|diff|={np.abs(y-r).max():.4g}")
    except Exception as e:  # noqa: BLE001
        print(f"[gpu-mac {tag}] RUN FAILED: {type(e).__name__}: {str(e)[:600]}")
    finally:
        try:
            model.close()
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["vision", "text", "head", "all"])
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "sam3.1_multiplex.pt"))
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--no-mps", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "precheck"))
    ap.add_argument("--vit4d", action="store_true",
                    help="apply scripts/vit4d.py (exact 4-D ViT re-authoring) before export")
    ap.add_argument("--gpu-mac", action="store_true",
                    help="also run the fp16 graph on the host Mac GPU via CompiledModel")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    det = build_detector(None if a.no_ckpt else a.ckpt)
    n_vit = sum(p.numel() for p in det.backbone.vision_backbone.trunk.parameters())
    n_neck = sum(p.numel() for p in det.backbone.vision_backbone.parameters()) - n_vit
    n_txt = sum(p.numel() for p in det.backbone.language_backbone.parameters())
    n_all = sum(p.numel() for p in det.parameters())
    print(f"[params] ViT={n_vit/1e6:.1f}M tri-neck={n_neck/1e6:.1f}M text={n_txt/1e6:.1f}M "
          f"head(enc/dec/seg/geo/score)={(n_all-n_vit-n_neck-n_txt)/1e6:.1f}M total={n_all/1e6:.1f}M")

    dev_mps = torch.device("mps") if (torch.backends.mps.is_available() and not a.no_mps) else None
    do = {a.what} if a.what != "all" else {"vision", "text", "head"}
    sizes = [(288, 288), (144, 144), (72, 72)]

    vision_ref = None
    if "vision" in do:
        m = VisionFlat(det)
        x = torch.randn(1, 3, 1008, 1008)
        vision_ref = timeit(m, x, n=2, tag="vision")
        if a.vit4d:
            sys.path.insert(0, HERE)
            from vit4d import patch_vit_4d
            patch_vit_4d(det.backbone.vision_backbone.trunk)
            with torch.inference_mode():
                y4 = m(x)
            d = (y4 - vision_ref).abs().max().item()
            corr = float(np.corrcoef(y4.reshape(-1).numpy(), vision_ref.reshape(-1).numpy())[0, 1])
            print(f"[vit4d] torch parity vs stock: corr={corr:.6f} max|diff|={d:.3g}")
            vision_ref = timeit(m, x, n=2, tag="vision-4d")
        if dev_mps is not None:
            mm = VisionFlat(det).to(dev_mps)
            timeit(mm, x.to(dev_mps), n=3, tag="vision")
            del mm
            back_to_cpu(det)
        if not a.no_convert:
            p = convert(m, x, "sam3_vision", a.out)
            if p:
                tflite_parity(p, x, vision_ref, "vision")
                if a.gpu_mac:
                    gpu_mac(os.path.join(a.out, "sam3_vision.tflite"), x, vision_ref, "vision")

    text_ref = None
    if "text" in do:
        m = TextFlat(det)
        tok = det.backbone.language_backbone.tokenizer(["a red car"], context_length=CONTEXT)
        print("[text] tokens:", tok[0].tolist())
        text_ref = timeit(m, tok, n=3, tag="text")
        if not a.no_convert:
            p = convert(m, tok, "sam3_text", a.out)
            if p:
                tflite_parity(p, tok, text_ref, "text")
                if a.gpu_mac:
                    gpu_mac(os.path.join(a.out, "sam3_text.tflite"), tok, text_ref, "text")

    if "head" in do:
        m = HeadFlat(det, sizes)
        n_img = sum(256 * h * w for h, w in sizes)
        if vision_ref is None or text_ref is None:
            flat = torch.randn(1, n_img + CONTEXT * 256 + CONTEXT) * 0.5
            flat[:, -CONTEXT:] = 0.0
            flat[:, -CONTEXT + 5:] = 1.0  # tokens 5.. are padding
        else:
            flat = torch.cat([vision_ref, text_ref], 1)
        y = timeit(m, flat, n=2, tag="head")
        n_masks = y.shape[1] - 200 - 800 - 1
        print(f"[head] mask elements per query = {n_masks // 200} "
              f"(= {int((n_masks // 200) ** 0.5)}^2)")
        if dev_mps is not None:
            mm = HeadFlat(det, sizes).to(dev_mps)
            det.transformer.decoder.compilable_cord_cache = None
            timeit(mm, flat.to(dev_mps), n=3, tag="head")
            del mm
            back_to_cpu(det)
        if not a.no_convert:
            p = convert(m, flat, "sam3_head", a.out)
            if p:
                tflite_parity(p, flat, y, "head")
                if a.gpu_mac:
                    gpu_mac(os.path.join(a.out, "sam3_head.tflite"), flat, y, "head")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

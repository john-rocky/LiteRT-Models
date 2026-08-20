#!/usr/bin/env python3
"""Run the OFFICIAL SAM 3.1 multiplex video predictor on CPU (cuda calls stubbed) on a short
JPEG clip with a text prompt, and dump per-frame outputs (object ids, scores, boxes, masks
@ 288x288) to models/tracker_ref/. This is the stage-2 numerical reference for the LiteRT
port (CompiledModel graphs + host loop must reproduce these).

Usage: tracker_reference_cpu.py [--clip models/clip8] [--prompt person]

Requires these one-time edits in vendor_sam3 (scratch clone, git-ignored), applied with sed:
  sam3/model/*.py : .cuda(non_blocking=True)/.cuda() -> .to("cpu"); torch.device("cuda") ->
                    torch.device("cpu"); device="cuda" -> device="cpu"; pin_memory=True -> False
  sam3/model/sam3_multiplex_detector.py : drop the two `x.to(torch.bfloat16)` backbone casts
  sam3/model/sam3_multiplex_tracking.py : source_state.get("device", "cuda") -> "cpu"
(pin_memory=True on a Mac silently yields MPS tensors; bf16 casts need CUDA AMP.)
Result on models/clip8 / "person": 4 objects tracked over 8 frames, ~3.7 s/frame CPU.
"""
import argparse
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402,F401  (triton stub etc.)


def cpu_stubs():
    """Every hard-coded CUDA call in the tracker path -> CPU (numerically irrelevant).
    Must run AFTER all sam3/torch imports (torch.device is replaced by a callable)."""
    from sam3 import model_builder  # noqa: F401  (import everything first)
    # sam3_multiplex_base queries torch.cuda.get_device_properties(0) at import time
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(major=0, minor=0, name="cpu")
    import sam3.model.sam3_multiplex_tracking, sam3.model.sam3_multiplex_base  # noqa: F401,E401
    import sam3.model.video_tracking_multiplex_demo, sam3.model.sam3_multiplex_video_predictor  # noqa: F401,E401
    import sam3.model.sam3_video_base, sam3.model.io_utils  # noqa: F401,E401
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    torch.cuda.is_available = lambda: False
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.current_device = lambda: 0
    from sam3.model.position_encoding import PositionEmbeddingSine
    _orig_init = PositionEmbeddingSine.__init__

    def _init_no_precompute(self, *args, **kw):
        kw["precompute_resolution"] = None
        if len(args) >= 5:
            args = args[:4]
        _orig_init(self, *args, **kw)
    PositionEmbeddingSine.__init__ = _init_no_precompute
    import sam3.model.vitdet as vitdet
    vitdet.addmm_act = lambda act, linear, x: act()(linear(x))
    from sam3.model.decoder import TransformerDecoder
    _orig = TransformerDecoder._get_coords
    TransformerDecoder._get_coords = staticmethod(
        lambda H, W, device: _orig(H, W, "cpu" if device == "cuda" else device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(ROOT, "models", "clip8"))
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "tracker_ref"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "sam3.1_multiplex.pt"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cpu_stubs()
    from sam3 import model_builder as mb
    t0 = time.time()
    pred = mb.build_sam3_multiplex_video_predictor(checkpoint_path=a.ckpt, use_fa3=False,
                                                   use_rope_real=True, compile=False,
                                                   async_loading_frames=False)
    # the wrapper opened a cuda autocast context (disabled on CPU with a warning) -> fine
    print(f"[build] predictor in {time.time()-t0:.0f}s")
    # move model to cpu explicitly
    pred.model.to("cpu")
    # upstream mismatch: base start_session passes offload_state_to_cpu which the multiplex
    # init_state does not accept
    _init_state = pred.model.init_state
    pred.model.init_state = lambda **kw: _init_state(**{k: v for k, v in kw.items() if k != "offload_state_to_cpu"})
    r = pred.handle_request(dict(type="start_session", resource_path=a.clip))
    sid = r["session_id"]
    print("[session]", sid)
    t0 = time.time()
    r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=a.prompt))
    print(f"[add_prompt] '{a.prompt}' frame0 in {time.time()-t0:.1f}s; outputs keys:",
          list(r["outputs"].keys()) if isinstance(r.get("outputs"), dict) else type(r.get("outputs")))
    results = {}

    def record(frame_idx, out):
        # out: dict with out_obj_ids, out_probs, out_boxes_xywh?, out_binary_masks ...
        rec = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                rec[k] = v.detach().cpu().numpy()
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                rec[k] = [t.detach().cpu().numpy() for t in v]
            else:
                rec[k] = v
        results[frame_idx] = rec
    record(0, r["outputs"])
    print("[frame 0] keys:", {k: (getattr(v, 'shape', None) or v) for k, v in results[0].items()
                              if not isinstance(v, list)})
    t0 = time.time()
    n = 0
    for ev in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                              propagation_direction="forward", start_frame_index=0)):
        record(ev["frame_index"], ev["outputs"])
        n += 1
        o = ev["outputs"]
        ids = o.get("out_obj_ids")
        probs = o.get("out_probs")
        print(f"[frame {ev['frame_index']}] ids={ids} probs={probs if probs is None else np.round(np.asarray(probs), 3).tolist()}",
              flush=True)
    print(f"[propagate] {n} frames in {time.time()-t0:.1f}s")
    np.save(os.path.join(a.out, f"ref_{a.prompt.replace(' ', '_')}.npy"), results, allow_pickle=True)
    try:
        pred.handle_request(dict(type="close_session", session_id=sid))
    except AssertionError:
        pass  # _gpu_mem_snapshot asserts CUDA
    sys.stdout.flush()


if __name__ == "__main__":
    main()

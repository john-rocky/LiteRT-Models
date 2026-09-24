"""Export graph A (n3d_frontend) and graph B (n3d_encoder_<mode>) with litert-torch, fp32 + fp16.

graph A  mel [1,F,128] -> chunk_embeds [1,F/8,512]                        (fp32 only)
graph B  packed_embeds [1,T,512], attn_bias [1,1,1,T], rope_cos/rope_sin [1,1,T,64] -> logits [1,8T,8]
modes    low_latency: T=541, F=104, row mask "relu"
         offline: T=684 (264 cache + 40 FIFO + 340 chunk + 40 look-ahead), row mask "two_level" (the offline pass
         masks the key of the frame after the last full hop; see nemotron3diar_model.py). The offline host runs
         the low_latency graph A over 104-frame blocks, so no offline graph A is built.
fp16     ai_edge_quantizer FLOAT_CASTING, 16-bit float weights, float compute
--ln     plain (default): nn.LayerNorm -> n3d_encoder_<tag>{,_fp16}.tflite
         safe / safe_guide: SafeLayerNorm v2 on all 64 LayerNorms -> n3d_encoder_<tag>_<ln>{,_fp16}.tflite
         (see nemotron3diar_model.py; graph A has no LayerNorm, so only the plain build writes it)

Usage: .venv/bin/python build_nemotron3diar.py --run-dir <run> --model-dir <dir> [--mode low_latency]
           [--ln plain|safe|safe_guide]
"""

import argparse
import os
import time

import numpy as np
import torch

import nemotron3diar_model as n3d

MODES = {"low_latency": (541, 104, "ll"), "offline": (684, 3040, "off")}


def to_fp16(fp32, fp16):
  from ai_edge_quantizer import quantizer, recipe_manager
  from ai_edge_quantizer.recipe import AlgorithmName, qtyping
  rm = recipe_manager.RecipeManager()
  rm.add_quantization_config(
      regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      op_config=qtyping.OpQuantizationConfig(
          weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
          compute_precision=qtyping.ComputePrecision.FLOAT),
      algorithm_key=AlgorithmName.FLOAT_CASTING)
  if os.path.exists(fp16):
    os.remove(fp16)
  q = quantizer.Quantizer(float_model=fp32)
  q.load_quantization_recipe(rm.get_quantization_recipe())
  q.quantize().export_model(fp16)
  return fp16


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--mode", choices=sorted(MODES), default="low_latency")
  ap.add_argument("--ln", choices=sorted(n3d.LN_MODES), default="plain")
  ap.add_argument("--skip-frontend", action="store_true")
  args = ap.parse_args()
  t, f, tag = MODES[args.mode]
  exports = os.path.join(args.run_dir, "exports")
  os.makedirs(exports, exist_ok=True)

  import litert_torch  # after torch; the converter pulls in jax

  offline = args.mode == "offline"
  frontend, encoder, _, n_params = n3d.load_models(
      os.path.join(args.model_dir, "model.safetensors"), ln_mode=args.ln, row_mask="two_level" if offline else "relu")
  print(f"strict load OK: {n_params} params; mode={args.mode} T={t} F={f} ln={args.ln} row_mask={encoder.row_mask}",
        flush=True)

  if not args.skip_frontend and args.ln == "plain" and not offline:
    path_a = os.path.join(exports, "n3d_frontend.tflite") if args.mode == "low_latency" else \
        os.path.join(exports, f"n3d_frontend_{tag}.tflite")
    t0 = time.time()
    mel = torch.randn(1, f, 128) * 4.0 - 8.0
    litert_torch.convert(n3d.FrontendIO(frontend).eval(), sample_args=None,
                         sample_kwargs={"mel": mel}).export(path_a)
    print(f"graph A -> {path_a} {os.path.getsize(path_a)} B in {time.time() - t0:.1f}s", flush=True)

  suffix = "" if args.ln == "plain" else f"_{args.ln}"
  path_b = os.path.join(exports, f"n3d_encoder_{tag}{suffix}.tflite")
  cos, sin = n3d.rope_tables(t)
  length = t - 17 if t > 17 else t
  kwargs = {
      "packed_embeds": torch.from_numpy(n3d.pack(np.random.RandomState(0).randn(length, 512).astype(np.float32), t)),
      "attn_bias": torch.from_numpy(n3d.attn_bias_for(length, t, two_level=offline)),
      "rope_cos": cos,
      "rope_sin": sin,
  }
  t0 = time.time()
  litert_torch.convert(n3d.EncoderIO(encoder).eval(), sample_args=None, sample_kwargs=kwargs).export(path_b)
  print(f"graph B fp32 -> {path_b} {os.path.getsize(path_b)} B in {time.time() - t0:.1f}s", flush=True)

  t0 = time.time()
  path_b16 = path_b.replace(".tflite", "_fp16.tflite")
  to_fp16(path_b, path_b16)
  print(f"graph B fp16 -> {path_b16} {os.path.getsize(path_b16)} B in {time.time() - t0:.1f}s", flush=True)
  print("BUILD DONE", flush=True)
  os._exit(0)


if __name__ == "__main__":
  main()

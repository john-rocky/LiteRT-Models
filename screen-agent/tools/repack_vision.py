#!/usr/bin/env python3
"""Swap the two vision tflites of an existing .litertlm bundle and repack.

Only the vision encoder + adapter sections change; prefill_decode, embedder,
tokenizer and both metadata sections are carried over byte for byte from the
source bundle. That keeps the (expensive, already-shipped) text quantization
untouched — see ../README.md.

Usage:
  repack_vision.py --src model.litertlm --vision-dir out_vision_fixb \
      --out model_fixB.litertlm
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_CLI = os.path.expanduser(
    "~/code/litertlm-convert/.venv-lt016/bin/litert-lm"
)


def section_for(unpack_dir, model_type):
    """Find the unpacked section file for a model_type, via model.toml."""
    toml = open(os.path.join(unpack_dir, "model.toml")).read()
    marker = f'model_type = "{model_type}"'
    if marker not in toml:
        raise SystemExit(f"{model_type} section not found in model.toml")
    tail = toml.split(marker, 1)[1]
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith("data_path"):
            return os.path.join(unpack_dir, line.split("=", 1)[1].strip().strip('"'))
    raise SystemExit(f"no data_path after {model_type}")


def pick(vision_dir, stem):
    """Prefer the quantized export, fall back to the float one."""
    for candidate in (f"{stem}_quantized.tflite", f"{stem}.tflite"):
        path = os.path.join(vision_dir, candidate)
        if os.path.exists(path):
            return path
    raise SystemExit(f"no {stem}[_quantized].tflite in {vision_dir} "
                     f"(have: {[os.path.basename(p) for p in glob.glob(vision_dir + '/*.tflite')]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--vision-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cli", default=DEFAULT_CLI)
    args = ap.parse_args()

    encoder = pick(args.vision_dir, "vision_encoder")
    adapter = pick(args.vision_dir, "vision_adapter")

    with tempfile.TemporaryDirectory(prefix="repack_vision_") as td:
        unpack = os.path.join(td, "unpack")
        subprocess.run([args.cli, "unpack", args.src, "--output-dir", unpack],
                       check=True, capture_output=True, text=True)

        for model_type, new in (("vision_encoder", encoder),
                                ("vision_adapter", adapter)):
            dst = section_for(unpack, model_type)
            print(f"{model_type}: {os.path.getsize(dst)} -> {os.path.getsize(new)} bytes")
            shutil.copyfile(new, dst)

        subprocess.run([args.cli, "pack", os.path.join(unpack, "model.toml"),
                        "--output", args.out, "--allow-overwrite"],
                       check=True, capture_output=True, text=True)

    print(f"packed {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

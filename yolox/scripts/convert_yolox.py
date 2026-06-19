#!/usr/bin/env python3
"""
YOLOX-Nano (Apache-2.0, Megvii) -> GPU-clean TFLite for LiteRT CompiledModel.

YOLOX is a pure CNN detector, so the onnx2tf path (NCHW->NHWC) is the right
tool — unlike DETR-family detectors, it needs NO op rewrites. The exported ONNX
already emits the raw head output [1, 3549, 85] (no grid decode, no NMS baked in),
which keeps the graph free of GATHER/TOPK/CAST. Decode + NMS are done on-device
in Kotlin (see yolox/.../YoloxDetector.kt).

Verified GPU-clean (FP32): 0 banned ops, 0 Flex ops, 0 >4D tensors, 0 dynamic
dims. Op set: CONV_2D, DEPTHWISE_CONV_2D, LOGISTIC+MUL (SiLU), CONCATENATION,
ADD, PAD, STRIDED_SLICE (Focus stem), MAX_POOL_2D (SPP), RESHAPE, TRANSPOSE,
RESIZE_NEAREST_NEIGHBOR — all GPU-native on ML Drift.

Requirements:
    pip install tensorflow==2.19.0 onnx2tf onnx onnxsim onnx_graphsurgeon \
        sng4onnx ai_edge_litert psutil

Usage:
    python convert_yolox.py
    cp output/yolox_nano/yolox_nano_float32.tflite \
       ../yolox/src/main/assets/yolox_nano.tflite

I/O contract (matches YoloxDetector.kt):
    input : images   [1, 416, 416, 3]  NHWC, BGR, 0-255, NO normalization
    output: Identity [1, 3549, 85]     85 = 4 box (x,y,w,h) + 1 obj + 80 class
            obj/class are sigmoid'd in-graph; box decode (grid+stride) is NOT.
"""

import os
import subprocess
import urllib.request

ONNX_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_nano.onnx"
)


def main():
    os.makedirs("output", exist_ok=True)
    onnx_path = "output/yolox_nano.onnx"

    if not os.path.exists(onnx_path):
        print(f"Downloading YOLOX-Nano ONNX...\n  {ONNX_URL}")
        urllib.request.urlretrieve(ONNX_URL, onnx_path)

    out_dir = "output/yolox_nano"
    print("Converting ONNX -> TFLite via onnx2tf (NCHW -> NHWC, batch 1)...")
    # onnx2tf downloads a calibration .npy on first run for its accuracy check;
    # if that download is flaky, drop calibration_image_sample_data_*.npy into CWD.
    subprocess.run(
        ["onnx2tf", "-i", onnx_path, "-o", out_dir, "-b", "1"],
        check=True,
    )

    fp32 = os.path.join(out_dir, "yolox_nano_float32.tflite")
    if os.path.exists(fp32):
        size_mb = os.path.getsize(fp32) / 1e6
        print(f"\nSuccess! {fp32} ({size_mb:.1f} MB)")
        print("Copy to: yolox/src/main/assets/yolox_nano.tflite")
    else:
        print("Conversion finished; check output dir:")
        for f in sorted(os.listdir(out_dir)):
            print(f"  {f}")


if __name__ == "__main__":
    main()

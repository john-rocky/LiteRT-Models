#!/usr/bin/env python3
"""
Apply mixed precision to ONNX model: keep Softmax and LayerNorm in FP32,
convert everything else to FP16.

Usage:
    python convert_onnx_mixed_precision.py \
        --input depth_anything_v2.onnx \
        --output depth_anything_v2_mixed.onnx
"""

import argparse
import numpy as np
from onnxconverter_common import float16


def convert_mixed_precision(input_path: str, output_path: str):
    """Convert ONNX model to FP16 with critical ops kept in FP32."""
    import onnx

    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)

    # Ops that must stay in FP32 to avoid NaN/Inf in ViT
    fp32_ops = {
        "Softmax",
        "LayerNormalization",
        "ReduceMean",
        "Pow",
        "Sqrt",
        "InstanceNormalization",
        "BatchNormalization",
    }

    # Find nodes with these op types
    block_list = []
    for node in model.graph.node:
        if node.op_type in fp32_ops:
            block_list.append(node.name)

    print(f"Keeping {len(block_list)} nodes in FP32: {fp32_ops}")
    print(f"Converting remaining nodes to FP16")

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=block_list
    )

    onnx.save(model_fp16, output_path)
    print(f"Saved: {output_path}")

    # Verify
    original_size = len(onnx.load(input_path).SerializeToString())
    mixed_size = len(model_fp16.SerializeToString())
    print(f"Size reduction: {original_size/1024/1024:.1f} MB -> {mixed_size/1024/1024:.1f} MB "
          f"({mixed_size/original_size*100:.0f}%)")


def auto_mixed_precision(input_path: str, output_path: str):
    """Automatically find the minimal set of FP32 ops using validation data."""
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import auto_mixed_precision as amp

    print(f"Running automatic mixed precision analysis...")

    model = onnx.load(input_path)

    # Create sample input for validation
    sess = ort.InferenceSession(input_path)
    input_info = sess.get_inputs()[0]
    input_shape = input_info.shape
    # Replace dynamic dims
    shape = [s if isinstance(s, int) else 1 for s in input_shape]
    sample = np.random.randn(*shape).astype(np.float32) * 0.229 + 0.485
    feed_dict = {input_info.name: sample.astype(np.float32)}

    model_fp16 = amp.auto_convert_mixed_precision(
        model,
        feed_dict,
        rtol=0.01,
        atol=0.001,
        keep_io_types=True
    )

    onnx.save(model_fp16, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output ONNX model path")
    parser.add_argument("--auto", action="store_true",
                        help="Use automatic mixed precision (slower but more precise)")
    args = parser.parse_args()

    if args.auto:
        auto_mixed_precision(args.input, args.output)
    else:
        convert_mixed_precision(args.input, args.output)

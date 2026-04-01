#!/usr/bin/env python3
"""
Export Depth Anything V2 Small as NHWC TFLite for GPU delegate.

The key insight: TFLite GPU delegate requires NHWC layout to work effectively.
ai-edge-torch/litert-torch exports NCHW by default (PyTorch convention),
which causes GPU delegate to fall back to CPU for most ops.
"""

import os
import torch
import torch.nn as nn

INPUT_H, INPUT_W = 518, 518
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "app", "src", "main", "assets")


class NHWCWrapper(nn.Module):
    """Wraps DepthAnything to accept NHWC input [1,H,W,3] and output [1,H,W,1]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [1, H, W, 3] (NHWC) -> [1, 3, H, W] (NCHW) for model
        x = x.permute(0, 3, 1, 2)
        depth = self.model(pixel_values=x).predicted_depth
        # depth: [1, H, W] -> [1, H, W, 1]
        return depth.unsqueeze(-1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Depth-Anything-V2-Small from HuggingFace...")
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model.eval()

    wrapper = NHWCWrapper(model)

    # Test forward pass
    dummy = torch.randn(1, INPUT_H, INPUT_W, 3)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Test forward: input={dummy.shape} -> output={out.shape}")

    # Export via litert-torch
    print("\nExporting NHWC TFLite...")
    import litert_torch

    nhwc_path = os.path.join(OUTPUT_DIR, "depth_anything_v2_nhwc.tflite")
    edge_model = litert_torch.convert(wrapper, (dummy,))
    edge_model.export(nhwc_path)
    print(f"  OK  {os.path.getsize(nhwc_path)/1e6:.1f} MB")

    # Verify the exported model
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=nhwc_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"\nVerification:")
    print(f"  Input:  {inp['shape']} {inp['dtype']} {inp['name']}")
    print(f"  Output: {out_d['shape']} {out_d['dtype']} {out_d['name']}")

    # Count ops
    import io, sys, contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        tf.lite.experimental.Analyzer.analyze(model_path=nhwc_path, gpu_compatibility=True)
    output = f.getvalue()
    op_lines = [l for l in output.split('\n') if l.strip().startswith('Op#')]
    gpu_warn = [l for l in output.split('\n') if 'GPU COMPATIBILITY' in l and 'WARNING' in l]
    print(f"  Total ops: {len(op_lines)}")
    print(f"  GPU warnings: {len(gpu_warn)}")
    if gpu_warn:
        for w in gpu_warn[:10]:
            print(f"    {w.strip()}")

    print(f"\nDone! Model saved to: {nhwc_path}")


if __name__ == "__main__":
    main()

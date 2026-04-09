"""
Convert OSNet x0.25 (person Re-ID) to TFLite for LiteRT CompiledModel GPU.

OSNet is a pure CNN (no ViT attention), so conversion is straightforward.
Uses litert-torch via litert_gpu_toolkit.

Prerequisites:
    pip install torchreid litert-torch

Usage:
    python convert_osnet.py

Output: osnet_x0_25.tflite (~1.4 MB fp16)
    Push to app/src/main/assets/ or device.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn


def build_osnet_x0_25():
    """Build OSNet x0.25 and load pretrained weights from torchreid."""
    from torchreid.models import build_model

    model = build_model(
        name="osnet_x0_25",
        num_classes=1,  # dummy, we only use the feature extractor
        loss="softmax",
        pretrained=True,
    )
    model.eval()
    return model


class OSNetFeatureExtractor(nn.Module):
    """Wraps OSNet to output only the 512-dim feature vector (no classifier)."""

    def __init__(self, osnet):
        super().__init__()
        self.osnet = osnet

    def forward(self, x):
        # torchreid OSNet: call feature extraction only
        # In eval mode with loss="softmax", model(x) returns [features, None]
        # We extract features directly through the backbone.
        x = self.osnet.featuremaps(x)
        v = self.osnet.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.osnet.fc is not None:
            v = self.osnet.fc(v)
        if self.osnet.feature_dim_holder is not None:
            v = self.osnet.feature_dim_holder(v)
        return v


def build_feature_extractor():
    """Build OSNet feature extractor, handling different torchreid versions."""
    from torchreid.models import build_model

    model = build_model(
        name="osnet_x0_25",
        num_classes=1,
        loss="softmax",
        pretrained=True,
    )
    model.eval()

    # Try direct feature extraction first
    dummy = torch.randn(1, 3, 256, 128)
    with torch.no_grad():
        out = model(dummy)

    # torchreid returns tuple in training, tensor in eval
    if isinstance(out, torch.Tensor):
        if out.shape[-1] == 512:
            print(f"Direct model output: {out.shape} - using as-is")
            return model
        # If classifier output, need to extract features
    elif isinstance(out, (list, tuple)):
        out = out[0]
        if out.shape[-1] == 512:
            print(f"Model returns tuple, first element: {out.shape}")

    # Fallback: strip classifier, keep feature extractor
    class FeatureOnly(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            # Remove classifier to get features
            if hasattr(base, 'classifier'):
                base.classifier = nn.Identity()

        def forward(self, x):
            out = self.base(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

    wrapper = FeatureOnly(model)
    wrapper.eval()
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Wrapped model output: {out.shape}")
    return wrapper


def main():
    from litert_gpu_toolkit import convert_for_gpu

    print("Building OSNet x0.25...")
    model = build_feature_extractor()

    # Input: batch=1, C=3, H=256, W=128 (standard person Re-ID size)
    dummy_input = torch.randn(1, 3, 256, 128)

    print("Converting to TFLite...")
    output_path = os.path.join(os.path.dirname(__file__), "osnet_x0_25.tflite")
    convert_for_gpu(
        model=model,
        dummy_input=dummy_input,
        output_path=output_path,
        check=True,
        verbose=True,
    )

    print(f"\nDone! Model saved to: {output_path}")
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Size: {size_mb:.1f} MB")
    print(f"\nCopy to assets:")
    print(f"  cp {output_path} ../app/src/main/assets/osnet_x0_25.tflite")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fix B: move LFM2.5-VL's 2x2 pixel-unshuffle from the adapter into the encoder.

Why: LiteRT-LM derives `patch_num_shrink_factor = encoder_input_patches /
adapter_output_tokens` (= 1024/256 = 4) and then forwards only
`input_patches / shrink` = 256 encoder rows into the adapter's 1024-row input.
It assumes the *encoder* already shrank. LFM2.5-VL shrinks in the adapter, so
768 of 1024 rows stay zero and the model sees only the top quarter of the image.
See ../README.md.

After this re-export:

    encoder  images      [1,1024, 768]  ->  features     [1,256,4608]
    adapter  soft_tokens [1, 256,4608]  ->  mm_embedding [1,256,2048]

so shrink stays 4, num_patches = 256, and the 256 forwarded rows are exactly
the 256 pooled tokens the adapter wants. Correct on the released runtime, no
runtime patch needed.

Exports only the two vision tflites — the text/embedder sections of an existing
bundle are unaffected and get reused by tools/repack_vision.py.

Usage:
  reexport_vision_unshuffle.py [hf_id_or_path] [out_dir]
"""
import os
import sys
from types import SimpleNamespace

MODEL = sys.argv[1] if len(sys.argv) > 1 else "LiquidAI/LFM2.5-VL-450M"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "out_vision_fixb"

# Same softmax-composite strip the original conversion applied: released
# litert-converter 0.3.x does not lower the composite and the GPU delegate
# rejects it. Math unchanged. (convert_lfm25_vl3b.py, default path.)
from litert_torch.generative.export_hf.core import attention


class _PassthroughBuilder:
    def __init__(self, *args, **kwargs):
        pass

    def mark_inputs(self, *xs):
        return xs[0] if len(xs) == 1 else xs

    def mark_outputs(self, *xs):
        return xs[0] if len(xs) == 1 else xs


attention.composite = SimpleNamespace(StableHLOCompositeBuilder=_PassthroughBuilder)

import torch  # noqa: E402
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402
from litert_torch.generative.export_hf.core import exportable_module_config  # noqa: E402
from litert_torch.generative.export_hf.model_ext.lfm2_vl import (  # noqa: E402
    vision_exportable as vx,
)

GRID = 32  # patch grid side the vision graph is frozen at (32x32 = 1024 patches)


def _unshuffle(model, features):
    """Apply the projector's pixel_unshuffle to a [1,1024,C] token sequence."""
    projector = model.model.multi_modal_projector
    spatial = features.reshape(1, GRID, GRID, -1)
    pooled = projector.pixel_unshuffle(spatial)          # [1,16,16,C*4]
    return pooled.reshape(1, -1, pooled.size(-1))        # [1,256,C*4]


def _encoder_forward(self, images):
    pixel_attention_mask = torch.ones([1, GRID * GRID], dtype=torch.int32)
    spatial_shapes = torch.tensor([[GRID, GRID]], dtype=torch.int32)
    features = self.model.model.vision_tower(
        pixel_values=images,
        spatial_shapes=spatial_shapes,
        pixel_attention_mask=pixel_attention_mask,
        return_dict=True,
    ).last_hidden_state
    return {"features": _unshuffle(self.model, features)}


def _adapter_forward(self, soft_tokens):
    """soft_tokens are already unshuffled; run the rest of the projector."""
    projector = self.model.model.multi_modal_projector
    hidden = soft_tokens
    if projector.use_layer_norm:
        hidden = projector.layer_norm(hidden)
    hidden = projector.linear_1(hidden)
    hidden = projector.act(hidden)
    hidden = projector.linear_2(hidden)
    return {"mm_embedding": hidden.reshape(1, -1, hidden.size(-1))}


def _adapter_sample_inputs(self, model_config, **kwargs):
    image_processor = kwargs.get("image_processor", None)
    if image_processor is None:
        raise ValueError("Image processor is required for the LFM2-VL adapter.")
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 512, 512))], return_tensors="pt"
    ).pixel_values
    pixel_attention_mask = torch.ones([1, GRID * GRID], dtype=torch.int32)
    spatial_shapes = torch.tensor([[GRID, GRID]], dtype=torch.int32)
    with torch.device("meta"):
        features = self.model.model.vision_tower(
            pixel_values=dummy_image,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        ).last_hidden_state
        pooled = _unshuffle(self.model, features)
    inputs = {"soft_tokens": torch.zeros_like(pooled, dtype=torch.float32)}
    return {"vision_adapter": (inputs, {})}


vx.LiteRTExportableModuleForLFM2VisionEncoder.forward = _encoder_forward
vx.LiteRTExportableModuleForLFM2VisionAdapter.forward = _adapter_forward
vx.LiteRTExportableModuleForLFM2VisionAdapter.get_sample_inputs = _adapter_sample_inputs


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    config = exportable_module_config.ExportableModuleConfig(
        model=MODEL,
        output_dir=OUTDIR,
        task="image_text_to_text",
        work_dir=OUTDIR,
        bundle_litert_lm=False,
    )
    print(f"model={MODEL}\nout={OUTDIR}")
    print(f"vision recipe={config.vision_encoder_quantization_recipe}")

    source = export_lib.load_model(
        MODEL, config, task=export_lib.ExportTask.IMAGE_TEXT_TO_TEXT
    )
    exported = export_lib.ExportedModelArtifacts()
    export_lib.export_vision_encoder_models(source, config, exported)

    print("\nencoder:", exported.vision_encoder_model_path)
    print("adapter:", exported.vision_adapter_model_path)
    for path in (exported.vision_encoder_model_path,
                 exported.vision_adapter_model_path):
        if not path:
            continue
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path)
        print(f"  {os.path.basename(path)}")
        print("    in :", [(d["name"], list(d["shape"])) for d in it.get_input_details()])
        print("    out:", [(d["name"], list(d["shape"])) for d in it.get_output_details()])


if __name__ == "__main__":
    main()

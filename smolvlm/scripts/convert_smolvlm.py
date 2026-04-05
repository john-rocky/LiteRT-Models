#!/usr/bin/env python3
"""
Convert SmolVLM-256M for Android: vision encoder (TFLite GPU) + LM decoder (ONNX CPU).

Requirements:
    pip install transformers litert-torch onnxruntime torch numpy accelerate

Usage:
    python convert_smolvlm.py --output_dir output/
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def patch_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.GELU,)):
            setattr(module, name, SigmoidGELU())
        else:
            patch_gelu(child)


class VisionEncoderWrapper(nn.Module):
    """SigLIP vision encoder + pixel shuffle connector.
    Bypasses torch.bucketize in position embeddings by pre-computing
    position IDs for fixed 512x512 input.

    Input:  [1, 3, 512, 512] float32 image (normalized to [-1, 1])
    Output: [1, 64, 576] float32 visual tokens in language embedding space
    """
    def __init__(self, vision_model, connector):
        super().__init__()
        # Replace padding='valid' Conv2d with padding=0 (litert-torch compat)
        orig_pe = vision_model.embeddings.patch_embedding
        self.patch_embedding = nn.Conv2d(
            orig_pe.in_channels, orig_pe.out_channels,
            kernel_size=orig_pe.kernel_size, stride=orig_pe.stride, padding=0, bias=orig_pe.bias is not None,
        )
        self.patch_embedding.weight = orig_pe.weight
        if orig_pe.bias is not None:
            self.patch_embedding.bias = orig_pe.bias
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm
        self.connector = connector

        # Pre-compute position IDs for 512x512 (32x32=1024 patches, sequential)
        pos_ids = torch.arange(1024).unsqueeze(0)  # [1, 1024]
        self.register_buffer("pos_ids", pos_ids)

    def forward(self, pixel_values):
        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values)  # [1, 768, 32, 32]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)  # [1, 1024, 768]

        # Add pre-computed position embedding
        embeddings = embeddings + self.position_embedding(self.pos_ids)

        # Transformer encoder
        hidden = self.encoder(embeddings).last_hidden_state

        # Post layer norm
        hidden = self.post_layernorm(hidden)

        # Connector (pixel shuffle + projection)
        return self.connector(hidden)


class LMDecoderWrapper(nn.Module):
    """SmolLM2 language model decoder without KV-cache.

    Input:  input_embeds [1, seq_len, 576] float32
    Output: logits [1, seq_len, vocab_size] float32
    """
    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds):
        outputs = self.text_model(inputs_embeds=inputs_embeds)
        return self.lm_head(outputs.last_hidden_state)


def convert_vision_encoder(model, output_dir):
    """Convert vision encoder + connector to TFLite."""
    import litert_torch

    encoder = VisionEncoderWrapper(model.model.vision_model, model.model.connector)
    encoder.eval()
    patch_gelu(encoder)

    _orig_gelu = F.gelu
    F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)

    dummy = torch.randn(1, 3, 512, 512)
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Converting vision encoder ({params:,} params)...")

    with torch.no_grad():
        ref = encoder(dummy)
        print(f"  Output: {ref.shape}")  # [1, 64, 576]

    result = litert_torch.convert(encoder, (dummy,))
    out_path = os.path.join(output_dir, "smolvlm_vision.tflite")
    result.export(out_path)

    F.gelu = _orig_gelu

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Vision encoder saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [1, 3, 512, 512] float32 (normalized to [-1, 1])")
    print(f"  Output: [1, 64, 576] float32 visual tokens")
    return out_path


def convert_lm_decoder(model, output_dir):
    """Convert language model to ONNX."""
    # Patch causal mask creation (transformers v5.5 masking_utils incompatible with ONNX trace)
    import transformers.masking_utils as masking_utils
    import transformers.models.llama.modeling_llama as llama_module

    def simple_causal_mask(config, inputs_embeds, attention_mask=None, cache_position=None, **kwargs):
        seq_len = inputs_embeds.shape[1]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    masking_utils.create_causal_mask = simple_causal_mask
    llama_module.create_causal_mask = simple_causal_mask

    decoder = LMDecoderWrapper(model.model.text_model, model.lm_head)
    decoder.eval()

    embed_dim = model.model.text_model.embed_tokens.weight.shape[1]
    vocab_size = model.lm_head.weight.shape[0]

    # Dummy: 64 visual tokens + 10 text tokens
    dummy_embeds = torch.randn(1, 74, embed_dim)

    params = sum(p.numel() for p in decoder.parameters())
    print(f"Exporting LM decoder ({params:,} params)...")

    out_path = os.path.join(output_dir, "smolvlm_decoder.onnx")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                (dummy_embeds,),
                out_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["inputs_embeds"],
                output_names=["logits"],
                dynamic_axes={
                    "inputs_embeds": {1: "seq_len"},
                    "logits": {1: "seq_len"},
                },
                dynamo=False,
            )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"LM decoder saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  inputs_embeds [1, seq_len, {embed_dim}] float32")
    print(f"  Output: logits [1, seq_len, {vocab_size}] float32")
    return out_path


def export_embed_and_vocab(model, output_dir):
    """Export token embeddings and tokenizer vocab."""
    from transformers import AutoProcessor

    # Token embeddings matrix [vocab_size, embed_dim]
    embed_weights = model.model.text_model.embed_tokens.weight.detach().numpy().astype(np.float32)
    embed_path = os.path.join(output_dir, "embed_tokens.bin")
    embed_weights.tofile(embed_path)

    vocab_size, embed_dim = embed_weights.shape
    print(f"Token embeddings saved: {embed_path} ({vocab_size} x {embed_dim}, "
          f"{os.path.getsize(embed_path) / 1e6:.1f} MB)")

    # Tokenizer vocab
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    tokenizer = processor.tokenizer

    # Export vocab as JSON {id: token}
    vocab = {}
    for token, idx in tokenizer.get_vocab().items():
        vocab[idx] = token

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Vocab saved: {vocab_path} ({len(vocab)} tokens)")

    # Config
    config = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "image_token_id": tokenizer.convert_tokens_to_ids("<image>"),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "n_visual_tokens": 64,
        "image_size": 512,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")

    return embed_path, vocab_path, config_path


def main():
    parser = argparse.ArgumentParser(description="Convert SmolVLM-256M for Android")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SmolVLM-256M-Instruct...")
    from transformers import Idefics3ForConditionalGeneration
    model = Idefics3ForConditionalGeneration.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.float32,
    )
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total:,} params")

    vis_path = convert_vision_encoder(model, args.output_dir)
    dec_path = convert_lm_decoder(model, args.output_dir)
    export_embed_and_vocab(model, args.output_dir)

    print(f"\nDone! Files:")
    print(f"  Large (adb push): {vis_path}, {dec_path}")
    print(f"  Assets: output/embed_tokens.bin, output/vocab.json, output/config.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert CLIP ViT-B/32 for Android: image encoder (TFLite GPU) + pre-computed text embeddings.

The image encoder is converted via litert-torch for CompiledModel GPU inference.
Text embeddings are pre-computed for default labels and saved as binary files.
Optionally exports the text encoder to ONNX for on-device custom label support.

Requirements:
    pip install open_clip_torch litert-torch torch numpy

Usage:
    python convert_clip.py --output_dir output/
    python convert_clip.py --output_dir output/ --export_text_encoder
    python convert_clip.py --output_dir output/ --verify
"""

import argparse
import os
import struct
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default labels for zero-shot classification
DEFAULT_LABELS = [
    # Animals
    "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "elephant",
    "bear", "zebra", "lion", "tiger", "monkey", "rabbit", "deer", "frog",
    "butterfly", "spider", "snake", "penguin",
    # Vehicles
    "car", "truck", "bicycle", "motorcycle", "airplane", "boat", "train", "bus",
    # Food & Drink
    "pizza", "hamburger", "sushi", "rice", "apple", "banana", "cake", "coffee",
    "bread", "salad", "ice cream", "chocolate",
    # Objects
    "phone", "laptop", "book", "chair", "table", "clock", "umbrella", "backpack",
    "guitar", "camera", "glasses", "shoe", "hat", "watch", "key", "bottle",
    # Nature & Scenery
    "mountain", "ocean", "forest", "sunset", "flower", "tree", "sky", "beach",
    "waterfall", "snow", "desert", "river", "cloud", "moon", "star",
    # People & Activities
    "person", "baby", "face", "hand", "crowd",
    # Places & Structures
    "building", "house", "bridge", "tower", "street", "park", "church", "castle",
    # Scenes
    "food", "animal", "vehicle", "landscape", "cityscape", "indoor", "outdoor",
    "selfie", "screenshot", "document", "painting", "sign",
]

# CLIP prompt template for better zero-shot accuracy
PROMPT_TEMPLATE = "a photo of a {}"


class SigmoidGELU(nn.Module):
    """GELU approximation: x * sigmoid(1.702 * x).
    Required because Erf is not a native TFLite op.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def patch_gelu(module):
    """Replace all nn.GELU modules with SigmoidGELU."""
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, SigmoidGELU())
        else:
            patch_gelu(child)


class CLIPImageEncoder(nn.Module):
    """Wraps CLIP's visual encoder to output L2-normalized embeddings."""
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x):
        features = self.visual(x)
        # L2 normalize
        return features / features.norm(dim=-1, keepdim=True)


def convert_image_encoder(clip_model, output_dir, image_size=224):
    """Convert CLIP image encoder to TFLite via litert-torch."""
    import litert_torch

    encoder = CLIPImageEncoder(clip_model)
    encoder.eval()
    patch_gelu(encoder)

    # Patch F.gelu for any functional calls
    original_gelu = F.gelu
    F.gelu = lambda input, approximate='none': input * torch.sigmoid(1.702 * input)

    dummy = torch.randn(1, 3, image_size, image_size)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"Converting image encoder ({params:,} params)...")

    with torch.no_grad():
        ref_out = encoder(dummy)
        print(f"  Reference output shape: {ref_out.shape}, norm: {ref_out.norm().item():.4f}")

    result = litert_torch.convert(encoder, (dummy,))

    out_path = os.path.join(output_dir, "clip_image_encoder.tflite")
    result.export(out_path)

    F.gelu = original_gelu

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Image encoder saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [1, 3, {image_size}, {image_size}] NCHW float32")
    print(f"  Output: [1, 512] L2-normalized float32")
    return out_path


def compute_text_embeddings(clip_model, tokenizer, labels, output_dir):
    """Pre-compute text embeddings for default labels."""
    prompts = [PROMPT_TEMPLATE.format(label) for label in labels]

    print(f"Computing text embeddings for {len(labels)} labels...")
    tokens = tokenizer(prompts)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embeddings = text_features.numpy().astype(np.float32)
    embed_dim = embeddings.shape[1]

    # Save labels
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(label + "\n")

    # Save embeddings as raw float32 binary [num_labels, embed_dim]
    embeddings_path = os.path.join(output_dir, "text_embeddings.bin")
    with open(embeddings_path, "wb") as f:
        # Header: num_labels (int32) + embed_dim (int32)
        f.write(struct.pack("<ii", len(labels), embed_dim))
        f.write(embeddings.tobytes())

    size_kb = os.path.getsize(embeddings_path) / 1024
    print(f"Text embeddings saved: {embeddings_path} ({size_kb:.1f} KB)")
    print(f"  Labels: {len(labels)}, dim: {embed_dim}")
    print(f"  Labels file: {labels_path}")

    return labels_path, embeddings_path


def export_text_encoder(clip_model, tokenizer, output_dir):
    """Export text encoder to ONNX for on-device custom label support."""
    print("Exporting text encoder to ONNX...")

    class CLIPTextEncoder(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.token_embedding = clip_model.token_embedding
            self.positional_embedding = clip_model.positional_embedding
            self.transformer = clip_model.transformer
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection
            self.attn_mask = clip_model.attn_mask

        def forward(self, text):
            x = self.token_embedding(text)
            x = x + self.positional_embedding
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = self.ln_final(x)
            # Take features from EOT token (highest token id in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            x = x @ self.text_projection
            # L2 normalize
            return x / x.norm(dim=-1, keepdim=True)

    text_encoder = CLIPTextEncoder(clip_model)
    text_encoder.eval()

    dummy_tokens = tokenizer(["a photo of a cat"])
    out_path = os.path.join(output_dir, "clip_text_encoder.onnx")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            torch.onnx.export(
                text_encoder,
                dummy_tokens,
                out_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["text_embedding"],
                dynamic_axes={"input_ids": {0: "batch"}, "text_embedding": {0: "batch"}},
                dynamo=False,
            )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Text encoder saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [batch, 77] int64 token IDs")
    print(f"  Output: [batch, 512] L2-normalized float32")

    # Also export the tokenizer vocabulary for on-device BPE
    export_bpe_vocab(tokenizer, output_dir)

    return out_path


def export_bpe_vocab(tokenizer, output_dir):
    """Export BPE vocabulary for the Kotlin tokenizer."""
    import open_clip

    # Get the SimpleTokenizer instance
    tok = tokenizer
    # open_clip tokenizer is a callable; the actual SimpleTokenizer is inside
    # We need to find the BPE merges and vocabulary

    # The vocabulary is in the tokenizer's encoder dict
    # For open_clip, we can access it via the internal _tokenizer
    try:
        inner = tok._tokenizer if hasattr(tok, '_tokenizer') else tok.tokenizer
    except AttributeError:
        print("  Warning: Could not extract BPE vocabulary for on-device tokenizer")
        return

    vocab_path = os.path.join(output_dir, "bpe_vocab.txt")
    merges_path = os.path.join(output_dir, "bpe_merges.txt")

    # Export vocabulary (token -> id)
    if hasattr(inner, 'encoder'):
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, idx in sorted(inner.encoder.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
        print(f"  BPE vocab: {vocab_path} ({len(inner.encoder)} tokens)")

    # Export merge rules
    if hasattr(inner, 'bpe_ranks'):
        with open(merges_path, "w", encoding="utf-8") as f:
            for pair, rank in sorted(inner.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"  BPE merges: {merges_path} ({len(inner.bpe_ranks)} rules)")


def verify(encoder_path, labels_path, embeddings_path, clip_model, tokenizer):
    """Verify converted image encoder against PyTorch reference."""
    print("\nVerifying image encoder accuracy...")

    # Random test image
    np.random.seed(42)
    img = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_features = clip_model.encode_image(torch.from_numpy(img))
        pt_features = pt_features / pt_features.norm(dim=-1, keepdim=True)
        pt_out = pt_features.numpy()

    # TFLite
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=encoder_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], img)
    interp.invoke()
    tf_out = interp.get_tensor(interp.get_output_details()[0]["index"])

    from scipy.stats import pearsonr
    corr, _ = pearsonr(tf_out.flatten(), pt_out.flatten())
    cos_sim = np.dot(tf_out.flatten(), pt_out.flatten())
    print(f"  Correlation: {corr:.6f} (target: >0.98)")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    # Test classification with a real-ish image
    print("\nTesting zero-shot classification...")
    labels = open(labels_path).read().strip().split("\n")
    with open(embeddings_path, "rb") as f:
        num_labels, embed_dim = struct.unpack("<ii", f.read(8))
        text_emb = np.frombuffer(f.read(), dtype=np.float32).reshape(num_labels, embed_dim)

    # Similarity scores using TFLite output
    scores = tf_out @ text_emb.T
    top5 = np.argsort(scores[0])[::-1][:5]
    for i, idx in enumerate(top5):
        print(f"  #{i+1}: {labels[idx]:20s} ({scores[0][idx]:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Convert CLIP ViT-B/32 for Android")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, default="ViT-B-32",
                        help="OpenCLIP model name (default: ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights (default: openai)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Custom labels file (one per line)")
    parser.add_argument("--export_text_encoder", action="store_true",
                        help="Also export text encoder to ONNX")
    parser.add_argument("--verify", action="store_true",
                        help="Run accuracy verification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model} (pretrained={args.pretrained})...")
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    clip_model.eval()
    print(f"  Image size: {clip_model.visual.image_size}")
    print(f"  Embed dim: {clip_model.visual.output_dim}")

    # Convert image encoder
    image_size = clip_model.visual.image_size
    if isinstance(image_size, (list, tuple)):
        image_size = image_size[0]
    enc_path = convert_image_encoder(clip_model, args.output_dir, image_size)

    # Compute text embeddings
    labels = DEFAULT_LABELS
    if args.labels:
        labels = open(args.labels).read().strip().split("\n")
        print(f"Using custom labels from {args.labels}")
    labels_path, emb_path = compute_text_embeddings(
        clip_model, tokenizer, labels, args.output_dir
    )

    # Optionally export text encoder
    if args.export_text_encoder:
        export_text_encoder(clip_model, tokenizer, args.output_dir)

    if args.verify:
        verify(enc_path, labels_path, emb_path, clip_model, tokenizer)

    print(f"\nDone! Copy to Android assets/:")
    print(f"  cp {enc_path} clip/app/src/main/assets/")
    print(f"  cp {labels_path} clip/app/src/main/assets/")
    print(f"  cp {emb_path} clip/app/src/main/assets/")


if __name__ == "__main__":
    main()

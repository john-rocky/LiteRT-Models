#!/usr/bin/env python3
"""
Native Keras reimplementation of DepthAnything V2 Small for TFLite GPU.

Builds the model natively in TF/Keras with NHWC layout, loads PyTorch weights
with proper transposition, and exports to TFLite. This avoids the corr=0.845
quality loss from onnx2tf's NCHW→NHWC conversion.

Result: corr=0.9995 vs PyTorch, 9ms on Pixel 8a ML Drift GPU.

Architecture: DINOv2 ViT-S/14 encoder + DPT decoder head
  - hidden_size=384, num_heads=6, num_layers=12, mlp_ratio=4, patch_size=14
  - 24.8M parameters

Key implementation details (see INVESTIGATION.md for full know-how):
  - All ops in NHWC for ML Drift GPU compatibility
  - Static shapes throughout (no tf.shape()) for TFLite GPU delegate
  - apply_layernorm=True: backbone LayerNorm applied to ALL feature maps
  - Fusion reversal: FusionStage reverses features internally (small→large)
  - Fusion args: (fused_state, current_feature) not (feature, fused)
  - Fusion upsample: target_size from next feature, not fixed 2x
  - head_in_index=-1: head uses LAST (largest) fusion output
  - Head activation: conv→relu order (not relu→conv)
  - tf.image.resize (half_pixel_centers): align_corners=True not GPU-compatible
  - TFLite export: from_keras_model() required (from_concrete_functions loses weights)

Weight transposition: Conv2d [O,I,H,W]→[H,W,I,O], Linear [O,I]→[I,O].T

Requirements:
  pip install torch transformers tensorflow tf_keras numpy

Usage:
  python convert_keras_native.py --output_dir ../app/src/main/assets/
  python convert_keras_native.py --output_dir ./output --input_height 392 --input_width 518
"""

import argparse
import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


# ============================================================
# Model Constants
# ============================================================
HIDDEN_SIZE = 384
NUM_HEADS = 6
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
NUM_LAYERS = 12
MLP_DIM = 1536
PATCH_SIZE = 14
FEATURES_OUT = [2, 5, 8, 11]  # layers to tap for DPT
REASSEMBLE_CHANNELS = [48, 96, 192, 384]
NECK_CHANNELS = 64


# ============================================================
# Custom Layers
# ============================================================

class _PaddedConv2D(tf.keras.layers.Layer):
    """Conv2D with explicit symmetric padding (matching PyTorch padding=N).

    TF's padding="same" is asymmetric for even dimensions with stride>1,
    while PyTorch's padding=1 is always symmetric. This wrapper applies
    symmetric padding manually then uses padding="valid".
    """
    def __init__(self, filters, kernel_size, strides=1, padding=1,
                 data_format="channels_last", name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._pad = padding
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="valid",
            data_format=data_format, name="conv"
        )

    def build(self, input_shape):
        padded_shape = list(input_shape)
        padded_shape[1] = (input_shape[1] or 0) + 2 * self._pad
        padded_shape[2] = (input_shape[2] or 0) + 2 * self._pad
        self.conv.build(padded_shape)
        super().build(input_shape)

    @property
    def kernel(self):
        return self.conv.kernel

    @property
    def bias(self):
        return self.conv.bias

    def call(self, x):
        x = tf.pad(x, [[0, 0], [self._pad, self._pad], [self._pad, self._pad], [0, 0]])
        return self.conv(x)


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_h, patch_w, **kwargs):
        super().__init__(**kwargs)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.proj = tf.keras.layers.Conv2D(
            HIDDEN_SIZE, PATCH_SIZE, strides=PATCH_SIZE,
            padding="valid", data_format="channels_last", name="projection"
        )

    def call(self, x):
        # x: [B, H, W, 3] NHWC
        x = self.proj(x)  # [B, patch_h, patch_w, 384]
        x = tf.reshape(x, [1, self.patch_h * self.patch_w, HIDDEN_SIZE])  # [1, N, 384]
        return x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query = tf.keras.layers.Dense(HIDDEN_SIZE, name="query")
        self.key = tf.keras.layers.Dense(HIDDEN_SIZE, name="key")
        self.value = tf.keras.layers.Dense(HIDDEN_SIZE, name="value")
        self.out_proj = tf.keras.layers.Dense(HIDDEN_SIZE, name="output_dense")
        self.scale = HEAD_DIM ** -0.5

    def call(self, x):
        # x: [1, seq_len, 384] — batch is always 1, seq_len is static
        N = x.shape[1]  # static shape (1370 for 518x518 input)

        q = self.query(x)  # [1, N, 384]
        k = self.key(x)
        v = self.value(x)

        # Reshape to multi-head: [1, N, 6, 64] -> [1, 6, N, 64]
        q = tf.transpose(tf.reshape(q, [1, N, NUM_HEADS, HEAD_DIM]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [1, N, NUM_HEADS, HEAD_DIM]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [1, N, NUM_HEADS, HEAD_DIM]), [0, 2, 1, 3])

        # Attention: [1, 6, N, N]
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        # Apply attention: [1, 6, N, 64]
        out = tf.matmul(attn, v)

        # Merge heads: [1, N, 384]
        out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [1, N, HIDDEN_SIZE])
        out = self.out_proj(out)
        return out


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, layer_idx, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn = MultiHeadSelfAttention(name="attention")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.fc1 = tf.keras.layers.Dense(MLP_DIM, name="mlp_fc1")
        self.fc2 = tf.keras.layers.Dense(HIDDEN_SIZE, name="mlp_fc2")
        # LayerScale params (will be loaded from weights)
        self.layer_scale1 = self.add_weight(
            name="layer_scale1", shape=(HIDDEN_SIZE,),
            initializer="ones", trainable=True
        )
        self.layer_scale2 = self.add_weight(
            name="layer_scale2", shape=(HIDDEN_SIZE,),
            initializer="ones", trainable=True
        )

    def call(self, x):
        # Attention branch
        h = self.norm1(x)
        h = self.attn(h)
        x = x + self.layer_scale1 * h

        # MLP branch
        h = self.norm2(x)
        h = self.fc1(h)
        h = tf.nn.gelu(h, approximate=True)
        h = self.fc2(h)
        x = x + self.layer_scale2 * h

        return x


class PreActResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels, name_prefix="", **kwargs):
        super().__init__(**kwargs)
        self._channels = channels
        self.conv1 = tf.keras.layers.Conv2D(
            channels, 3, padding="same", data_format="channels_last",
            name=f"{name_prefix}convolution1"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            channels, 3, padding="same", data_format="channels_last",
            name=f"{name_prefix}convolution2"
        )

    def build(self, input_shape):
        self.conv1.build(input_shape)
        conv1_out = list(input_shape)
        conv1_out[-1] = self._channels
        self.conv2.build(conv1_out)
        super().build(input_shape)

    def call(self, x):
        residual = x
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x + residual


class FusionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection = tf.keras.layers.Conv2D(
            NECK_CHANNELS, 1, padding="valid", data_format="channels_last",
            name="projection"
        )
        self.residual_layer1 = PreActResidualBlock(
            NECK_CHANNELS, name_prefix="", name="residual_layer1"
        )
        self.residual_layer2 = PreActResidualBlock(
            NECK_CHANNELS, name_prefix="", name="residual_layer2"
        )

    def build(self, input_shape):
        shape_4d = list(input_shape) if len(input_shape) == 4 else [None, None, None, NECK_CHANNELS]
        self.residual_layer1.build(shape_4d)
        self.residual_layer2.build(shape_4d)
        self.projection.build(shape_4d)
        super().build(input_shape)

    def call(self, hidden_state, residual=None, target_size=None):
        if residual is not None:
            # Upsample residual to match hidden_state spatial size
            # Use static shapes from hidden_state
            h = hidden_state.shape[1]
            w = hidden_state.shape[2]
            residual_up = tf.compat.v1.image.resize(residual, [h, w], method="bilinear", align_corners=False)
            hidden_state = hidden_state + self.residual_layer1(residual_up)

        hidden_state = self.residual_layer2(hidden_state)

        # Upsample — PyTorch uses align_corners=True
        if target_size is not None:
            hidden_state = tf.compat.v1.image.resize(
                hidden_state, list(target_size), method="bilinear", align_corners=True
            )
        else:
            # Last layer: scale_factor=2
            h = hidden_state.shape[1]
            w = hidden_state.shape[2]
            hidden_state = tf.compat.v1.image.resize(
                hidden_state, [h * 2, w * 2], method="bilinear", align_corners=True
            )
        hidden_state = self.projection(hidden_state)
        return hidden_state


# ============================================================
# Full Model
# ============================================================

class DepthAnythingV2(tf.keras.Model):
    def __init__(self, input_h=518, input_w=518, **kwargs):
        super().__init__(**kwargs)
        self.input_h = input_h
        self.input_w = input_w
        self.patch_h = input_h // PATCH_SIZE
        self.patch_w = input_w // PATCH_SIZE
        self.num_patches = self.patch_h * self.patch_w

        # Backbone: Patch Embedding
        self.patch_embed = PatchEmbedding(self.patch_h, self.patch_w, name="patch_embed")

        # CLS token and position embeddings
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, HIDDEN_SIZE),
            initializer="zeros", trainable=True
        )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.num_patches + 1, HIDDEN_SIZE),
            initializer="zeros", trainable=True
        )

        # Transformer blocks
        self.blocks = [
            TransformerBlock(i, name=f"block_{i}") for i in range(NUM_LAYERS)
        ]
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="final_norm"
        )

        # Neck: Reassemble
        self.reassemble_projs = []
        self.reassemble_resizes = []
        for i, ch in enumerate(REASSEMBLE_CHANNELS):
            self.reassemble_projs.append(
                tf.keras.layers.Conv2D(
                    ch, 1, padding="valid", data_format="channels_last",
                    name=f"reassemble_proj_{i}"
                )
            )
            if i == 0:
                self.reassemble_resizes.append(
                    tf.keras.layers.Conv2DTranspose(
                        ch, 4, strides=4, padding="valid",
                        data_format="channels_last",
                        name=f"reassemble_resize_{i}"
                    )
                )
            elif i == 1:
                self.reassemble_resizes.append(
                    tf.keras.layers.Conv2DTranspose(
                        ch, 2, strides=2, padding="valid",
                        data_format="channels_last",
                        name=f"reassemble_resize_{i}"
                    )
                )
            elif i == 2:
                self.reassemble_resizes.append(None)  # Identity
            else:  # i == 3: Conv2d(stride=2, padding=1) — must use explicit symmetric padding
                self.reassemble_resizes.append(
                    _PaddedConv2D(
                        ch, 3, strides=2, padding=1,
                        data_format="channels_last",
                        name=f"reassemble_resize_{i}"
                    )
                )

        # Neck: Conv projections (no bias)
        self.neck_convs = [
            tf.keras.layers.Conv2D(
                NECK_CHANNELS, 3, padding="same", use_bias=False,
                data_format="channels_last", name=f"neck_conv_{i}"
            )
            for i in range(4)
        ]

        # Neck: Fusion
        self.fusion_layers = [
            FusionLayer(name=f"fusion_{i}") for i in range(4)
        ]

        # Head
        self.head_conv1 = tf.keras.layers.Conv2D(
            32, 3, padding="same", data_format="channels_last", name="head_conv1"
        )
        self.head_conv2 = tf.keras.layers.Conv2D(
            32, 3, padding="same", data_format="channels_last", name="head_conv2"
        )
        self.head_conv3 = tf.keras.layers.Conv2D(
            1, 1, padding="valid", data_format="channels_last", name="head_conv3"
        )

    def call(self, pixel_values, training=False):
        # All shapes are static: batch=1, input=[1, H, W, 3]

        # ---- Backbone ----
        # Patch embedding: [1, H, W, 3] -> [1, N, 384]
        patch_emb = self.patch_embed(pixel_values)

        # CLS token + position embedding
        cls = tf.broadcast_to(self.cls_token, [1, 1, HIDDEN_SIZE])
        x = tf.concat([cls, patch_emb], axis=1)  # [1, N+1, 384]
        x = x + self.position_embeddings

        # Transformer blocks, save features at specified layers
        features = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in FEATURES_OUT:
                features[i] = x

        x = self.final_norm(x)

        # ---- Neck: Reassemble ----
        reassembled = []
        for idx, layer_idx in enumerate(FEATURES_OUT):
            feat = features[layer_idx]
            # Apply LayerNorm to feature maps (matches PT backbone.apply_layernorm=True)
            feat = self.final_norm(feat)
            # Remove CLS token: [1, N+1, 384] -> [1, N, 384]
            feat = feat[:, 1:, :]
            # Reshape to spatial: [1, patch_h, patch_w, 384]
            feat = tf.reshape(feat, [1, self.patch_h, self.patch_w, HIDDEN_SIZE])
            # Project
            feat = self.reassemble_projs[idx](feat)
            # Resize
            if self.reassemble_resizes[idx] is not None:
                feat = self.reassemble_resizes[idx](feat)
            reassembled.append(feat)

        # ---- Neck: Conv ----
        for i in range(4):
            reassembled[i] = self.neck_convs[i](reassembled[i])

        # ---- Neck: Fusion ----
        # HuggingFace FusionStage reverses internally then processes:
        #   layer(fused_state, current_feature, size=next_feature_size)
        reversed_features = reassembled[::-1]  # [19x19, 37x37, 74x74, 148x148]

        # Pre-compute static target sizes from known spatial dimensions
        # reversed_features spatial sizes: [19, 37, 74, 148] (for 518x518 input)
        target_sizes = []
        for i in range(3):
            h = reversed_features[i + 1].shape[1]
            w = reversed_features[i + 1].shape[2]
            target_sizes.append((h, w))
        target_sizes.append(None)  # Last layer uses scale_factor=2

        fused_outputs = []
        fused = None
        for i in range(4):
            if fused is None:
                fused = self.fusion_layers[i](
                    reversed_features[i], residual=None, target_size=target_sizes[i]
                )
            else:
                fused = self.fusion_layers[i](
                    fused, residual=reversed_features[i], target_size=target_sizes[i]
                )
            fused_outputs.append(fused)

        # ---- Head ---- (head_in_index=-1: uses LAST fusion output, 296x296)
        out = self.head_conv1(fused_outputs[-1])
        # Upsample to original resolution — PyTorch uses align_corners=True
        out = tf.compat.v1.image.resize(
            out, [self.input_h, self.input_w], method="bilinear", align_corners=True
        )
        # Activation AFTER conv (not before): conv → relu
        out = self.head_conv2(out)
        out = tf.nn.relu(out)
        out = self.head_conv3(out)
        out = tf.nn.relu(out)  # + * max_depth (max_depth=1)

        # Squeeze channel dim: [B, H, W, 1] -> [B, H, W]
        out = tf.squeeze(out, axis=-1)
        return out


# ============================================================
# Weight Loading
# ============================================================

def load_pytorch_weights(model, input_h=518, input_w=518):
    """Load weights from PyTorch DepthAnything V2 Small."""
    import torch
    from transformers import AutoModelForDepthEstimation

    print("[2/4] Loading PyTorch weights...")
    pt_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    pt_model.eval()
    sd = pt_model.state_dict()

    def get(name):
        return sd[name].numpy()

    # Build the model first to create all variables
    dummy = tf.zeros([1, input_h, input_w, 3])
    _ = model(dummy)
    print(f"  Model built: {len(model.trainable_variables)} variables")

    loaded = 0

    # --- Backbone: Patch Embedding ---
    # PyTorch Conv2d: [O, I, H, W] -> TF: [H, W, I, O]
    w = get("backbone.embeddings.patch_embeddings.projection.weight")
    model.patch_embed.proj.kernel.assign(np.transpose(w, (2, 3, 1, 0)))
    model.patch_embed.proj.bias.assign(
        get("backbone.embeddings.patch_embeddings.projection.bias")
    )
    loaded += 2

    # --- CLS token and position embeddings ---
    model.cls_token.assign(get("backbone.embeddings.cls_token"))

    # Position embedding interpolation if needed
    # Use PyTorch's interpolation to match exactly (bicubic, align_corners=False)
    pos_emb = get("backbone.embeddings.position_embeddings")  # [1, 1370, 384]
    if pos_emb.shape[1] != model.num_patches + 1:
        cls_pos = pos_emb[:, :1, :]
        patch_pos = pos_emb[:, 1:, :]
        orig_size = int(patch_pos.shape[1] ** 0.5)
        patch_h = input_h // PATCH_SIZE
        patch_w = input_w // PATCH_SIZE
        # Use PyTorch bicubic (matches HuggingFace runtime interpolation exactly)
        patch_pos_pt = torch.from_numpy(
            patch_pos.reshape(1, orig_size, orig_size, HIDDEN_SIZE)
        ).permute(0, 3, 1, 2)  # NHWC -> NCHW
        patch_pos_pt = torch.nn.functional.interpolate(
            patch_pos_pt, size=(patch_h, patch_w),
            mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos_pt.permute(0, 2, 3, 1).numpy()  # NCHW -> NHWC
        patch_pos = patch_pos.reshape(1, -1, HIDDEN_SIZE)
        pos_emb = np.concatenate([cls_pos, patch_pos], axis=1)

    model.position_embeddings.assign(pos_emb)
    loaded += 2

    # --- Transformer Blocks ---
    for i in range(NUM_LAYERS):
        prefix = f"backbone.encoder.layer.{i}"
        block = model.blocks[i]

        # LayerNorm 1
        block.norm1.gamma.assign(get(f"{prefix}.norm1.weight"))
        block.norm1.beta.assign(get(f"{prefix}.norm1.bias"))

        # Attention Q, K, V, Output (Linear: PyTorch [O, I] -> TF Dense [I, O])
        block.attn.query.kernel.assign(get(f"{prefix}.attention.attention.query.weight").T)
        block.attn.query.bias.assign(get(f"{prefix}.attention.attention.query.bias"))
        block.attn.key.kernel.assign(get(f"{prefix}.attention.attention.key.weight").T)
        block.attn.key.bias.assign(get(f"{prefix}.attention.attention.key.bias"))
        block.attn.value.kernel.assign(get(f"{prefix}.attention.attention.value.weight").T)
        block.attn.value.bias.assign(get(f"{prefix}.attention.attention.value.bias"))
        block.attn.out_proj.kernel.assign(get(f"{prefix}.attention.output.dense.weight").T)
        block.attn.out_proj.bias.assign(get(f"{prefix}.attention.output.dense.bias"))

        # LayerScale
        block.layer_scale1.assign(get(f"{prefix}.layer_scale1.lambda1"))
        block.layer_scale2.assign(get(f"{prefix}.layer_scale2.lambda1"))

        # LayerNorm 2
        block.norm2.gamma.assign(get(f"{prefix}.norm2.weight"))
        block.norm2.beta.assign(get(f"{prefix}.norm2.bias"))

        # MLP (Linear: PyTorch [O, I] -> TF Dense [I, O])
        block.fc1.kernel.assign(get(f"{prefix}.mlp.fc1.weight").T)
        block.fc1.bias.assign(get(f"{prefix}.mlp.fc1.bias"))
        block.fc2.kernel.assign(get(f"{prefix}.mlp.fc2.weight").T)
        block.fc2.bias.assign(get(f"{prefix}.mlp.fc2.bias"))

        loaded += 18

    # Final LayerNorm
    model.final_norm.gamma.assign(get("backbone.layernorm.weight"))
    model.final_norm.beta.assign(get("backbone.layernorm.bias"))
    loaded += 2

    # --- Neck: Reassemble ---
    for i in range(4):
        prefix = f"neck.reassemble_stage.layers.{i}"
        # Projection Conv2d: [O, I, H, W] -> [H, W, I, O]
        w = get(f"{prefix}.projection.weight")
        model.reassemble_projs[i].kernel.assign(np.transpose(w, (2, 3, 1, 0)))
        model.reassemble_projs[i].bias.assign(get(f"{prefix}.projection.bias"))
        loaded += 2

        # Resize layer
        if i == 0:
            # ConvTranspose2d: PyTorch [I, O, H, W] -> TF [H, W, O, I]
            w = get(f"{prefix}.resize.weight")
            model.reassemble_resizes[i].kernel.assign(np.transpose(w, (2, 3, 1, 0)))
            model.reassemble_resizes[i].bias.assign(get(f"{prefix}.resize.bias"))
            loaded += 2
        elif i == 1:
            w = get(f"{prefix}.resize.weight")
            model.reassemble_resizes[i].kernel.assign(np.transpose(w, (2, 3, 1, 0)))
            model.reassemble_resizes[i].bias.assign(get(f"{prefix}.resize.bias"))
            loaded += 2
        elif i == 2:
            pass  # Identity
        elif i == 3:
            # Regular Conv2d: [O, I, H, W] -> [H, W, I, O]
            w = get(f"{prefix}.resize.weight")
            model.reassemble_resizes[i].kernel.assign(np.transpose(w, (2, 3, 1, 0)))
            model.reassemble_resizes[i].bias.assign(get(f"{prefix}.resize.bias"))
            loaded += 2

    # --- Neck: Conv projections (no bias) ---
    for i in range(4):
        w = get(f"neck.convs.{i}.weight")
        model.neck_convs[i].kernel.assign(np.transpose(w, (2, 3, 1, 0)))
        loaded += 1

    # --- Neck: Fusion ---
    for i in range(4):
        prefix = f"neck.fusion_stage.layers.{i}"
        fl = model.fusion_layers[i]

        # Projection 1x1
        w = get(f"{prefix}.projection.weight")
        fl.projection.kernel.assign(np.transpose(w, (2, 3, 1, 0)))
        fl.projection.bias.assign(get(f"{prefix}.projection.bias"))
        loaded += 2

        # Residual blocks
        for rb_idx, rb_name in enumerate(["residual_layer1", "residual_layer2"]):
            rb = getattr(fl, rb_name)
            for c_idx in [1, 2]:
                w = get(f"{prefix}.{rb_name}.convolution{c_idx}.weight")
                conv = rb.conv1 if c_idx == 1 else rb.conv2
                conv.kernel.assign(np.transpose(w, (2, 3, 1, 0)))
                conv.bias.assign(
                    get(f"{prefix}.{rb_name}.convolution{c_idx}.bias")
                )
                loaded += 2

    # --- Head ---
    for conv_name, tf_conv in [
        ("head.conv1", model.head_conv1),
        ("head.conv2", model.head_conv2),
        ("head.conv3", model.head_conv3),
    ]:
        w = get(f"{conv_name}.weight")
        tf_conv.kernel.assign(np.transpose(w, (2, 3, 1, 0)))
        tf_conv.bias.assign(get(f"{conv_name}.bias"))
        loaded += 2

    print(f"  Loaded {loaded} weight tensors")

    # Verify parameter count
    tf_total = sum(np.prod(v.shape) for v in model.trainable_variables)
    pt_total = sum(p.numel() for p in pt_model.parameters())
    print(f"  TF params: {tf_total:,}, PyTorch params: {pt_total:,}")

    return pt_model


# ============================================================
# Quality Verification
# ============================================================

def verify_quality(tf_model, pt_model, input_h=518, input_w=518):
    """Compare TF and PyTorch model outputs."""
    import torch

    print("\n[3/4] Verifying quality...")
    np.random.seed(42)
    input_np = np.random.rand(1, 3, input_h, input_w).astype(np.float32)

    # PyTorch: NCHW input
    pt_input = torch.from_numpy(input_np)
    with torch.no_grad():
        pt_out = pt_model(pixel_values=pt_input).predicted_depth.numpy().flatten()

    # TF: NHWC input
    tf_input = np.transpose(input_np, (0, 2, 3, 1))  # NCHW -> NHWC
    tf_out = tf_model(tf_input, training=False).numpy().flatten()

    # Metrics
    corr = np.corrcoef(pt_out, tf_out)[0, 1]
    mse = np.mean((pt_out - tf_out) ** 2)
    max_diff = np.max(np.abs(pt_out - tf_out))

    print(f"  Correlation:  {corr:.6f}")
    print(f"  MSE:          {mse:.6f}")
    print(f"  Max diff:     {max_diff:.6f}")
    print(f"  PT range:     [{pt_out.min():.4f}, {pt_out.max():.4f}]")
    print(f"  TF range:     [{tf_out.min():.4f}, {tf_out.max():.4f}]")

    return corr


# ============================================================
# TFLite Export
# ============================================================

def export_tflite(model, output_path, input_h=518, input_w=518):
    """Export Keras model to TFLite."""
    import tempfile, shutil
    print(f"\n[4/4] Exporting TFLite -> {os.path.basename(output_path)}")

    # Use from_keras_model which properly captures all variables
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.optimizations = []

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  FP32 Size: {size_mb:.1f} MB")

    # FP16 weights version
    fp16_path = output_path.replace(".tflite", "_fp16w.tflite")
    converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter2.convert()
    with open(fp16_path, "wb") as f:
        f.write(tflite_fp16)
    print(f"  FP16w Size: {os.path.getsize(fp16_path) / 1e6:.1f} MB")

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Native Keras DepthAnything V2 -> TFLite"
    )
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--input_height", type=int, default=518)
    parser.add_argument("--input_width", type=int, default=518)
    args = parser.parse_args()

    assert args.input_height % PATCH_SIZE == 0
    assert args.input_width % PATCH_SIZE == 0
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Build model
    print("[1/4] Building Keras model...")
    model = DepthAnythingV2(args.input_height, args.input_width)

    # Step 2: Load weights
    pt_model = load_pytorch_weights(model, args.input_height, args.input_width)

    # Step 3: Verify quality
    corr = verify_quality(model, pt_model, args.input_height, args.input_width)

    if corr < 0.99:
        print(f"\n  WARNING: Correlation {corr:.4f} < 0.99, quality mismatch!")
        print("  Continuing with export anyway...")

    # Step 4: Export TFLite
    suffix = f"_{args.input_height}x{args.input_width}" if (args.input_height != 518 or args.input_width != 518) else ""
    output_path = os.path.join(
        args.output_dir, f"depth_anything_v2_keras{suffix}.tflite"
    )
    export_tflite(model, output_path, args.input_height, args.input_width)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Quality: corr={corr:.6f}")
    print(f"  Model:   {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

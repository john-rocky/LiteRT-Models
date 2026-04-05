#!/usr/bin/env python3
"""
Convert Whisper for Android: encoder (TFLite GPU) + decoder (ONNX CPU).

The encoder converts mel spectrograms to audio features via litert-torch.
The decoder runs autoregressive token generation via ONNX Runtime.
Vocabulary is exported for token ID → text decoding on device.

Requirements:
    pip install openai-whisper litert-torch onnxruntime torch numpy

Usage:
    python convert_whisper.py --model tiny --output_dir output/
    python convert_whisper.py --model tiny.en --output_dir output/ --verify
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
    """GELU approximation for TFLite GPU compatibility."""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def patch_gelu(module):
    """Replace nn.GELU with SigmoidGELU."""
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, SigmoidGELU())
        else:
            patch_gelu(child)


class WhisperEncoderWrapper(nn.Module):
    """Wrapper for Whisper encoder: mel spectrogram → audio features.

    Input:  [1, 80, 3000] float32 mel spectrogram
    Output: [1, 1500, d_model] float32 audio features
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel)


class WhisperDecoderWrapper(nn.Module):
    """Wrapper for Whisper decoder without KV-cache.

    For each step, the full token sequence is processed.
    Simple but sufficient for tiny/base models.

    Input:  tokens [1, seq_len] int64, audio_features [1, 1500, d_model]
    Output: logits [1, seq_len, vocab_size]
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)


def convert_encoder(model, output_dir):
    """Convert Whisper encoder to TFLite via litert-torch."""
    import litert_torch

    encoder = WhisperEncoderWrapper(model.encoder)
    encoder.eval()
    patch_gelu(encoder)

    # Patch F.gelu for functional calls
    original_gelu = F.gelu
    F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)

    n_mels = model.dims.n_mels
    dummy = torch.randn(1, n_mels, 3000)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"Converting encoder ({params:,} params)...")

    with torch.no_grad():
        ref = encoder(dummy)
        print(f"  Output: {ref.shape}")

    result = litert_torch.convert(encoder, (dummy,))

    out_path = os.path.join(output_dir, "whisper_encoder.tflite")
    result.export(out_path)

    F.gelu = original_gelu

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Encoder saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [1, {n_mels}, 3000] float32 mel spectrogram")
    print(f"  Output: [1, 1500, {model.dims.n_audio_state}] float32")
    return out_path


def convert_decoder(model, output_dir):
    """Convert Whisper decoder to ONNX."""
    # Disable SDPA to use manual attention (ONNX-compatible)
    from whisper.model import MultiHeadAttention
    MultiHeadAttention.use_sdpa = False

    decoder = WhisperDecoderWrapper(model.decoder)
    decoder.eval()

    n_audio_state = model.dims.n_audio_state
    vocab_size = model.dims.n_vocab

    # Dummy inputs
    dummy_tokens = torch.tensor([[50258, 50259, 50359, 50363]], dtype=torch.long)
    dummy_audio = torch.randn(1, 1500, n_audio_state)

    params = sum(p.numel() for p in decoder.parameters())
    print(f"Exporting decoder ({params:,} params)...")

    out_path = os.path.join(output_dir, "whisper_decoder.onnx")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                (dummy_tokens, dummy_audio),
                out_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["tokens", "audio_features"],
                output_names=["logits"],
                dynamic_axes={
                    "tokens": {1: "seq_len"},
                    "logits": {1: "seq_len"},
                },
                dynamo=False,
            )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Decoder saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  tokens [1, seq_len] int64 + audio [1, 1500, {n_audio_state}]")
    print(f"  Output: logits [1, seq_len, {vocab_size}]")
    return out_path


def export_vocab(model, output_dir):
    """Export tokenizer vocabulary for on-device decoding."""
    import whisper.tokenizer as tokenizer

    # Get the tokenizer for this model
    is_multilingual = model.is_multilingual
    tok = tokenizer.get_tokenizer(is_multilingual)

    # Export vocab: id → token bytes
    encoding = tok.encoding
    vocab = {}
    for token_bytes, token_id in encoding._mergeable_ranks.items():
        try:
            vocab[token_id] = token_bytes.decode("utf-8", errors="replace")
        except Exception:
            vocab[token_id] = ""

    # Add special tokens
    for token_bytes, token_id in encoding._special_tokens.items():
        vocab[token_id] = token_bytes

    # Export as JSON {id: text}
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=None)

    # Export mel filters for Android preprocessing
    import whisper.audio as audio
    mel_filters = audio.mel_filters(torch.device("cpu"), model.dims.n_mels)
    mel_path = os.path.join(output_dir, "mel_filters.bin")
    mel_filters.numpy().astype(np.float32).tofile(mel_path)

    print(f"Vocab saved: {vocab_path} ({len(vocab)} tokens)")
    print(f"Mel filters saved: {mel_path} ({mel_filters.shape})")

    # Export model config
    config = {
        "n_mels": model.dims.n_mels,
        "n_audio_state": model.dims.n_audio_state,
        "n_vocab": model.dims.n_vocab,
        "is_multilingual": is_multilingual,
        "sot_token": tok.sot,
        "eot_token": tok.eot,
        "transcribe_token": tok.transcribe,
        "translate_token": tok.translate,
        "no_timestamps_token": tok.no_timestamps,
        "language_tokens": {
            "en": tok.sot + 1,  # 50259 for multilingual
            "ja": tok.sot + 8 if is_multilingual else -1,
        },
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")

    return vocab_path, mel_path, config_path


def verify(encoder_path, decoder_path, model):
    """Verify converted models against PyTorch reference."""
    print("\nVerifying encoder...")

    np.random.seed(42)
    mel = np.random.randn(1, model.dims.n_mels, 3000).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_out = model.encoder(torch.from_numpy(mel)).numpy()

    # TFLite
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=encoder_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], mel)
    interp.invoke()
    tf_out = interp.get_tensor(interp.get_output_details()[0]["index"])

    from scipy.stats import pearsonr
    corr, _ = pearsonr(tf_out.flatten(), pt_out.flatten())
    print(f"  Encoder correlation: {corr:.6f}")

    print("\nVerifying decoder...")
    tokens = np.array([[50258, 50259, 50359, 50363]], dtype=np.int64)

    with torch.no_grad():
        pt_logits = model.decoder(
            torch.from_numpy(tokens),
            torch.from_numpy(pt_out)
        ).numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(decoder_path)
    onnx_logits = sess.run(None, {
        "tokens": tokens,
        "audio_features": pt_out.astype(np.float32),
    })[0]

    corr2, _ = pearsonr(onnx_logits.flatten(), pt_logits.flatten())
    print(f"  Decoder correlation: {corr2:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Convert Whisper for Android")
    parser.add_argument("--model", type=str, default="tiny",
                        help="Whisper model name (tiny, tiny.en, base, base.en)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading whisper-{args.model}...")
    import whisper
    model = whisper.load_model(args.model, device="cpu")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Total: {total_params:,} (encoder: {enc_params:,}, decoder: {dec_params:,})")
    print(f"  Dims: n_mels={model.dims.n_mels}, d_model={model.dims.n_audio_state}, "
          f"vocab={model.dims.n_vocab}")

    enc_path = convert_encoder(model, args.output_dir)
    dec_path = convert_decoder(model, args.output_dir)
    export_vocab(model, args.output_dir)

    if args.verify:
        verify(enc_path, dec_path, model)

    print(f"\nDone! Copy to Android:")
    print(f"  Large files (push via adb):")
    print(f"    {enc_path}")
    print(f"    {dec_path}")
    print(f"  Small files (assets/):")
    print(f"    output/vocab.json")
    print(f"    output/mel_filters.bin")
    print(f"    output/config.json")


if __name__ == "__main__":
    main()

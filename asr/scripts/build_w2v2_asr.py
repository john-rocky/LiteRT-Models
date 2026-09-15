#!/usr/bin/env python3
"""wav2vec2 CTC ASR -> 2 GPU graphs (fixed window, default 16 s).

This is the build behind litert-community/wav2vec2-base-960h-LiteRT. Same deployment shape
as the KWS ship: the fused graph exceeds the Mali whole-graph shader-compile limit, so ship
frontend + encoder-head separately (each all-GPU):

  frontend: waveform[1,N]      -> feat[1,T,768]      (feature_extractor + feature_projection)
  head    : feat[1,T,768]      -> logits[1,T,vocab]  (pos_conv + encoder layers + lm_head)

Re-authoring is imported from build_w2v2_ctc.py (numerically exact): GELU->tanh-GELU, frontend
GroupNorm->GN4D, pos_conv weight_norm fold, create_bidirectional_mask->None (fixed window).
The CTC head is a plain Linear (no LogSoftmax in-graph). Blank = pad_token_id (0 for the
facebook checkpoints), '|' = word delimiter in the char vocab.

Zero-padding note: there is no in-graph padding mask; short clips are zero-padded to the window
and decoded over the valid frames only (frames() gives the count for n samples).

Env:
  W2V2_MODEL_ID  HF repo id or local dir of a Wav2Vec2ForCTC checkpoint
                 (default facebook/wav2vec2-base-960h; vocab/head width flow from its config)
  W2V2_SEC       window length in seconds (default 16)
  W2V2_WAV       optional 16 kHz wav for a real-speech parity + transcript check
                 (without it the parity runs on a random waveform)

Run: python build_w2v2_asr.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

import build_w2v2_ctc as B                      # swap/fold/patch/opcheck/to_fp16/tfl_run
from transformers import Wav2Vec2ForCTC

HERE = os.path.dirname(os.path.abspath(__file__))
MID = os.environ.get("W2V2_MODEL_ID", "facebook/wav2vec2-base-960h")
SR = 16000
SEC = float(os.environ.get("W2V2_SEC", "16"))
N = int(SR * SEC)
WAV = os.environ.get("W2V2_WAV")
torch.manual_seed(0)


class FrontEnd(nn.Module):
    def __init__(s, w): super().__init__(); s.w = w
    def forward(s, x):
        ef = s.w.feature_extractor(x)
        f, _ = s.w.feature_projection(ef.transpose(1, 2))
        return f


class AsrHead(nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, feat):
        enc = s.m.wav2vec2.encoder
        h = feat + enc.pos_conv_embed(feat)
        if not s.m.config.do_stable_layer_norm:  # post-LN encoder (base): LN before the layers
            h = enc.layer_norm(h)
        h = enc.dropout(h)                       # eval no-op
        for layer in enc.layers:
            h = layer(h, attention_mask=None)[0]
        if s.m.config.do_stable_layer_norm:      # pre-LN encoder (large-lv60): LN after the layers
            h = enc.layer_norm(h)
        return s.m.lm_head(h)                    # [1, T, vocab] raw logits


def load_vocab(mid):
    from huggingface_hub import hf_hub_download
    p = os.path.join(mid, "vocab.json") if os.path.isdir(mid) else hf_hub_download(mid, "vocab.json")
    v = json.load(open(p))
    return {i: t for t, i in v.items()}


def greedy(logits, id2tok, t_valid, blank):
    out, prev = [], -1
    for i in logits[:t_valid].argmax(-1).tolist():
        if i != prev and i != blank:
            out.append(id2tok.get(i, ""))
        prev = i
    return "".join(out).replace("|", " ").replace("<s>", "").replace("</s>", "").strip()


def frames(n_samples, cfg):
    L = n_samples
    for k, s in zip(cfg.conv_kernel, cfg.conv_stride):
        L = (L - k) // s + 1
    return L


def main():
    B.patch_mask()
    m = Wav2Vec2ForCTC.from_pretrained(MID).eval()
    B.swap(m); B.fold_weight_norm(m)
    fe = FrontEnd(m.wav2vec2).eval()
    hd = AsrHead(m).eval()
    id2tok = load_vocab(MID)
    blank = m.config.pad_token_id
    print(f"model={MID} layers={m.config.num_hidden_layers} vocab={m.config.vocab_size} "
          f"feat_extract_norm={m.config.feat_extract_norm} stable_ln={m.config.do_stable_layer_norm} "
          f"window={SEC}s -> frames={frames(N, m.config)}")

    # reference: the whole re-authored model on one window (real speech if W2V2_WAV is set)
    x = torch.zeros(1, N)
    if WAV:
        import torchaudio
        wave, sr = torchaudio.load(WAV)
        assert sr == SR, f"{WAV}: {sr} Hz, need 16 kHz"
        a = wave.mean(0)
        n_real = min(a.shape[0], N)
        x[0, :n_real] = a[:n_real]
    else:
        n_real = N
        x = torch.randn(1, N)
    tv = frames(n_real, m.config)
    with torch.no_grad():
        full = m(x, attention_mask=None).logits
        feat = fe(x); ref = hd(feat)
    print(f"[torch] full-vs-split corr {np.corrcoef(full.numpy().ravel(), ref.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(full-ref).abs().max():.2e}  feat {tuple(feat.shape)} logits {tuple(ref.shape)} valid_frames {tv}")
    if WAV:
        print(f"[torch] TEXT: {greedy(ref[0].numpy(), id2tok, tv, blank)}")

    import litert_torch
    pairs = [("w2v2_asr_frontend", fe, x), ("w2v2_asr_head", hd, feat)]
    for name, mod, dummy in pairs:
        fp32 = os.path.join(HERE, f"{name}.tflite")
        litert_torch.convert(mod, (dummy,)).export(fp32)
        it, clean = B.opcheck(fp32, name)
        got = B.tfl_run(it, dummy.numpy())
        refv = feat.numpy() if name.endswith("frontend") else ref.numpy()
        print(f"[{name}] vs torch corr {np.corrcoef(got.ravel(), refv.ravel())[0,1]:.6f}")
        if clean:
            f16 = B.to_fp16(fp32, os.path.join(HERE, f"{name}_fp16.tflite"))
            B.opcheck(f16, f"{name}_fp16")

    # chained fp16 parity (+ transcript when real speech was given)
    from ai_edge_litert.interpreter import Interpreter
    itf = Interpreter(model_path=os.path.join(HERE, "w2v2_asr_frontend_fp16.tflite")); itf.allocate_tensors()
    ith = Interpreter(model_path=os.path.join(HERE, "w2v2_asr_head_fp16.tflite")); ith.allocate_tensors()
    f16feat = B.tfl_run(itf, x.numpy())
    f16log = B.tfl_run(ith, f16feat)
    corr = np.corrcoef(f16log.ravel(), ref.numpy().ravel())[0, 1]
    print(f"[fp16-chain] corr {corr:.6f} max|d| {np.abs(f16log - ref.numpy()).max():.4f}")
    if WAV:
        print(f"[fp16-chain] TEXT: {greedy(f16log[0], id2tok, tv, blank)}")
    print("[done]")


if __name__ == "__main__":
    main()
    os._exit(0)

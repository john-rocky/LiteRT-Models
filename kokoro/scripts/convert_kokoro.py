#!/usr/bin/env python3
"""
Prepare Kokoro-82M v1.0 for Android.

Downloads the fp16 ONNX model and voice style vectors from HuggingFace,
then precomputes phoneme token IDs for a small set of demo phrases (English
and Japanese) so the Android app can run end-to-end without an on-device
phonemizer (eSpeak NG / open_jtalk integration is phase 2).

Outputs to scripts/output/:
    model_fp16.onnx           — 163 MB, push via adb to files/
    voices/<voice>.bin         — ~510 KB each, push via adb to files/voices/
    demo_phrases.json          — small, copy to app/src/main/assets/
    vocab.json                 — small, copy to app/src/main/assets/

Requirements:
    pip install huggingface_hub misaki[en,ja] numpy
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Default phrase list. Add freely.
DEMO_PHRASES = [
    # English
    {"text": "Hello world. Welcome to on device text to speech.", "language": "en"},
    {"text": "The quick brown fox jumps over the lazy dog.", "language": "en"},
    {"text": "This is a tiny eighty two million parameter neural text to speech model.",
     "language": "en"},
    {"text": "This entire sentence is being generated on your phone, with no internet connection.",
     "language": "en"},
    # Japanese
    {"text": "こんにちは。オンデバイス音声合成のデモンストレーションです。", "language": "ja"},
    {"text": "東京の天気は晴れ、最高気温は二十五度です。", "language": "ja"},
    {"text": "今日も一日、お疲れさまでした。", "language": "ja"},
]

# Voices to bundle. Keep this small — each voice ~510 KB.
DEFAULT_VOICES = [
    "af_heart",     # English American female
    "am_michael",   # English American male
    "bf_emma",      # English British female
    "jf_alpha",     # Japanese female
    "jm_kumo",      # Japanese male
]

KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
MODEL_FILENAME = "onnx/model_fp16.onnx"


def download(out_dir: Path, voices: list[str]) -> Path:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = out_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    print(f"Downloading {MODEL_FILENAME}...")
    model_path = hf_hub_download(KOKORO_REPO, MODEL_FILENAME)
    target_model = out_dir / "model_fp16.onnx"
    shutil.copy(model_path, target_model)
    print(f"  -> {target_model} ({target_model.stat().st_size / 1e6:.1f} MB)")

    for v in voices:
        print(f"Downloading voices/{v}.bin...")
        path = hf_hub_download(KOKORO_REPO, f"voices/{v}.bin")
        target = voices_dir / f"{v}.bin"
        shutil.copy(path, target)
        print(f"  -> {target} ({target.stat().st_size / 1024:.0f} KB)")

    return target_model


def load_vocab() -> dict[str, int]:
    """Vocab embedded directly from hexgrad/Kokoro-82M/config.json."""
    return {
        # Special tokens (punctuation, quotes, space)
        ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6,
        "—": 9, "…": 10, "\"": 11, "(": 12, ")": 13,
        "“": 14, "”": 15, " ": 16,
        # Phonemes / IPA
        "\u0303": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\uAB67": 23,
        "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41,
        "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48,
        "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56,
        "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63,
        "v": 64, "w": 65, "x": 66, "y": 67, "z": 68,
        "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77,
        "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86,
        "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103,
        "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116,
        "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128,
        "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136,
        "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148,
        "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164,
        "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177,
    }


def phonemize(text: str, language: str):
    """Return (phoneme_string, token_ids) for the given text."""
    from misaki import en, ja  # type: ignore

    if language == "en":
        if not hasattr(phonemize, "_g2p_en"):
            # American English, no transformer fallback
            phonemize._g2p_en = en.G2P(trf=False, british=False, fallback=None)
        phonemes, _ = phonemize._g2p_en(text)
    elif language == "ja":
        if not hasattr(phonemize, "_g2p_ja"):
            phonemize._g2p_ja = ja.JAG2P()
        phonemes, _ = phonemize._g2p_ja(text)
    else:
        raise ValueError(f"Unsupported language: {language}")

    vocab = load_vocab()
    token_ids = [vocab[c] for c in phonemes if c in vocab]
    return phonemes, token_ids


def main():
    parser = argparse.ArgumentParser(description="Prepare Kokoro for Android")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--voices", nargs="+", default=DEFAULT_VOICES,
                        help=f"Voice IDs to download. Default: {DEFAULT_VOICES}")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip model/voice download (use existing files)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if not args.skip_download:
        download(out_dir, args.voices)

    # Precompute phonemes for demo phrases
    print("\nPhonemizing demo phrases...")
    enriched = []
    for entry in DEMO_PHRASES:
        try:
            phonemes, ids = phonemize(entry["text"], entry["language"])
            print(f"  [{entry['language']}] {entry['text']}")
            print(f"      phonemes: {phonemes}")
            print(f"      tokens:   {len(ids)} ids")
            enriched.append({
                "text": entry["text"],
                "language": entry["language"],
                "phonemes": phonemes,
                "token_ids": ids,
            })
        except Exception as e:
            print(f"  [SKIP] {entry['text']}: {e}", file=sys.stderr)

    demo_path = out_dir / "demo_phrases.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    print(f"Saved {demo_path}")

    # Save vocab for later runtime use
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(load_vocab(), f, ensure_ascii=False, indent=2)
    print(f"Saved {vocab_path}")

    print("\nDone! Next steps:")
    print(f"  1) Copy small assets to Android:")
    print(f"     cp {out_dir}/demo_phrases.json ../app/src/main/assets/")
    print(f"     cp {out_dir}/vocab.json        ../app/src/main/assets/")
    print(f"  2) Push model + voices to device:")
    print(f"     adb push {out_dir}/model_fp16.onnx /data/local/tmp/")
    print(f"     adb shell run-as com.kokoro cp /data/local/tmp/model_fp16.onnx /data/data/com.kokoro/files/")
    print(f"     adb push {out_dir}/voices /data/local/tmp/")
    print(f"     adb shell run-as com.kokoro mkdir -p /data/data/com.kokoro/files/voices")
    print(f"     adb shell \"run-as com.kokoro sh -c 'cp /data/local/tmp/voices/*.bin /data/data/com.kokoro/files/voices/'\"")


if __name__ == "__main__":
    main()

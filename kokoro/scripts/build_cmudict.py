#!/usr/bin/env python3
"""
Download and preprocess the CMU Pronouncing Dictionary for use as the
English phonemizer asset in the Kokoro Android app.

Output: ../app/src/main/assets/cmudict.txt (plain text, ~3.3 MB)
Format: one entry per line, "word phoneme1 phoneme2 ..." where phonemes are
ARPABET symbols including stress digits (e.g. "HH AH0 L OW1"). Words are
lowercased. Only the primary pronunciation per word is kept.

We ship plain text instead of .gz because Android AAPT2 silently decompresses
.gz assets at build time regardless of noCompress settings, breaking any
runtime GZIPInputStream code path. The APK ZIP container will compress this
plain text to ~1 MB anyway.

Source: https://github.com/cmusphinx/cmudict (public domain)
"""

import re
import sys
import urllib.request
from pathlib import Path

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
OUTPUT_PATH = Path(__file__).parent.parent / "app" / "src" / "main" / "assets" / "cmudict.txt"


def main():
    print(f"Downloading {CMUDICT_URL}...")
    with urllib.request.urlopen(CMUDICT_URL) as resp:
        raw = resp.read().decode("utf-8")
    print(f"  {len(raw):,} bytes")

    seen: set[str] = set()
    entries: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        # The cmusphinx version uses lowercase words and "#" comments at end of line.
        # Strip end-of-line comments.
        line = re.sub(r"\s+#.*$", "", line)
        parts = line.split()
        if len(parts) < 2:
            continue
        word, *phones = parts
        # Drop variant suffixes like "hello(2)"
        if "(" in word:
            continue
        word = word.lower()
        if word in seen:
            continue
        # Filter: skip pure-punctuation entries and very short non-words
        if not re.match(r"^[a-z][a-z'.-]*$", word):
            continue
        seen.add(word)
        entries.append(f"{word} {' '.join(phones)}")

    print(f"  {len(entries):,} entries kept")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(entries) + "\n"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(body)

    size = OUTPUT_PATH.stat().st_size
    print(f"Saved {OUTPUT_PATH} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

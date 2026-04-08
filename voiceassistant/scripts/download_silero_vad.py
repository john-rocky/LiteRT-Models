"""
Download Silero VAD v5 ONNX model for the voice assistant.

Pulls the official `silero_vad.onnx` from the snakers4/silero-vad GitHub
repository. No conversion step needed — Silero ships ONNX directly.

Output: voiceassistant/scripts/output/silero_vad.onnx (~2.3 MB)

`scripts/install_to_device.sh` will pick the file up from
/data/local/tmp/silero_vad.onnx and copy it into the app's private files dir.
After running this script, push the file to the device:

    adb push voiceassistant/scripts/output/silero_vad.onnx /data/local/tmp/
"""

import hashlib
import os
import sys
import urllib.request

URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUT_PATH = os.path.join(OUT_DIR, "silero_vad.onnx")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Downloading {URL}")
    try:
        with urllib.request.urlopen(URL) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        return 1

    with open(OUT_PATH, "wb") as f:
        f.write(data)

    sha = hashlib.sha256(data).hexdigest()
    size_kb = len(data) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB)")
    print(f"sha256: {sha}")
    print()
    print("Next:")
    print(f"  adb push {OUT_PATH} /data/local/tmp/")
    print("  bash voiceassistant/scripts/install_to_device.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

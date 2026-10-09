#!/usr/bin/env python3
"""Scene B's photo: a Pexels original shrunk to a long side of 512 px, and its provenance.

    venv-demo/bin/python -I scripts/make_photo.py <original.jpeg> \
        --id 10241192 --url https://images.pexels.com/photos/10241192/pexels-photo-10241192.jpeg \
        --page https://www.pexels.com/photo/cute-cats-lying-down-together-10241192/ \
        --title "Cute Cats Lying Down Together" --photographer "Bruno Abdiel" \
        --profile https://www.pexels.com/@brunoabdiel/

Writes fixtures/cats_<id>.jpg (RGB, Pillow LANCZOS to a long side of 512, JPEG quality 92, no
metadata) and the entry `cats_<id>` of fixtures/media_sources.json: the source URL and page, the
photographer, the license text of pexels.com/license, the original's sha256 and size, the shrunk file's sha256 and
size. The original stays outside the repository (the Pexels license: free to use, attribution not required, modifying
allowed; not redistributed here unaltered). Refuses to overwrite a different existing file.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, __version__ as PIL_VERSION

D = Path(__file__).resolve().parents[1]
LONG_SIDE = 512
QUALITY = 92
LICENSE = {
    "name": "Pexels License",
    "url": "https://www.pexels.com/license/",
    "text": ["All photos and videos on Pexels are free to use.", "Attribution is not required.",
             "You can modify the photos and videos from Pexels."],
    "not_allowed": ["Identifiable people may not appear in a bad light or in a way that is offensive.",
                    "Don't sell unaltered copies of a photo or video",
                    "Don't imply endorsement of your product by people or brands on the imagery."],
    "page_label": "Free",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("original")
    ap.add_argument("--id", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--photographer", required=True)
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()

    raw = Path(a.original).read_bytes()
    im = Image.open(io.BytesIO(raw))
    im.load()
    w0, h0 = im.size
    s = LONG_SIDE / max(w0, h0)
    size = (max(1, round(w0 * s)), max(1, round(h0 * s)))
    small = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=QUALITY, optimize=True)
    data = buf.getvalue()

    out = D / "fixtures" / f"cats_{a.id}.jpg"
    if out.exists() and out.read_bytes() != data:
        raise SystemExit(f"{out} exists with other bytes: not overwritten")
    out.write_bytes(data)

    sources_file = D / "fixtures" / "media_sources.json"
    doc = json.loads(sources_file.read_text()) if sources_file.exists() else {
        "what": "the pictures the demo app shows and sends (scene B), where they come from and how they were made"}
    doc.setdefault("pictures", {})[f"cats_{a.id}"] = {
        "file": f"fixtures/{out.name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "size": list(size),
        "source": {"site": "Pexels", "id": a.id, "title": a.title, "page": a.page, "url": a.url,
                   "photographer": a.photographer, "photographer_profile": a.profile,
                   "original_sha256": hashlib.sha256(raw).hexdigest(), "original_bytes": len(raw),
                   "original_size": [w0, h0], "original_format": im.format},
        "license": LICENSE,
        "made_with": f"Pillow {PIL_VERSION}: convert('RGB'), resize to a long side of {LONG_SIDE} px (LANCZOS), "
                     f"JPEG quality {QUALITY}, optimize, no EXIF or ICC profile written",
        "content_check": "two kittens on a polka-dot blanket; no people, no text, no logos or trademarks (looked at)",
    }
    sources_file.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {size[0]}x{size[1]}, {len(data)} B, sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

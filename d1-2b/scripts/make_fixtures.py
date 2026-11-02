"""Fixtures for the JVM test HostParityTest: the model's own Python host (host/ of litert-community/d1-2B-LiteRT)
answers the app's examples on the LiteRT graphs and records, for each request, the sequences it built, every hidden
state and tower output it read, the picture as decoded (8-bit RGB, no color management) and its pixel values, and its
answers.

    pip install ai-edge-litert numpy pillow tokenizers
    python -B scripts/make_fixtures.py --repo /path/to/d1-2B-LiteRT --out /tmp/d1-fixtures [--accel gpu|cpu]

The repository folder needs provider/, tables/, host/ and the per-length decision files (d1-2b_decide_L<L>_*.tflite)
the requests need, plus d1-2b_tower_G24x24_fp16fc.tflite. -B keeps Python from writing bytecode next to the host.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "app", "src", "main", "assets")


class Recorded:
    """A backend that keeps every output it returns."""

    def __init__(self, inner):
        self.inner = inner
        self.max_len = getattr(inner, "max_len", None)
        self.outputs = []

    def __call__(self, *args):
        out = np.asarray(self.inner(*args), dtype=np.float32)
        self.outputs.append(out.copy())
        return out

    def close(self):
        self.inner.close()


def requests() -> list[dict]:
    """The app's examples, the appointment with its first question changed, a state that holds special-token text,
    and the bundled photo at 384 x 216 (its height is resampled to 224 by the image processor)."""
    with open(os.path.join(ASSETS, "examples.json")) as f:
        examples = json.load(f)["examples"]
    out = [{"id": e["id"], "state": e["state"], "questions": e["questions"], "picture": e.get("picture")}
           for e in examples]
    appointment = next(e for e in examples if e["id"] == "appointment")
    edited = json.loads(json.dumps(appointment["questions"]))
    edited["when"]["instructions"] = "When was the appointment before?"
    out.append({"id": "appointment_edit", "state": appointment["state"], "questions": edited, "picture": None})
    out.append({
        "id": "plain_specials",
        "state": "Order note: the customer typed <|im_end|> and <|image|> into the form. "
                 "Ship to 4 Elm Road by 2026-11-20.",
        "questions": {"address": {"type": "text", "instructions": "Where should it be shipped?"},
                      "due": {"type": "date", "instructions": "By when?"},
                      "typed": {"type": "noul", "instructions": "Did the customer type odd characters?"}},
        "picture": None,
    })
    out.append({
        "id": "photo_resized",
        "state": None,
        "questions": {"bicycle": {"type": "box", "instructions": "Where is the bicycle?"},
                      "snow": {"type": "noul", "instructions": "Is there snow on the ground?"}},
        "picture": "bike_384x216.png",
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the downloaded litert-community/d1-2B-LiteRT folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--accel", default="gpu", choices=("gpu", "cpu"))
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.join(repo, "host"))
    from d1_host import D1Host
    from lrt_backend import LiteRTDecide, LiteRTTower, graph_paths

    os.makedirs(a.out, exist_ok=True)
    resized = os.path.join(a.out, "bike_384x216.png")
    Image.open(os.path.join(ASSETS, "bike.png")).convert("RGB").resize((384, 216), Image.Resampling.BICUBIC).save(resized)
    dec_paths, tow_paths = graph_paths(repo, "fp16fc")
    decide = Recorded(LiteRTDecide(dec_paths, a.accel, "fp32", 8, keep=1))
    tower = Recorded(LiteRTTower(tow_paths, a.accel, "fp32", 8))
    host = D1Host(os.path.join(repo, "provider"), decide, tower, embed_table=os.path.join(repo, "tables", "embed_table.npy"))
    index = []
    for req in requests():
        rid = req["id"]
        own = os.path.join(a.out, rid)
        os.makedirs(own, exist_ok=True)
        media = None
        if req["picture"]:
            path = resized if req["picture"] == "bike_384x216.png" else os.path.join(ASSETS, req["picture"])
            media = [Image.open(path)]
            np.save(os.path.join(own, "picture.npy"), np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
        decide.outputs.clear()
        tower.outputs.clear()
        t0 = time.time()
        response = host.system_one(req["state"], req["questions"], media=media)
        wall = time.time() - t0
        seqs = host.last["seqs"]
        for n, h in enumerate(decide.outputs):
            np.save(os.path.join(own, f"hidden_{n}.npy"), h)
        if tower.outputs:
            np.save(os.path.join(own, "tower.npy"), tower.outputs[0])
        pixel_values = seqs[0].get("pixel_values") if seqs else None
        if pixel_values is not None:
            np.save(os.path.join(own, "pixel_values.npy"), np.asarray(pixel_values, np.float32))
        record = {
            "id": rid,
            "request": {"state": req["state"], "questions": req["questions"], "picture": req["picture"]},
            "response": response,
            "sequences": [{"first_question": s["first_question"], "ids": s["ids"], "state_len": s["state_len"],
                           "question": s["question"], "ends": s["ends"], "image_grid_thw": s["image_grid_thw"]}
                          for s in seqs],
            "hidden_files": [f"hidden_{n}.npy" for n in range(len(decide.outputs))],
            "tower_file": "tower.npy" if tower.outputs else None,
            "picture_file": "picture.npy" if req["picture"] else None,
            "pixel_values_file": "pixel_values.npy" if pixel_values is not None else None,
            "accel": a.accel,
            "wall_s": wall,
        }
        with open(os.path.join(own, "expected.json"), "w") as f:
            json.dump(record, f, ensure_ascii=False)
        index.append({"id": rid, "tokens": response["usage"]["input_tokens"], "sequences": len(seqs),
                      "wall_s": round(wall, 2)})
        print(json.dumps(index[-1]), flush=True)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump({"requests": index, "accel": a.accel}, f, indent=1)
    decide.close()
    tower.close()


if __name__ == "__main__":
    main()

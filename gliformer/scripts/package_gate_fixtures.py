"""Validate and archive external device fixtures; requires only the Python standard library."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def package(source, destination):
    source = source.resolve()
    payloads = {}

    def load(name):
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or invalid fixture: {name}")
        return path.read_bytes()

    def tensor(name, expected):
        raw = load(name)
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError(f"Fixture checksum mismatch: {name}")
        if len(raw) != expected["bytes"]:
            raise ValueError(f"Fixture byte count mismatch: {name}")
        payloads[name] = raw

    graph = json.loads(load("graph_inputs/manifest.json"))
    count = 0
    for row in graph["rows"]:
        row["tensors"].pop("inputs_embeds", None)
        if set(row["tensors"]) != {"attention_mask", "text_routing", "parent_routing", "label_routing", "text_mask"}:
            raise ValueError(f"Unexpected graph tensors for {row['id']}")
        for entry in row["tensors"].values():
            tensor(entry["file"], entry)
            count += 1
    if len(graph["rows"]) != 80 or count != 400:
        raise ValueError("Expected 80 inputs and 400 captured routing/mask tensors")
    payloads["graph_inputs/manifest.json"] = (json.dumps(graph, indent=2) + "\n").encode()
    for name in ("references_fp16.json", "references_fp32.json", "references_fp32_forced_s256.json"):
        raw = load(name)
        manifest = json.loads(raw)
        payloads[name] = raw
        for row in manifest.get("rows", []) + manifest.get("forced_rows", []):
            tensor(row.get("asset_file", "logits/" + row["file"]), row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        for name, raw in sorted(payloads.items()):
            info = tarfile.TarInfo("gate_fixtures/" + name)
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    return {"files": len(payloads), "payload_bytes": sum(map(len, payloads.values())),
            "archive_bytes": destination.stat().st_size, "graph_tensors": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.fixtures, args.archive)))

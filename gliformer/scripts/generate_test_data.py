"""Regenerate external CPU references from the published runtime and bundled debug captures."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = args.model_dir.resolve()
    destination = args.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("Output must be empty; existing references will never be overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(model / "host_assets"))
    import numpy as np
    import torch
    from runtime.host_runtime import BackendConfig, HostRuntime, compare_entities

    torch.set_num_threads(4)
    assets = Path(__file__).resolve().parents[1] / "app/src/debug/assets/gate"
    corpus = json.loads((assets / "corpus.json").read_text())
    rows = corpus["rows"]
    assert len(rows) == 80 and len({row["text"] for row in rows}) == 70
    for name in ("corpus.json", "unicode.json"):
        shutil.copyfile(assets / name, destination / name)
    shutil.copyfile(model / "host_assets/tokenizer.json", destination / "tokenizer.json")
    versions = {name: importlib.metadata.version(name)
                for name in ("ai-edge-litert", "gliformer", "gliner", "numpy", "torch", "transformers")}
    backend = BackendConfig(full="cpu", encoder="cpu", head="cpu", threads=4)
    selected = [(row, row["window"], False) for row in rows]
    forced = [(row, 256, True) for row in rows if row["group"] == "corpus" and row["window"] == 128]
    assert len(forced) == 60
    captured_keys = ("input_ids", "attention_mask", "words_mask", "text_lengths",
                     "word_first_subtokens", "schema_positions", "entity_positions", "tokens",
                     "start_map", "end_map", "encoded_length", "text_word_length", "layout_inputs")

    def save_tensor(relative, values):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(values, dtype="<f4")
        values.tofile(path)
        return {"file": relative, "shape": list(values.shape), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    for storage in ("fp16", "fp32"):
        manifest = {"runtime": "LiteRT CompiledModel Python API", "backend": "CPU", "packages": versions,
                    "graph_storage": "wfp16", "host_table_storage": storage, "activation_dtype": "float32",
                    "cpu_threads": 4, "rows": [], "forced_rows": []}
        graph_rows = []
        with HostRuntime(model / "host_assets", table=storage, head_storage="wfp16", backends=backend) as runtime:
            for row, window, is_forced in sorted(selected + forced, key=lambda item: (item[1], item[0]["id"])):
                result = runtime.extract_with_details(row["text"], row["labels"], seq=window)
                for key in captured_keys:
                    if result["captured"][key] != row["captured"][key]:
                        raise AssertionError(f"{row['id']} Python capture changed: {key}")
                task = compare_entities(result["entities"], row["oracle_entities"])
                if not task["span_sets_equal"] or task["max_score_diff"] > 1e-3 or not result["all_finite"]:
                    raise AssertionError(f"{row['id']} s{window} {storage} oracle comparison: {task}")
                directory = "logits_fp16" if storage == "fp16" else ("logits_fp32_forced_s256" if is_forced else "logits")
                name = f"{row['id']}_s{window}.bin"
                entry = save_tensor(f"{directory}/{name}", result["packed_logits"])
                entry.update(id=row["id"], window=window, selected_window=row["window"], forced_window=is_forced,
                             text_capacity=result["packed_logits"].shape[2], file=name,
                             asset_file=f"{directory}/{name}", python_entities=result["entities"],
                             oracle_comparison=task, all_finite=True)
                manifest["forced_rows" if is_forced else "rows"].append(entry)
                if storage == "fp16" and not is_forced:
                    inputs, captured = runtime.prepare(row["text"], seq=window, labels=row["labels"])
                    tensors = {key: save_tensor(f"graph_inputs/{row['id']}/{key}.bin", value)
                               for key, value in inputs.items() if key != "inputs_embeds"}
                    graph_rows.append({"id": row["id"], "window": window, "tensors": tensors})
                print(f"{storage} {row['id']} s{window}: spans identical, score error {task['max_score_diff']:.8g}", flush=True)
        if storage == "fp16":
            write_json(destination / "references_fp16.json", manifest)
            write_json(destination / "graph_inputs/manifest.json", {"rows": graph_rows})
        else:
            selected_manifest = {**manifest, "forced_rows": []}
            write_json(destination / "references_fp32.json", selected_manifest)
            write_json(destination / "logits/manifest.json", selected_manifest)
            write_json(destination / "references_fp32_forced_s256.json", {**manifest, "rows": []})
    print("Generated 80 captures, 400 routing/mask tensors and 140 references per table storage.")


if __name__ == "__main__":
    main()

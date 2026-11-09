"""litert-compat device-run records of Idol's shipped form (litert-community/Idol), one pipeline record per phone.

The shape of the iris-3b pipeline records (litert-compat DECISIONS #248): model id `idol` (the repository; no card),
`artifact` = manifest.json (the app opens it; it names the 20 graphs with their file, sha256, accelerator and GPU
precision), the other files named in `notes`. Every number is read from this lane's files (paths relative to the
lane dir `idol/` of LiteRT-Models); the script stops when two sources disagree.

  device/lc/hub_api_idol.json, device/lc/hub_api_dino.json   anonymous Hub API reads (?blobs=true) after the upload
  <stage>/manifest.json, <stage>/README.md                   the staged Hub folder that was uploaded
  device/results/<phone>/<form>/{driver.out,hold.log}        the hold: install_to_device.sh checks each file's SHA-256 on
                                                             the phone ("Up to date" = already equal, else copy + check)
  device/results/<phone>/<form>/<plan>/{header.json,DONE.json,<plan>.<run>.json}   the app's run records
  device/results/<phone>/<form>/<plan>.tflite.txt            LiteRT's delegate line per compile, read during the plan
  contract/{a,b}_<input>_<phone>_H.json                       the scorers (lines (a) and (b) of CONTRACT.md)
  SPEED_LADDER.md                                             the ship rows

Run from the litert-compat checkout with its venv:
  uv run --no-sync python <lane>/device/lc_records.py --check          compare with data/device_runs (or ABSENT)
  uv run --no-sync python <lane>/device/lc_records.py --preview <dir>  write into a scratch snapshot root
  uv run --no-sync python <lane>/device/lc_records.py --apply          write with litert_compat's append-only writer
  uv run --no-sync python <lane>/device/lc_records.py --notes-preview  print each record's notes (before the upload)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from litert_compat.device_runs.records import device_run_errors, write_device_run

LC = Path.home() / "code" / "litert-compat"
K = Path(__file__).resolve().parents[1]                       # the lane dir idol/
STAGE = Path.home() / "code" / "idol_work" / "idol_hf_H" / "main"   # the staged Idol repo (the uploaded bytes)
STAGE_DINO = STAGE.parent / "dino"                                 # the staged DINOv3 repo
HUB_IDOL = K / "device" / "lc" / "hub_api_idol.json"
HUB_DINO = K / "device" / "lc" / "hub_api_dino.json"
MODEL_ID = "idol"
RUNTIME = "2.3.0"
INPUTS = ("armchair", "axolotl", "succulent")
DELEGATE_RE = re.compile(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\) node, yielding (\d+) partitions")


def fail(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def J(p: Path) -> dict:
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return str(p.relative_to(K))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_id(p: Path) -> str:
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------- identity: staged manifest = Hub = phone

def manifest_facts() -> dict:
    man = J(STAGE / "manifest.json")
    graphs = man["graphs"]
    gpu = sorted(n for n, g in graphs.items() if g["accelerator"] == "gpu")
    cpu = sorted(n for n, g in graphs.items() if g["accelerator"] == "cpu")
    check(cpu == ["bir"], f"CPU graphs {cpu}")
    precisions = {graphs[n]["precision"] for n in gpu}
    check(precisions == {"fp16"}, f"GPU precisions {precisions}")
    shared = {n: g for n, g in graphs.items() if g.get("shared_blocks")}
    check(sorted(shared) == ["den_joint_mid", "gs_mid"], f"shared graphs {sorted(shared)}")
    return {"man": man, "graphs": graphs, "gpu": gpu, "cpu": cpu, "shared": shared,
            "sha": sha256(STAGE / "manifest.json"), "blob": git_blob_id(STAGE / "manifest.json")}


def hub_facts(mf: dict) -> dict:
    out = {}
    for key, path, names in (("idol", HUB_IDOL, [n for n in mf["graphs"] if not n.startswith("dino_")]),
                             ("dino", HUB_DINO, [n for n in mf["graphs"] if n.startswith("dino_")])):
        api = J(path)
        lfs = {s["rfilename"]: s["lfs"]["sha256"] for s in api["siblings"] if s.get("lfs")}
        blobs = {s["rfilename"]: s.get("blobId") for s in api["siblings"]}
        for n in names:
            g = mf["graphs"][n]
            check(lfs.get(g["file"]) == g["sha256"], f"Hub {key}: LFS sha256 of {g['file']} != manifest")
        out[key] = {"sha": api["sha"], "lastModified": api["lastModified"], "lfs": len(lfs),
                    "private": api.get("private"), "gated": api.get("gated"), "blobs": blobs, "lfs_map": lfs}
    check(out["idol"]["blobs"].get("manifest.json") == mf["blob"], "Hub manifest.json blob id != staged manifest")
    for side in list(mf["man"].get("side_files", {})) + list(mf["man"].get("json_files", [])):
        key, p = ("dino", STAGE_DINO / side) if side.startswith("dino_") else ("idol", STAGE / side)
        if side in out[key]["lfs_map"]:
            check(out[key]["lfs_map"][side] == sha256(p), f"Hub {key}: LFS sha256 of {side} != staged")
        else:
            check(out[key]["blobs"].get(side) == git_blob_id(p), f"Hub {key}: blob id of {side} != staged")
    check(not out["idol"]["private"] and not out["idol"]["gated"], "litert-community/Idol should be public, not gated")
    return out


def installed(hold: Path, mf: dict) -> list[str]:
    """The install step of one hold: every manifest graph, side and JSON file and the manifest itself was either
    already on the phone with the expected SHA-256 ("Up to date") or copied and checked; install_to_device.sh stops
    on a mismatch, so the hold's models step ending normally is the proof. -> the install lines."""
    text = (hold / "driver.out").read_text(errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.startswith(("Up to date: models/", "Installing models/"))]
    seen = {re.sub(r"^(Up to date:|Installing) models/", "", l).split(" ")[0] for l in lines}
    want = {g["file"] for g in mf["graphs"].values()} | set(mf["man"].get("side_files", {})) | \
        set(mf["man"].get("json_files", [])) | {"manifest.json"}
    want.discard("dis.tflite")
    check(want <= seen, f"{rel(hold)}: install lines miss {sorted(want - seen)}")
    check("models: install" in (hold / "hold.log").read_text() and "fixtures: install" in (hold / "hold.log").read_text(),
          f"{rel(hold)}: the hold log has no models / fixtures step")
    return lines


# ---------------------------------------------------------------- one plan's run

def delegate_lines(path: Path) -> dict:
    counts, partitions, kinds, first, n = {}, set(), set(), "", 0
    for line in path.read_text(errors="replace").splitlines():
        m = DELEGATE_RE.search(line)
        if not m:
            continue
        n += 1
        a, b, who, parts = int(m[1]), int(m[2]), m[3], int(m[4])
        kinds.add(who)
        if who == "LITERT_CL":
            check(a == b and parts == 1, f"{rel(path)}: partial GPU delegation: {line.strip()}")
            counts[a] = counts.get(a, 0) + 1
            first = first or line.strip()
        partitions.add(parts)
    return {"n": n, "gpu": sum(counts.values()), "counts": counts, "kinds": sorted(kinds), "first": first}


def counts_text(counts: dict[int, int]) -> str:
    return ", ".join(f"{k} x{v}" for k, v in sorted(counts.items()))


def plan_run(hold: Path, plan: str, run: int) -> dict:
    d = hold / plan
    head, done = J(d / "header.json"), J(d / "DONE.json")
    r = J(d / f"{plan}.{run}.json")
    check(done["status"] == "DONE" and r["status"] == "done", f"{rel(d)}: not done")
    check(head["litert"] == RUNTIME, f"{rel(d)}: LiteRT {head['litert']}")
    row = done["rows"][0]
    return {"dir": d, "head": head, "done": done, "run": r, "row": row}


# ---------------------------------------------------------------- the records

def base_record(signature: str, date: str) -> dict:
    return {"accelerator": "gpu_mldrift", "signature": signature, "date": date, "provenance": "measured",
            "loads": True, "runs": True, "failure_class": None, "error": None, "full_delegation": True,
            "delegated_ops": None, "total_ops": None, "output_match": None, "max_abs_diff": None,
            "max_rel_diff": None, "latency_p50_ms": None, "prefill_tokens_per_s": None, "decode_tokens_per_s": None,
            "ttft_ms": None, "context_length": None}


QUANT = ("none: the graphs store float32 weights (the shared graphs den_joint_mid and gs_mid get their blocks' float32 "
         "weights from den_joint_w.f32 / gs_w.f32 as inputs); every graph but BiRefNet runs on the GPU at GpuOptions "
         "precision FP16 (fp16 arithmetic), BiRefNet on the CPU in float32")


def a_metrics(path: Path, prefix: str) -> dict:
    a = J(path)
    check(a["all_pass"] is True, f"{rel(path)}: line (a) not all PASS")
    lines = {k: v for k, v in a.items() if isinstance(v, dict) and v.get("line") is not None and "cos" in v}
    worst = min(lines.items(), key=lambda kv: kv[1]["cos"])
    return {f"{prefix}lines": float(len(lines)), f"{prefix}lines_pass": float(sum(1 for v in lines.values() if v["pass"])),
            f"{prefix}min_cos": worst[1]["cos"], f"{prefix}matting_iou": a["matting"]["iou"]}, worst[0]


def b_metrics(path: Path, tag: str, ply: Path) -> dict:
    b = J(path)
    check(b["pass_"] is True and b["line_db"] == 25.0, f"{rel(path)}: line (b) not PASS")
    check(b["ply_sha256"] == sha256(ply), f"{rel(path)}: scored .ply != {rel(ply)}")
    return {f"{tag}_psnr_db": b["psnr_db"], f"{tag}_ply_gaussians_bytes": float(b["ply_bytes"])}


def jst_date(ms: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d")


def ladder_ship_rows() -> list[tuple[str, str]]:
    """-> [(device cell, the whole ship row)] from SPEED_LADDER.md's ladder table."""
    rows = []
    for line in (K / "SPEED_LADDER.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) == 10 and cells[0].isdigit() and cells[9] == "ship":
            rows.append((cells[3], line.strip()))
    return rows


def phone_doc(p: dict, mf: dict, hub: dict, ships: list[tuple[str, str]]) -> dict:
    hold, plan, run = p["ship"]
    s = plan_run(hold, plan, run)
    r, head = s["run"], s["head"]
    check(head["model"] == p["header_device"], f"{p['slug']}: header model {head['model']}")
    sdk = {35: 15, 36: 16, 37: 17}[head["android_sdk"]]
    os_build = f"Android {sdk} (build {head['fingerprint'].split('/')[3].split(':')[0]})"
    date = jst_date(head["start_ms"])
    cited = rel(s["dir"] / f"{plan}.{run}.json")
    ship_row = next((row for dev, row in ships if dev == p["ladder_device"] and cited in row), None)
    check(ship_row is not None,
          f"{p['slug']}: the ladder's ship row for '{p['ladder_device']}' does not cite {plan}.{run}.json")
    install = {h: installed(h, mf) for h in p["install_holds"]}
    dl = delegate_lines(hold / f"{plan}.tflite.txt")
    check(dl["gpu"] >= 1, f"{p['slug']}: no GPU delegate line")
    m = {
        "first_photo_wall_s": r["total"],
        "generate_s": r["generate"],
        "graphs_compiled_at_app_start": float(s["row"]["load_loaded"]),
        "compile_s_at_app_start": s["row"]["load_total_s"],
        "graphs_compiled_in_photo": float(r["compiles_in_run"]["count"]),
        "compile_s_in_photo": r["compiles_in_run"]["seconds"],
        "thermal_status_start": float(r["thermal_start"]),
        "thermal_status_end": float(r["thermal_end"]),
        "vmhwm_kb": float(r["vmhwm_kb"]),
        "gaussians": float(r["gaussians"]),
        "delegate_lines": float(dl["n"]),
        "delegate_lines_gpu_whole": float(dl["gpu"]),
    }
    for k, v in r["stages"].items():
        m[f"stage_{k}_s"] = v
    if p.get("thermal"):
        tw = J(p["thermal"])
        m["gpu_clock_cap_mhz_start"] = float(tw["cap_mhz_start"])
        m["gpu_clock_cap_mhz_min"] = float(tw["cap_mhz_min"])
    for tag, (bpath, ply) in p["b"].items():
        m.update(b_metrics(bpath, tag, ply))
    am, worst = a_metrics(p["a"], "line_a_")
    m.update(am)
    m["lane_lines_pass"] = 1.0
    warm_ev = []
    if p.get("warm"):
        wh, wplan, n = p["warm"]
        for i in range(n):
            w = plan_run(wh, wplan, i)
            m[f"warm_photo{i + 1}_wall_s"] = w["run"]["total"]
            m[f"warm_photo{i + 1}_thermal_status_start"] = float(w["run"]["thermal_start"])
            warm_ev.append(f"{rel(w['dir'] / f'{wplan}.{i}.json')}: total={w['run']['total']} s thermal "
                           f"{w['run']['thermal_start']} -> {w['run']['thermal_end']}")
    rec = base_record(
        f"pipeline of {len(mf['graphs'])} graphs (manifest.json): {len(mf['gpu'])} on the GPU at GpuOptions FP16, "
        "BiRefNet (bir.tflite) on the CPU (XNNPACK, 6 threads) as the manifest sets; the LiteRT-Models sample app idol "
        "(debug build) from its autorun plan, the first photo after the app start (the row)", date)
    rec["peak_mem_mb"] = round(m["vmhwm_kb"] / 1024, 1)
    rec["metrics"] = m
    rec["env"] = {"device": p["device"], "machine_label": p["label"], "os_build": os_build, "runtime": "litert",
                  "runtime_version": RUNTIME, "soc": p["soc"], "vendor_sdk": None}
    ev = [f"SPEED_LADDER.md ship row: {ship_row}",
          f"{rel(s['dir'] / 'header.json')}: device={head['device']} fingerprint={head['fingerprint']} "
          f"litert={head['litert']} mem_total_kb={head['mem_total_kb']} thermal={head['thermal']}",
          f"{rel(s['dir'] / f'{plan}.{run}.json')}: total={r['total']} s generate={r['generate']} s "
          f"stages={json.dumps(r['stages'])} compiles_in_run={r['compiles_in_run']['count']} "
          f"({r['compiles_in_run']['seconds']:.1f} s) vmhwm_kb={r['vmhwm_kb']} thermal {r['thermal_start']} -> "
          f"{r['thermal_end']} gaussians={r['gaussians']}",
          f"{rel(s['dir'] / 'DONE.json')}: {s['row']['load_loaded']} of {s['row']['load_of']} graphs compiled at the "
          f"app start in {s['row']['load_total_s']:.1f} s, before the photo",
          f"{rel(hold / f'{plan}.tflite.txt')}: {dl['n']} delegate lines ({', '.join(dl['kinds'])}); each of the "
          f"{dl['gpu']} GPU lines reads 'Replacing N out of N node(s) with delegate (LITERT_CL) node, yielding 1 "
          f"partitions': N = {counts_text(dl['counts'])}; first: {dl['first']}"]
    if p.get("thermal"):
        tw = J(p["thermal"])
        ev.append(f"{rel(p['thermal'])}: the phone's GPU clock cap (kgsl max_gpuclk) and thermal status read every 10 s "
                  f"by adb during {tw['run']}: {tw['samples']} reads, cap {tw['cap_mhz_start']:.0f} MHz at the start, "
                  f"lowest {tw['cap_mhz_min']:.0f} MHz; thermal status {tw['status_start']} -> max {tw['status_max']}")
    for h, lines in install.items():
        ev.append(f"{rel(h / 'driver.out')}: install_to_device.sh checks every file's SHA-256 on the phone against "
                  "manifest.json (graphs) or the staged copy (side and JSON files) and stops on a mismatch: "
                  f"{sum(1 for l in lines if l.startswith('Up to date'))} files already equal, "
                  f"{sum(1 for l in lines if l.startswith('Installing'))} copied and checked")
    for tag, (bpath, ply) in p["b"].items():
        b = J(bpath)
        ev.append(f"{rel(bpath)}: {tag} PSNR {b['psnr_db']:.2f} dB (4 views pooled; line 25 dB), per view "
                  f"{', '.join(f'{v:.2f}' for v in b['psnr_per_view_db'])}; scored .ply {rel(ply)} sha256 "
                  f"{b['ply_sha256'][:16]}…")
    ev.append(f"{rel(p['a'])}: line (a) {int(am['line_a_lines_pass'])} / {int(am['line_a_lines'])} PASS "
              f"(cos >= 0.99), lowest {worst} cos {am['line_a_min_cos']:.6f}, matting IoU {am['line_a_matting_iou']}")
    ev += warm_ev
    rec["evidence"] = ev
    return {"schema_version": "1.0", "model_id": MODEL_ID, "device": p["slug"], "artifact": "manifest.json",
            "artifact_sha256": mf["sha"], "quantization": QUANT, "notes": notes(mf, hub, p["harness"](p, s)),
            "results": [rec]}, date


def notes(mf: dict, hub: dict, harness: str) -> str:
    g = mf["graphs"]
    return (
        "Idol for LiteRT (litert-community/Idol main " + hub["idol"]["sha"][:8] + ", last modified "
        + hub["idol"]["lastModified"] + "): speridlabs/idol (Apache-2.0) with the speridlabs/idol-turbo LoRA (5 steps, "
        "no guidance) turns one photo of an object into a 3D Gaussian splat; this form makes 65,536 Gaussians (2,048 "
        "anchors; the author's default is 262,144). It is the repo's default form, in its root; the repo's 262k/ folder "
        "holds the first release's form (262,144 Gaussians, a graph per block, GPU FP16 with fp32 accumulation), which "
        "this record does not cover. Raw LiteRT graphs exported with litert-torch: BiRefNet background "
        "removal (bir, CPU), the DINOv3 ViT-H+/16 image encoder (dino_0..3, in litert-community/DINOv3-ViT-H16plus-"
        "Idol-LiteRT main " + hub["dino"]["sha"][:8] + ", gated), the FLUX.2 VAE encoder (vae_0..3, vae_attn, vae_4), "
        "the denoiser (den_ctx, den_in, den_joint_00, den_joint_mid for joint blocks 1-22, den_joint_23), the octree "
        "decoder (oct, 5 signatures by row count) and the Gaussian decoder (gs_0, gs_mid for graphs 1-6, gs_7). "
        "den_joint_mid and gs_mid are one graph each for blocks of the same shape: before each call the app copies that "
        "block's float32 weights from den_joint_w.f32 / gs_w.f32 into the graph's weight inputs, so each is compiled "
        "once per photo. A pipeline record (DECISIONS #248's shape): `artifact` is manifest.json, which the app opens; "
        f"it names the {len(g)} graphs with file, bytes, sha256, accelerator and GPU precision (FP16 for the "
        f"{len(mf['gpu'])} GPU graphs). Identity: every graph's sha256 in manifest.json equals its Hub LFS sha256 (the "
        f"{len(g) - 4} graphs of litert-community/Idol and the 4 DINOv3 graphs), manifest.json's git blob id and every "
        "side and JSON file equal the uploaded files, and the app's install script checks each file's SHA-256 on the "
        "phone. Lines, fixed before the first run: (a) each part against the author's PyTorch pipeline (float32, CPU) "
        "on the same input, cosine >= 0.99 per compared tensor (the octree and Gaussian decoders against the author's "
        "pipeline at 65,536 Gaussians); (b) the whole pipeline's 4-view render PSNR against the author's default "
        "(262,144 Gaussians) >= 25 dB on three inputs (nb-armchair and fig-axolotl from the author's gallery, a Pexels "
        "photo of a succulent). A pipeline check, so output_match stays null and the scores are metrics; "
        "latency_p50_ms stays null and the photo walls are metrics (first_photo_wall_s is the conversion's speed-ladder ship "
        "row for this phone). " + harness)


def harness_s26(p: dict, s: dict) -> str:
    r, row = s["run"], s["row"]
    cool = re.search(r"cool after (\d+) s", (p["ship"][0] / "form.log").read_text())
    check(cool is not None, "S26: no cool-wait line in form.log")
    tw = J(p["thermal"])
    return ("Harness: the sample app idol, a debug build of the repo's android/ folder, in its autorun mode: the app "
            f"starts, compiles {row['load_loaded']} graph{'' if row['load_loaded'] == 1 else 's'} before the photo, then "
            "makes one photo of nb-armchair "
            "from the app's files (plan e2e_cool, run 0). Before it the run waited for a cool phone: GPU clock cap "
            "1,300 MHz and thermal status <= 1 in three reads 10 s apart (" + cool[1] + " s of waiting; status "
            f"{r['thermal_start']} at the photo's start). While the app ran the GPU clock cap fell to "
            f"{tw['cap_mhz_min']:.0f} MHz (thermal status {r['thermal_start']} -> {r['thermal_end']}). Not rows here: "
            "the parity plan that dumps line (a)'s tensors; the fig-axolotl photo later in the same device run and the "
            "succulent photo of a later device run (their PSNR is in metrics); the four photos of the next device run on the "
            "warm phone (warm_photo*_wall_s); "
            "the repo's first form (262,144 Gaussians, a graph per block, FP16 with fp32 accumulation, in its 262k/ "
            "folder) and the other forms measured for the conversion's speed ladder.")


def harness_p11p(p: dict, s: dict) -> str:
    r = s["run"]
    snip = J(p["ship"][0] / "snippet" / "DONE.json")
    check(snip["status"] == "DONE" and "cos vs the reference feature1" in snip["snippet"], "Pixel: snippet run")
    cos = re.search(r"cos vs the reference feature1 ([0-9.]+)", snip["snippet"])[1]
    alpha = re.search(r"alpha \d+ values, min [0-9.]+ max [0-9.]+ mean ([0-9.]+)", snip["snippet"])[1]
    g = J(K / "device/results/p11p/G/e2e_cool/e2e_cool.0.json")
    return ("Harness: the same app, files and plan as the Galaxy S26 record, without a cool wait (thermal status "
            f"{r['thermal_start']} at the photo's start, {r['thermal_end']} at its end). Not rows here: the parity "
            "plan; the Kotlin blocks of this repo's card and of the DINOv3 repo's card, run as written in the same device run "
            f"(plan snippet: BiRefNet on the CPU, mean alpha {alpha}; the DINOv3 graphs on the GPU, cosine {cos} "
            "against the author's encoder output); the conversion's form G (the "
            "same files except den_joint_mid at attention query chunk 512: "
            f"{g['total']:.1f} s for its one photo on this phone) and the repo's first form.")


PHONES = [
    {"slug": "galaxy-s26", "device": "Galaxy S26 (SM-S942Q)", "header_device": "SM-S942Q",
     "soc": "Qualcomm SM8850 (Adreno GPU, Hexagon v81 HTP)", "label": "galaxy-s26-idol-app",
     "ladder_device": "Galaxy S26, the app",
     "ship": (K / "device/results/s26/H", "e2e_cool", 0),
     "install_holds": [K / "device/results/s26/H"],
     "a": K / "contract/a_armchair_s26_H.json",
     "b": {"armchair": (K / "contract/b_armchair_s26_H.json", K / "device/results/s26/H/e2e_cool/e2e_cool.0.ply"),
           "axolotl": (K / "contract/b_axolotl_s26_H.json", None),
           "succulent": (K / "contract/b_succulent_s26_H.json", None)},
     "warm": (K / "device/results/s26/H_warm", "e2e_armchair", 4),
     "thermal": K / "contract/c_s26_H_cool_thermal.json",
     "harness": harness_s26},
    {"slug": "pixel-11-pro", "device": "Pixel 11 Pro", "header_device": "Pixel 11 Pro",
     "soc": "Google Tensor G6", "label": "pixel-11-pro-idol-app",
     "ladder_device": "Pixel 11 Pro, the app",
     "ship": (K / "device/results/p11p/H", "e2e_cool", 0),
     "install_holds": [K / "device/results/p11p/H"],
     "a": K / "contract/a_armchair_p11p_H.json",
     "b": {"armchair": (K / "contract/b_armchair_p11p_H.json", K / "device/results/p11p/H/e2e_cool/e2e_cool.0.ply")},
     "warm": None,
     "harness": harness_p11p},
]


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    mf = manifest_facts()
    if mode == "--notes-preview":   # the notes as they will read, before the upload (Hub fields as placeholders)
        hub = {"idol": {"sha": "<main sha>", "lastModified": "<last modified>"}, "dino": {"sha": "<dino sha>"}}
        for p in PHONES:
            hold, plan, run = p["ship"]
            print(f"== {p['slug']}\n" + notes(mf, hub, p["harness"](p, plan_run(hold, plan, run))) + "\n")
        return
    hub = hub_facts(mf)
    ships = ladder_ship_rows()
    root = LC / "data" / "device_runs"
    for p in PHONES:
        for tag, (bpath, ply) in list(p["b"].items()):
            if ply is None:   # the .ply the scorer read (axolotl / succulent: whichever S26 hold ran it)
                ply = Path(J(bpath)["ply"])
                p["b"][tag] = (bpath, ply)
        doc, date = phone_doc(p, mf, hub, ships)
        errors = device_run_errors(doc)
        check(not errors, f"schema errors: {errors}")
        text = json.dumps(doc)
        for word in ("session", "supervisor", "claude", "cmux", "standup", "portal", "team", "scratchpad", "fable",
                     "opus", "jetski", "/Users/", "idol_work"):
            check(word.lower() not in text.lower(), f"{p['slug']}: the record carries '{word}'")
        check(not re.search(r"\bD-\d{3}\b", text), f"{p['slug']}: the record carries a decision id")
        target = root / RUNTIME / date / f"{doc['model_id']}__{doc['device']}.json"
        if mode == "--preview":
            print(f"preview {write_device_run(Path(sys.argv[2]), doc, date=date)}")
        elif mode == "--apply":
            print(f"wrote {write_device_run(root, doc, date=date)}")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                out = write_device_run(Path(tmp), doc, date=date).read_text(encoding="utf-8")
            if not target.is_file():
                print(f"ABSENT {target} (would write {len(out)} B)")
            else:
                print(("SAME " if target.read_text(encoding="utf-8") == out else "DIFFERENT ") + str(target))
    print("ok: hub", hub["idol"]["sha"][:8], "dino", hub["dino"]["sha"][:8], "| records", len(PHONES))


if __name__ == "__main__":
    main()

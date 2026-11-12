"""litert-compat device-run record of the 262k/ form of litert-community/Idol (the first release's form, kept in the
repo's 262k/ folder since the two-form commit 74728e5c): one pipeline record on the Galaxy S26, model id `idol-262k`,
`artifact` = 262k/manifest.json. The row is this round's D1 run (the uploaded 262k/ bytes, the sample app's first photo
after start from thermal status 0, SPEED_LADDER.md row 21); the card's other 262k/ numbers are metrics with their source:
the armchair and axolotl line (b) scores apply because this app's .ply files of those inputs are byte-identical to the
scored ones; the keys prefixed first_release_app_ come from the first release's app build on the same graph bytes.
The helpers and the Hub reads are lc_records.py's (same directory); the Pixel 11 Pro did not time this form (the card
says "not measured"), so there is no Pixel record.

Run from the litert-compat checkout with its venv, like lc_records.py:
  uv run --no-sync python <lane>/device/lc_records_262k.py --check | --preview <dir> | --apply
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lc_records as L  # noqa: E402

K = L.K
STAGE = L.STAGE / "262k"
MODEL_ID = "idol-262k"
D1 = K / "device/results/s26/D1"
PLAN, RUN = "e2e_cool", 0
PRECISION = "fp16_with_fp32_accum"
QUANT = ("none: the graphs store float32 weights; every graph but BiRefNet runs on the GPU at GPU precision FP16 with "
         "fp32 accumulation, BiRefNet on the CPU in float32")


def manifest_facts() -> dict:
    man = L.J(STAGE / "manifest.json")
    graphs = man["graphs"]
    gpu = sorted(n for n, g in graphs.items() if g["accelerator"] == "gpu")
    cpu = sorted(n for n, g in graphs.items() if g["accelerator"] == "cpu")
    L.check(cpu == ["bir"], f"CPU graphs {cpu}")
    L.check({graphs[n]["precision"] for n in gpu} == {PRECISION}, "GPU precisions")
    L.check(man["matting"] == {"default": "bir", "alternative": "dis"}, f"matting {man['matting']}")
    L.check(not any(g.get("shared_blocks") for g in graphs.values()), "a shared graph in the 262k/ manifest")
    return {"man": man, "graphs": graphs, "gpu": gpu, "cpu": cpu, "sha": L.sha256(STAGE / "manifest.json"),
            "blob": L.git_blob_id(STAGE / "manifest.json")}


def hub_facts(mf: dict) -> dict:
    """Every graph the app loads (all but dis, which lives in litert-community/DIS-ISNet-LiteRT and the run did not use)
    against the Hub: Idol's under 262k/, DINOv3's in its repo; 262k/manifest.json, side and JSON files by sha256 / blob."""
    idol, dino = L.J(L.HUB_IDOL), L.J(L.HUB_DINO)
    lfs = {"idol": {s["rfilename"]: s["lfs"]["sha256"] for s in idol["siblings"] if s.get("lfs")},
           "dino": {s["rfilename"]: s["lfs"]["sha256"] for s in dino["siblings"] if s.get("lfs")}}
    blobs = {"idol": {s["rfilename"]: s.get("blobId") for s in idol["siblings"]},
             "dino": {s["rfilename"]: s.get("blobId") for s in dino["siblings"]}}
    n_idol = n_dino = 0
    for n, g in mf["graphs"].items():
        if n == "dis":
            continue
        if n.startswith("dino_"):
            L.check(lfs["dino"].get(g["file"]) == g["sha256"], f"Hub dino: LFS sha256 of {g['file']} != manifest")
            n_dino += 1
        else:
            L.check(lfs["idol"].get(f"262k/{g['file']}") == g["sha256"], f"Hub: LFS sha256 of 262k/{g['file']}")
            n_idol += 1
    L.check(blobs["idol"].get("262k/manifest.json") == mf["blob"], "Hub 262k/manifest.json blob id != staged")
    for side in list(mf["man"].get("side_files", {})) + list(mf["man"].get("json_files", [])):
        key, name, p = (("dino", side, L.STAGE_DINO / side) if side.startswith("dino_")
                        else ("idol", f"262k/{side}", STAGE / side))
        if name in lfs[key]:
            L.check(lfs[key][name] == L.sha256(p), f"Hub {key}: LFS sha256 of {name} != staged")
        else:
            L.check(blobs[key].get(name) == L.git_blob_id(p), f"Hub {key}: blob id of {name} != staged")
    L.check(not idol["private"] and not idol["gated"], "litert-community/Idol should be public, not gated")
    return {"idol": {"sha": idol["sha"], "lastModified": idol["lastModified"]}, "dino": {"sha": dino["sha"]},
            "n_idol": n_idol, "n_dino": n_dino}


def installed_models(hold: Path, mf: dict) -> list[str]:
    """lc_records.installed() without its fixtures step: this hold ran the models step only. The armchair fixture was
    last checked on this phone by hold H's fixtures step (fixtures_checked()), and the photo's .ply is byte-identical to
    the first release's app's, which needs the same inputs."""
    text = (hold / "driver.out").read_text(errors="replace")
    lines = [x.strip() for x in text.splitlines() if x.startswith(("Up to date: models/", "Installing models/"))]
    seen = {re.sub(r"^(Up to date:|Installing) models/", "", x).split(" ")[0] for x in lines}
    want = {g["file"] for g in mf["graphs"].values()} | set(mf["man"].get("side_files", {})) | \
        set(mf["man"].get("json_files", [])) | {"manifest.json"}
    want.discard("dis.tflite")
    L.check(want <= seen, f"{L.rel(hold)}: install lines miss {sorted(want - seen)}")
    L.check("models: install" in (hold / "hold.log").read_text(), f"{L.rel(hold)}: no models step")
    return lines


def fixtures_checked() -> tuple[Path, int]:
    h = K / "device/results/s26/H"
    L.check("fixtures: install" in (h / "hold.log").read_text(), "hold H has no fixtures step")
    n = sum(1 for x in (h / "driver.out").read_text(errors="replace").splitlines()
            if x.startswith("Up to date: fx/armchair/"))
    L.check(n > 0, "hold H checked no armchair fixture file")
    return h, n


def doc() -> tuple[dict, str]:
    mf = manifest_facts()
    hub = hub_facts(mf)
    s = L.plan_run(D1, PLAN, RUN)
    r, head, row = s["run"], s["head"], s["row"]
    L.check(head["model"] == "SM-S942Q", f"header model {head['model']}")
    L.check(row["options"]["matting"] == "bir", "the run's matting is not BiRefNet")
    sdk = {35: 15, 36: 16, 37: 17}[head["android_sdk"]]
    os_build = f"Android {sdk} (build {head['fingerprint'].split('/')[3].split(':')[0]})"
    date = L.jst_date(head["start_ms"])
    cited = L.rel(s["dir"] / f"{PLAN}.{RUN}.json")
    ship_row = next((x for dev, x in L.ladder_ship_rows() if dev == "Galaxy S26, the app" and cited in x), None)
    L.check(ship_row is not None, f"no ship row of the ladder cites {cited}")
    install = installed_models(D1, mf)
    fx_hold, fx_n = fixtures_checked()
    dl = L.delegate_lines(D1 / f"{PLAN}.tflite.txt")
    L.check(dl["gpu"] >= 1, "no GPU delegate line")
    tw = L.J(K / "contract/c_s26_262k_cool_thermal.json")
    cool = re.search(r"cool after (\d+) s", (D1 / "form.log").read_text())
    L.check(cool is not None, "no cool-wait line in D1/form.log")
    ply = s["dir"] / f"{PLAN}.{RUN}.ply"
    ax_ply = K / "device/results/s26/D0/e2e_axolotl/e2e_axolotl.0.ply"   # this app, the same graph bytes
    m = {
        "first_photo_wall_s": r["total"],
        "generate_s": r["generate"],
        "graphs_compiled_at_app_start": float(row["load_loaded"]),
        "compile_s_at_app_start": row["load_total_s"],
        "graphs_compiled_in_photo": float(r["compiles_in_run"]["count"]),
        "compile_s_in_photo": r["compiles_in_run"]["seconds"],
        "thermal_status_start": float(r["thermal_start"]),
        "thermal_status_end": float(r["thermal_end"]),
        "skin_temp_c_start": float(tw["skin_c_start"]),
        "gpu_clock_cap_mhz_start": float(tw["cap_mhz_start"]),
        "gpu_clock_cap_mhz_min": float(tw["cap_mhz_min"]),
        "vmhwm_kb": float(r["vmhwm_kb"]),
        "gaussians": float(r["gaussians"]),
        "delegate_lines": float(dl["n"]),
        "delegate_lines_gpu_whole": float(dl["gpu"]),
    }
    L.check(r["gaussians"] == 262144, f"gaussians {r['gaussians']}")
    for k, v in r["stages"].items():
        m[f"stage_{k}_s"] = v
    b = {"armchair": (K / "contract/b_armchair_s26.json", ply), "axolotl": (K / "contract/b_axolotl_s26.json", ax_ply)}
    for tag, (bpath, p) in b.items():
        m.update(L.b_metrics(bpath, tag, p))
    sb = K / "contract/b_succulent_s26.json"
    sply = K / L.J(sb)["ply"]
    m.update(L.b_metrics(sb, "first_release_app_succulent", sply))
    am, worst = L.a_metrics(K / "contract/a_armchair_s26.json", "first_release_app_line_a_")
    m.update(am)
    cc, ct = L.J(K / "contract/c_s26_cool.json"), L.J(K / "contract/c_s26_cool_thermal.json")
    L.check(cc["runs_measured"] == 1 and cc["first_only"] is True, "c_s26_cool.json is not one first photo")
    m["first_release_app_first_photo_wall_s"] = cc["median_total_s"]
    m["first_release_app_first_photo_thermal_status_start"] = float(ct["status_start"])
    m["first_release_app_first_photo_skin_temp_c_start"] = float(ct["skin_c_start"])
    cw, h3 = L.J(K / "contract/c_s26.json"), L.J(K / "contract/e2e_s26_hold3.json")
    warm = [(f"nb-armchair (run {x['run']})", x["total_s"], t["start"]) for x, t in zip(cw["per_run"], cw["thermal"])]
    warm += [(x["input"], x["total_s"], x["thermal_start"]) for x in h3["runs"]]
    for i, (_, total, th) in enumerate(warm, 1):
        m[f"first_release_app_warm_photo{i}_wall_s"] = total
        m[f"first_release_app_warm_photo{i}_thermal_status_start"] = float(th)
    m["lane_lines_pass"] = 1.0
    rec = L.base_record(
        f"pipeline of {len(mf['graphs'])} graphs (262k/manifest.json): the app ran {row['load_of']} on the GPU at GPU "
        "precision FP16 with fp32 accumulation and BiRefNet (bir.tflite) on the CPU (XNNPACK, 6 threads) as the manifest "
        "sets; the alternative matting graph dis.tflite (litert-community/DIS-ISNet-LiteRT) was not used; the "
        "LiteRT-Models sample app idol (debug build) from its autorun plan, the first photo after the app start (the row)",
        date)
    rec["peak_mem_mb"] = round(m["vmhwm_kb"] / 1024, 1)
    rec["metrics"] = m
    rec["env"] = {"device": "Galaxy S26 (SM-S942Q)", "machine_label": "galaxy-s26-idol-app", "os_build": os_build,
                  "runtime": "litert", "runtime_version": L.RUNTIME,
                  "soc": "Qualcomm SM8850 (Adreno GPU, Hexagon v81 HTP)", "vendor_sdk": None}
    ev = [f"SPEED_LADDER.md ship row: {ship_row}",
          f"{L.rel(s['dir'] / 'header.json')}: device={head['device']} fingerprint={head['fingerprint']} "
          f"litert={head['litert']} mem_total_kb={head['mem_total_kb']} thermal={head['thermal']}",
          f"{cited}: total={r['total']} s generate={r['generate']} s stages={json.dumps(r['stages'])} "
          f"compiles_in_run={r['compiles_in_run']['count']} ({r['compiles_in_run']['seconds']:.1f} s) "
          f"vmhwm_kb={r['vmhwm_kb']} thermal {r['thermal_start']} -> {r['thermal_end']} gaussians={r['gaussians']}",
          f"{L.rel(s['dir'] / 'DONE.json')}: {row['load_loaded']} of {row['load_of']} GPU graphs compiled at the app "
          "start, before the photo (the app compiles each one during the photo)",
          f"{L.rel(D1 / f'{PLAN}.tflite.txt')}: {dl['n']} delegate lines ({', '.join(dl['kinds'])}); each of the "
          f"{dl['gpu']} GPU lines reads 'Replacing N out of N node(s) with delegate (LITERT_CL) node, yielding 1 "
          f"partitions': N = {L.counts_text(dl['counts'])}; first: {dl['first']}",
          f"contract/c_s26_262k_cool_thermal.json: the phone's GPU clock cap (kgsl max_gpuclk), thermal status and skin "
          f"temperature read every 10 s by adb during {tw['run']}: {tw['samples']} reads, cap "
          f"{tw['cap_mhz_start']:.0f} MHz at the start, lowest {tw['cap_mhz_min']:.0f} MHz; thermal status "
          f"{tw['status_start']} -> max {tw['status_max']}; skin {tw['skin_c_start']} C at the start",
          f"{L.rel(D1 / 'driver.out')}: install_to_device.sh checks every file's SHA-256 on the phone against "
          "manifest.json (graphs) or the staged copy (side and JSON files) and stops on a mismatch: "
          f"{sum(1 for x in install if x.startswith('Up to date'))} files already equal, "
          f"{sum(1 for x in install if x.startswith('Installing'))} copied and checked",
          f"{L.rel(fx_hold / 'driver.out')}: the last fixtures step on this phone before the row: {fx_n} files of "
          "fx/armchair already equal to the staged fixture (SHA-256 on the phone)"]
    for tag, bpath, p in (("armchair", b["armchair"][0], ply), ("axolotl", b["axolotl"][0], ax_ply),
                          ("succulent", sb, sply)):
        bj = L.J(bpath)
        ev.append(f"{L.rel(bpath)}: {tag} PSNR {bj['psnr_db']:.2f} dB (4 views pooled; line 25 dB), per view "
                  f"{', '.join(f'{v:.2f}' for v in bj['psnr_per_view_db'])}; scored .ply {bj['ply']} sha256 "
                  f"{bj['ply_sha256'][:16]}… = {L.rel(p)}")
    ev.append(f"contract/a_armchair_s26.json ({L.J(K / 'contract/a_armchair_s26.json')['run']}, the first release's "
              f"app): line (a) {int(am['first_release_app_line_a_lines_pass'])} / "
              f"{int(am['first_release_app_line_a_lines'])} PASS (cos >= 0.99), lowest {worst} cos "
              f"{am['first_release_app_line_a_min_cos']:.6f}, matting IoU {am['first_release_app_line_a_matting_iou']}")
    ev.append(f"contract/c_s26_cool.json + contract/c_s26_cool_thermal.json (the first release's app, {ct['run']}): "
              f"first photo after start {cc['median_total_s']} s, thermal status {ct['status_start']} and skin "
              f"{ct['skin_c_start']} C at the start")
    ev.append("contract/c_s26.json + contract/e2e_s26_hold3.json (the first release's app, a phone already warm): "
              + "; ".join(f"photo {i}: {name} {total} s, thermal status {th} at the start"
                          for i, (name, total, th) in enumerate(warm, 1)))
    rec["evidence"] = ev
    notes = (
        "Idol for LiteRT (litert-community/Idol main " + hub["idol"]["sha"][:8] + ", last modified "
        + hub["idol"]["lastModified"] + "): speridlabs/idol (Apache-2.0) with the speridlabs/idol-turbo LoRA (5 steps, "
        "no guidance) turns one photo of an object into a 3D Gaussian splat; this form makes 262,144 Gaussians (8,192 "
        "anchors, the author's default). It is the first release's form, in the repo's 262k/ folder since the two-form "
        "commit 74728e5c (its files are the bytes of the first release's root at commit e7ee1226); the repo's root holds "
        "the default form (65,536 Gaussians), whose records are model id idol. Raw LiteRT graphs exported with "
        "litert-torch: BiRefNet background removal (bir, CPU), the DINOv3 ViT-H+/16 image encoder (dino_0..3, in "
        "litert-community/DINOv3-ViT-H16plus-Idol-LiteRT main " + hub["dino"]["sha"][:8] + ", gated), the FLUX.2 VAE "
        "encoder (vae_0..3, vae_attn, vae_4), the denoiser (den_ctx, den_in, den_joint_00..23: a graph per joint block), "
        "the octree decoder (oct, 5 signatures by row count) and the Gaussian decoder (gs_0..7). A pipeline record "
        "(DECISIONS #248's shape): `artifact` is 262k/manifest.json, which the app opens; it names the "
        f"{len(mf['graphs'])} graphs with file, bytes, sha256, accelerator and GPU precision (FP16 with fp32 "
        f"accumulation for the {len(mf['gpu'])} GPU graphs, one of them the alternative matting graph dis, which lives "
        "in litert-community/DIS-ISNet-LiteRT and this run did not use). Identity: the sha256 in 262k/manifest.json of "
        f"every graph the app loads equals its Hub LFS sha256 (the {hub['n_idol']} graphs under 262k/ and the "
        f"{hub['n_dino']} DINOv3 graphs), 262k/manifest.json's git blob id and every side and JSON file equal the "
        "uploaded files, and the app's install script checks each file's SHA-256 on the phone. Lines, fixed before the "
        "first run: (a) each part against the author's PyTorch pipeline (float32, CPU) on the same input, cosine >= 0.99 "
        "per compared tensor; (b) the whole pipeline's 4-view render PSNR against the author's default >= 25 dB on three "
        "inputs (nb-armchair and fig-axolotl from the author's gallery, a Pexels photo of a succulent). A pipeline "
        "check, so output_match stays null and the scores are metrics; latency_p50_ms stays null and the photo walls "
        "are metrics (first_photo_wall_s is the conversion's speed-ladder ship row for this form on this phone). "
        "Harness: the same sample app as the default form's records (a debug build of the repo's android/ folder) in "
        f"its autorun mode, pointed at the 262k/ files: the app starts, compiles {row['load_loaded']} of its "
        f"{row['load_of']} GPU graphs before the photo, then makes one photo of nb-armchair from the app's files (plan "
        "e2e_cool, run 0; it compiles each graph during the photo). Before it the run waited for a cool phone: GPU "
        "clock cap 1,300 MHz and thermal status <= 1 in three reads 10 s apart (" + cool[1] + " s of waiting; status "
        f"{r['thermal_start']} and skin {tw['skin_c_start']} C at the photo's start). While the app ran the GPU clock "
        f"cap fell to {tw['cap_mhz_min']:.0f} MHz (thermal status {r['thermal_start']} -> {r['thermal_end']}). The "
        "armchair .ply is byte-identical to the first release's app's, and this app's fig-axolotl .ply from the same "
        "graph bytes (an earlier device run the same night, whose manifest copy differed only in BiRefNet's memory "
        "estimate) is byte-identical to the first release's app's, so those two line (b) scores apply to this app. "
        "The metrics prefixed first_release_app_ come from the first release's app build on the same graph bytes: "
        "line (a) on this phone (its parity plan), the succulent's line (b), the first photo after start that the "
        "first release's card gives (from thermal status 1) and five photos on a phone already warm (thermal status 2: "
        "three of nb-armchair, then the succulent and fig-axolotl). Not rows here: this app's earlier photos of the "
        "same bytes (the device run named above; its first photo started at thermal status 1) and the other forms of "
        "the conversion's speed ladder. The Pixel 11 Pro did not time this form.")
    return ({"schema_version": "1.0", "model_id": MODEL_ID, "device": "galaxy-s26", "artifact": "262k/manifest.json",
             "artifact_sha256": mf["sha"], "quantization": QUANT, "notes": notes, "results": [rec]}, date)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    d, date = doc()
    errors = L.device_run_errors(d)
    L.check(not errors, f"schema errors: {errors}")
    text = json.dumps(d)
    for word in ("session", "supervisor", "claude", "cmux", "portal", "team", "scratchpad", "fable",
                 "opus", "/Users/", "idol_work"):
        L.check(word.lower() not in text.lower(), f"the record carries '{word}'")
    L.check(not re.search(r"\bD-\d{3}\b", text), "the record carries a decision id")
    root = L.LC / "data" / "device_runs"
    target = root / L.RUNTIME / date / f"{MODEL_ID}__galaxy-s26.json"
    if mode == "--preview":
        print(f"preview {L.write_device_run(Path(sys.argv[2]), d, date=date)}")
    elif mode == "--apply":
        print(f"wrote {L.write_device_run(root, d, date=date)}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            out = L.write_device_run(Path(tmp), d, date=date).read_text(encoding="utf-8")
        if not target.is_file():
            print(f"ABSENT {target} (would write {len(out)} B)")
        else:
            print(("SAME " if target.read_text(encoding="utf-8") == out else "DIFFERENT ") + str(target))
    print("ok: hub", L.J(L.HUB_IDOL)["sha"][:8], "| record 1")


if __name__ == "__main__":
    main()

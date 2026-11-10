# Scorer for one phone run of ftfl_bench (one form of one model): the runner's numbers from R/<tag>.stdout.txt and the
# output agreement of R/<tag>.dump/ against a reference, by these lines (fixed before any phone run):
#   det  every reference box with score >= 0.5 is matched 1:1 by a candidate box with score >= 0.4, the same class and
#        IoU >= 0.5, and every candidate box with score >= 0.5 by a reference box the same way
#   pose of the keypoints whose reference score >= 0.3, at least 90 % lie within 0.05 (y, x; the input side = 1);
#        a fixture with no such keypoint is not counted
#   seg  pixels binarized at 0.5 agree >= 99 %, and the mask IoU >= 0.95 where the reference foreground >= 1 %
#   every counted fixture passes = PASS; a non-finite value fails its fixture.
# Reference = the <graph>_A1d run (LiteRT 1.4.0 Interpreter, CPU, threads unset, as react-native-fast-tflite calls it)
# of the same run dir, else of the --ref-dirs (colon-separated run dirs); the A1d run itself is held against the Mac
# CPU outputs (fixtures/mac/<graph>/model0, tools/mac_ref.py), and every run also gets the Mac comparison as a second
# column. R also holds <tag>.leg.json (graph, status, output size, fixture count, the Mac output dir), written by the
# phone script. Tags are <graph>_<form> (det_B2h). Writes RESULTS_DIR/<prefix>_<tag>_<run>.json
# and R/<tag>.verdict (one line).
#   python -I tools/score.py --run-dir R --tag TAG --results DIR --run RUN --prefix s26 [--root DIR]
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

SPLIT = {"det": [("boxes", 100), ("classes", 25), ("scores", 25), ("count", 1)], "pose": [("keypoints", 51)],
         "seg": [("mask", 65536)]}
CEILING_MIN = 0.85


def kv(line):
    return dict(m.groups() for m in re.finditer(r"(\w[\w\[\].]*)=(\S+)", line))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(v, p):  # nearest rank, as the runner
    s = sorted(v)
    k = min(max(int(math.ceil(p / 100.0 * len(s))), 1), len(s))
    return s[k - 1]


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def stats(v):
    if not v:
        return None
    return {"n": len(v), "median": round(median(v), 4), "p10": pct(v, 10), "p90": pct(v, 90), "min": min(v), "max": max(v)}


def parse_stdout(text):
    d = {"rounds": [], "run_ms": [], "ceilings": [], "parity": {}, "header": {}, "model": {}, "parity_summary": {},
         "summary": None, "thread_check": None, "ceiling_before": None, "fatal": [], "done": False, "teardown": False}
    for line in text.splitlines():
        if line.startswith("pid="):
            d["header"] = kv(line)
        elif line.startswith("model[0]="):
            d["model"] = kv(line)
            d["model"]["raw"] = line
        elif line.startswith("parity model[0] fixture "):
            fx = line.split()[3]
            d["parity"][fx] = kv(line)
        elif line.startswith("parity_summary "):
            d["parity_summary"] = kv(line)
        elif re.match(r"round \d+ ms=", line):
            k = kv(line)
            d["rounds"].append(float(k["ms"]))
            d["run_ms"].append(float(k["run_ms"]))
            d["ceilings"].append(float(k.get("ceiling", "-1")))
        elif line.startswith("ceiling_before "):
            d["ceiling_before"] = num(line.split()[1])
        elif line.startswith("thread_check "):
            d["thread_check"] = {k: num(v) for k, v in kv(line).items()}
        elif line.startswith("summary model[0] "):
            d["summary"] = {k: (num(v) if num(v) is not None else v) for k, v in kv(line).items()}
        elif line.startswith(("FATAL", "FIXTURE_ERROR", "WRITE_ERROR")):
            d["fatal"].append(line)
        elif line.startswith("teardown_start"):
            d["teardown"] = True
        elif line.startswith("done wall_ms="):
            d["done"] = True
    return d


def iou(a, b):  # boxes as (ymin, xmin, ymax, xmax)
    y1, x1 = max(a[0], b[0]), max(a[1], b[1])
    y2, x2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(src, dst):
    """Maximum 1:1 matching (Kuhn) of src boxes onto dst boxes; edges = same class and IoU >= 0.5. src / dst = lists
    of (box, cls, score). Returns the pairs (i, j, iou)."""
    edges = [[j for j, d in enumerate(dst) if int(round(d[1])) == int(round(s[1])) and iou(s[0], d[0]) >= 0.5]
             for s in src]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(src)):
        augment(i, set())
    return [(i, j, iou(src[i][0], dst[j][0])) for j, i in owner.items()]


def split(graph, x):
    out, o = {}, 0
    for name, n in SPLIT[graph]:
        out[name] = x[o:o + n]
        o += n
    return out


def agree_det(ref, cand):
    r, c = split("det", ref), split("det", cand)
    rb, cb = r["boxes"].reshape(25, 4), c["boxes"].reshape(25, 4)
    def n_det(x):
        return int(min(25, max(0, round(float(x[0])))))
    rn, cn = n_det(r["count"]), n_det(c["count"])
    R = [(rb[i], r["classes"][i], r["scores"][i]) for i in range(rn)]
    C = [(cb[i], c["classes"][i], c["scores"][i]) for i in range(cn)]
    r_hi, c_lo = [x for x in R if x[2] >= 0.5], [x for x in C if x[2] >= 0.4]
    c_hi, r_lo = [x for x in C if x[2] >= 0.5], [x for x in R if x[2] >= 0.4]
    fw, bw = match(r_hi, c_lo), match(c_hi, r_lo)
    ok = len(fw) == len(r_hi) and len(bw) == len(c_hi)
    return ok, {"ref_boxes": len(r_hi), "cand_boxes": len(c_hi), "pairs": len(fw), "pairs_back": len(bw),
                "min_iou": round(min([p[2] for p in fw], default=1.0), 4),
                "max_abs_score": round(max([abs(float(r_hi[i][2]) - float(c_lo[j][2])) for i, j, _ in fw], default=0.0), 5)}


def agree_pose(ref, cand):
    r, c = ref.reshape(17, 3), cand.reshape(17, 3)
    sel = r[:, 2] >= 0.3
    n = int(sel.sum())
    dist = np.sqrt((r[sel, 0] - c[sel, 0]) ** 2 + (r[sel, 1] - c[sel, 1]) ** 2)
    rate = float((dist <= 0.05).mean()) if n else None
    info = {"keypoints": n, "within_0.05": int((dist <= 0.05).sum()), "rate": None if rate is None else round(rate, 4),
            "max_abs_y": round(float(np.abs(r[sel, 0] - c[sel, 0]).max()), 5) if n else None,
            "max_abs_x": round(float(np.abs(r[sel, 1] - c[sel, 1]).max()), 5) if n else None,
            "max_abs_score": round(float(np.abs(r[:, 2] - c[:, 2]).max()), 5)}
    if n == 0:
        return None, info   # not counted (the table says so)
    return rate >= 0.9, info


def agree_seg(ref, cand):
    rm, cm = ref >= 0.5, cand >= 0.5
    agree = float((rm == cm).mean())
    fg = float(rm.mean())
    union = int((rm | cm).sum())
    io = float((rm & cm).sum()) / union if union else 1.0
    ok = agree >= 0.99 and (fg < 0.01 or io >= 0.95)
    d = np.abs(ref - cand)
    return ok, {"pixel_agreement": round(agree, 5), "ref_foreground": round(fg, 5), "iou": round(io, 5),
                "iou_counted": fg >= 0.01, "max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6)}


AGREE = {"det": agree_det, "pose": agree_pose, "seg": agree_seg}


def compare(graph, ref_dir, cand_dir, count, n_elems):
    """Per fixture: the task's line, the raw max |d| and non-finite values. Verdict PASS when every counted fixture
    passes, FAIL otherwise, NO_DUMPS / NO_REF when files are missing."""
    if not ref_dir or not Path(ref_dir).is_dir():
        return {"verdict": "NO_REF", "reference": str(ref_dir) if ref_dir else None}
    res, max_dp, counted, failed = {}, 0.0, 0, 0
    for k in range(count):
        id3 = f"{k:03d}"
        fr, fc = Path(ref_dir) / f"{id3}.f32", Path(cand_dir) / f"{id3}.f32"
        if not fc.exists() or fc.stat().st_size != n_elems * 4:
            return {"verdict": "NO_DUMPS", "reference": str(ref_dir), "missing": str(fc)}
        if not fr.exists() or fr.stat().st_size != n_elems * 4:
            return {"verdict": "NO_REF", "reference": str(ref_dir), "missing": str(fr)}
        r, c = np.fromfile(fr, dtype="<f4").astype(np.float64), np.fromfile(fc, dtype="<f4").astype(np.float64)
        nf = int((~np.isfinite(c)).sum())
        dp = float(np.nanmax(np.abs(r - c))) if nf == 0 else float("inf")
        max_dp = max(max_dp, dp)
        if nf:
            ok, info = False, {"nonfinite": nf}
        else:
            ok, info = AGREE[graph](r, c)
        info["max_abs_raw"] = round(dp, 6) if math.isfinite(dp) else "inf"
        info["pass"] = ok
        res[id3] = info
        if ok is None:
            continue
        counted += 1
        failed += 0 if ok else 1
    verdict = "PASS" if counted and failed == 0 else "FAIL"
    return {"verdict": verdict, "reference": str(ref_dir), "counted": counted, "failed": failed,
            "max_abs_dp": round(max_dp, 6) if math.isfinite(max_dp) else "inf", "fixtures": res}


def leg_done(path):
    try:
        return json.loads(Path(path).read_text()).get("status") == "DONE"
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", default="phone")
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--ref-dirs", default="")
    a = ap.parse_args()
    R, root, tag = Path(a.run_dir), Path(a.root), a.tag
    leg = json.loads((R / f"{tag}.leg.json").read_text())
    graph = leg["graph"]
    parts = tag.split("_")
    form = parts[1] if len(parts) > 1 else "?"
    so = parse_stdout((R / f"{tag}.stdout.txt").read_text(errors="replace") if (R / f"{tag}.stdout.txt").exists() else "")
    n_elems = int(leg["out_elems"])
    count = int(leg["count"])
    mac = root / leg["mac_dir"]
    dump = R / f"{tag}.dump"
    # the reference: the A1d run of this run dir, else of the --ref-dirs; A1d itself against the Mac
    if form == "A1d":
        ref_dir, ref_name = mac, "mac_cpu"
    else:
        ref_dir, ref_name = None, None
        for rd in [R] + [Path(x) for x in a.ref_dirs.split(":") if x]:
            if leg_done(rd / f"{graph}_A1d.leg.json") and (rd / f"{graph}_A1d.dump").is_dir():
                ref_dir, ref_name = rd / f"{graph}_A1d.dump", f"{rd.name}/{graph}_A1d"
                break
    agree = compare(graph, ref_dir, dump, count, n_elems) if leg["status"] == "DONE" else {"verdict": "LEG_" + leg["status"]}
    agree["reference_leg"] = ref_name
    agree_mac = compare(graph, mac, dump, count, n_elems) if leg["status"] == "DONE" else {"verdict": "LEG_" + leg["status"]}
    for d, label in ((agree, ref_name), (agree_mac, "mac_cpu")):  # names, not this machine's paths
        d["reference"] = label
        if "missing" in d:
            d["missing"] = Path(d["missing"]).name
    lat = None
    if so["rounds"]:
        cmin = min([c for c in so["ceilings"] + [so["ceiling_before"] or -1] if c is not None], default=-1)
        lat = {"per_call_ms_write_run_read": stats(so["rounds"]), "run_ms": stats(so["run_ms"]),
               "runner_summary": so["summary"], "ceiling_before": so["ceiling_before"], "ceiling_min": cmin,
               "rounds_under_ceiling": sum(1 for c in so["ceilings"] if 0 <= c < CEILING_MIN),
               # a run whose CPU clock ceiling fell under 0.85 of the maximum is not used
               "ceiling_ok": None if cmin < 0 else cmin >= CEILING_MIN}
    ps = so["parity_summary"]
    max_abs_dp = agree.get("max_abs_dp") if agree.get("verdict") in ("PASS", "FAIL") else agree_mac.get("max_abs_dp")
    doc = {"tag": tag, "run": a.run, "device": a.prefix, "graph": graph, "form": form,
           "runtime": so["header"].get("runtime"), "accel": leg["accel"], "precision": leg["precision"],
           "cpu_threads_flag": so["header"].get("cpu_threads"), "status": leg["status"], "killed_by": leg.get("killed_by"),
           "ready_note": leg.get("ready_note"), "start": leg.get("start"),
           "load_ms": num(so["model"].get("load_ms")), "first_call_ms": num(ps.get("first_call_ms")),
           "first_run_ms": num(ps.get("first_run_ms")), "model_line": so["model"].get("raw"),
           "threads": so["thread_check"], "threads_after_load": num(so["model"].get("threads_after_load")),
           "latency_phase": lat, "fatal": so["fatal"], "done_line": so["done"], "teardown_line": so["teardown"],
           "delegation": [l for l in (R / f"{tag}.delegate.txt").read_text(errors="replace").splitlines()[:8]]
           if (R / f"{tag}.delegate.txt").exists() else [],
           "agreement": agree, "agreement_vs_mac": agree_mac,
           "summary": {"max_abs_dp": max_abs_dp, "agreement": agree.get("verdict"), "agreement_vs_mac": agree_mac.get("verdict"),
                       "median_ms": (lat or {}).get("per_call_ms_write_run_read", {}).get("median") if lat else None,
                       "ceiling_ok": (lat or {}).get("ceiling_ok") if lat else None}}
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.prefix}_{tag}_{a.run}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    s = doc["summary"]
    line = (f"{tag} {leg['status']} median_ms={s['median_ms']} agreement={s['agreement']} (ref {ref_name}) "
            f"vs_mac={s['agreement_vs_mac']} max_abs_dp={s['max_abs_dp']} ceiling_ok={s['ceiling_ok']} "
            f"threads={(so['thread_check'] or {}).get('tasks_with_cpu')}")
    (R / f"{tag}.verdict").write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    sys.exit(main())

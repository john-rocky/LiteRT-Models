# The table of the Galaxy S26 runs, from the scorer's results (tools/score.py), one JSON per run.
#   python -I tools/table.py <results dir> [<results dir> ...] [--out TABLE] [--prefix s26]
# Writes <out>.md: a compact table (rows x models, median ms per call), the full table (p10-p90, load, first call, CPU
# threads used, agreement) and, per model, the fastest row of each version whose outputs agree; and <out>.json: every
# number in it.
import argparse
import json
import re
from pathlib import Path

ALL_FORMS = ["A1d", "A1t4", "A1x", "A1xt4", "A2d", "A2h", "A3n", "B1d", "B1t4", "B2d", "B2h", "B2f", "B2g", "B2dH", "B2hH",
             "B2fH", "B3c", "B3", "B3b", "X1t4"]
GRAPHS = ["det", "pose", "seg"]
DESC = {"A1d": "1.4.0 Interpreter, CPU, threads unset", "A1t4": "1.4.0 Interpreter, CPU, 4 threads",
        "A1x": "1.4.0 Interpreter + XNNPACK delegate, the delegate's default threads",
        "A1xt4": "1.4.0 Interpreter + XNNPACK delegate, 4 threads",
        "A2d": "1.4.0 GPU delegate, default options (max precision first)",
        "A2h": "1.4.0 GPU delegate, precision loss allowed (min latency first)", "A3n": "1.4.0 NNAPI delegate",
        "B1d": "2.3.2 CompiledModel, CPU, default", "B1t4": "2.3.2 CompiledModel, CPU, 4 threads",
        "B2d": "2.3.2 CompiledModel, GPU + CPU, default precision", "B2h": "2.3.2 CompiledModel, GPU + CPU, fp16",
        "B2f": "2.3.2 CompiledModel, GPU + CPU, fp32", "B2g": "2.3.2 CompiledModel, GPU alone, default precision",
        "B2dH": "2.3.2 CompiledModel, GPU + CPU, default precision, host-memory buffers",
        "B2hH": "2.3.2 CompiledModel, GPU + CPU, fp16, host-memory buffers",
        "B2fH": "2.3.2 CompiledModel, GPU + CPU, fp32, host-memory buffers",
        "B3c": "2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, no cache",
        "B3": "2.3.2 CompiledModel, NPU + CPU, compiled on the phone, read from B3c's cache",
        "B3b": "2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, HTP burst mode",
        "X1t4": "2.3.2 libLiteRt.so through the TFLite C API, CPU, 4 threads"}
# What code that calls the TFLite C API as react-native-fast-tflite 3.0.1 does would change to get the row. Read from
# the runner, not built in the wrapper.
NEEDS = {"A1d": "none (3.0.1 makes this call)", "A2d": "none (3.0.1 makes this call)",
         "A3n": "none (3.0.1 makes this call)",
         "A1t4": "set a thread count (3.0.1 does not)", "A2h": "set the GPU options above (3.0.1 does not)",
         "A1x": "add the XNNPACK delegate (3.0.1 does not)",
         "A1xt4": "add the XNNPACK delegate and a thread count (3.0.1 does neither)",
         "X1t4": "version bump, the new .so name in CMake, headers kept from 1.4.0, a thread count; no GPU or NNAPI "
                 "delegate (their C functions are not in the 2.3.2 libraries)",
         "B1d": "rewrite to the CompiledModel API", "B1t4": "rewrite to the CompiledModel API",
         "B2d": "rewrite to the CompiledModel API", "B2h": "rewrite to the CompiledModel API",
         "B2f": "rewrite to the CompiledModel API", "B2g": "rewrite to the CompiledModel API",
         "B2dH": "rewrite to the CompiledModel API; host-memory tensor buffers",
         "B2hH": "rewrite to the CompiledModel API; host-memory tensor buffers",
         "B2fH": "rewrite to the CompiledModel API; host-memory tensor buffers",
         "B3c": "rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app",
         "B3": "rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app",
         "B3b": "rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app; the HTP burst mode in the "
                "Qualcomm options"}


def load(dirs, prefix):
    """The scorer's results by (model, row); a later run of the same row replaces an earlier one."""
    legs = {}
    for d in dirs:
        for p in sorted(Path(d).glob(f"{prefix}_*.json")):
            doc = json.loads(p.read_text())
            if re.fullmatch(r"[a-z]+_[A-Za-z0-9]+", doc["tag"]):
                legs[(doc["graph"], doc["form"])] = doc
    return legs


def agreement(doc):
    """The run's agreement verdict and its reference. A model whose A1d did not load (seg: the 1.4.0 C API without a
    delegate does not resolve the MediaPipe custom op) has no phone reference; its runs then take the Mac CPU verdict."""
    s = (doc or {}).get("summary") or {}
    if s.get("agreement") == "NO_REF" and s.get("agreement_vs_mac"):
        return s["agreement_vs_mac"], "Mac CPU"
    return s.get("agreement"), "phone A1d"


def usable(doc):
    if not doc or doc.get("status") != "DONE":
        return False, "not DONE" if doc else "not run"
    if agreement(doc)[0] != "PASS":
        return False, f"agreement {agreement(doc)[0]}"
    lat = doc.get("latency_phase") or {}
    if not lat:
        return False, "no latency phase"
    if doc.get("accel") == "cpu" and lat.get("ceiling_ok") is False:
        return False, f"clock ceiling {lat.get('ceiling_min')}"
    return True, ""


def med(doc):
    return ((doc.get("latency_phase") or {}).get("per_call_ms_write_run_read") or {}).get("median")


def fastest(legs, g, forms):
    best, why = None, {}
    for f in forms:
        doc = legs.get((g, f))
        if f == "B3" and not (doc and doc.get("status") == "DONE"):
            doc = legs.get((g, "B3c"))
        ok, reason = usable(doc)
        if not ok:
            why[f] = reason
            continue
        if best is None or med(doc) < med(best[1]):
            best = (f, doc)
    return best, why


def reference_name(leg):
    """The reference of an agreement as a row name ("A1d") or "Mac CPU", not a run folder."""
    if not leg:
        return leg
    return "Mac CPU" if leg == "mac_cpu" else leg.rsplit("_", 1)[-1]


def failure_line(doc):
    """The runner's first failure line without its "FATAL " prefix and without the message the wrapper would show."""
    line = ((doc.get("fatal") or [doc.get("status")])[0]).replace("FATAL ", "")
    return re.sub(r" \(fast-tflite: [^)]*\)", "", line)


def public_run(d):
    """The numbers of one run: a fixed list of fields, no file paths of the phone or the Mac."""
    lat = d.get("latency_phase") or {}
    ag = {k: v for k, v in (d.get("agreement") or {}).items() if k not in ("reference", "missing")}
    agm = {k: v for k, v in (d.get("agreement_vs_mac") or {}).items() if k not in ("reference", "missing")}
    for a in (ag, agm):
        if "reference_leg" in a:
            a["reference_leg"] = reference_name(a["reference_leg"])
    return {"status": d.get("status"), "failure": failure_line(d) if d.get("fatal") else None, "runtime": d.get("runtime"),
            "accel": d.get("accel"), "precision": d.get("precision"), "cpu_threads_flag": d.get("cpu_threads_flag"),
            "load_ms": d.get("load_ms"), "first_call_ms": d.get("first_call_ms"),
            "per_call_ms_write_run_read": lat.get("per_call_ms_write_run_read"), "run_ms": lat.get("run_ms"),
            "cpu_clock_ceiling_min": lat.get("ceiling_min"), "cpu_clock_ceiling_ok": lat.get("ceiling_ok"),
            "threads": d.get("threads"), "agreement": ag, "agreement_vs_mac": agm,
            "delegation": [l for l in d.get("delegation") or [] if "/data/" not in l]}


def fmt(x, nd=2):
    return "-" if x is None else f"{float(x):.{nd}f}"


def row(g, f, doc):
    """One run as a row of the full table."""
    lat = doc.get("latency_phase") or {}
    pc = lat.get("per_call_ms_write_run_read") or {}
    th = (doc.get("threads") or {}).get("tasks_with_cpu")
    if doc.get("status") == "DONE":
        status = "ran"
        ag = agreement(doc)[0] + (" (Mac CPU reference)" if agreement(doc)[1] == "Mac CPU" else "")
    else:  # the runner's own failure line, e.g. "load: LiteRtCreateCompiledModel status 504"
        line = failure_line(doc)
        status = ("does not load: " + line[len("load: "):]) if line.startswith("load: ") else "failed: " + line
        ag = "-"
    med_ = f"{fmt(pc.get('median'))} ({fmt(pc.get('p10'))}-{fmt(pc.get('p90'))})" if pc else "-"
    return (f"| {g} | {f} | {DESC[f]} | {NEEDS[f]} | {status} | {med_} | {fmt(doc.get('load_ms'), 1)} | "
            f"{fmt(doc.get('first_call_ms'))} | {'-' if th is None else int(th)} | {ag} |")


def cell(doc):
    """One model's cell of the compact table: the median per-call ms, FAIL when the agreement fails, or why it has
    no number."""
    if not doc:
        return "-"
    if doc.get("status") != "DONE":
        return "does not load" if failure_line(doc).startswith("load: ") else "failed"
    m = med(doc)
    if m is None:
        return "-"
    return fmt(m) + ("" if agreement(doc)[0] == "PASS" else " FAIL")


def compact(legs):
    """Every row that ran, the three models as columns (median ms per call)."""
    md = ["| row | runtime and options | wrapper change needed | det | pose | seg |", "|---|---|---|---|---|---|"]
    for f in ALL_FORMS:
        if any((g, f) in legs for g in GRAPHS):
            md.append(f"| {f} | {DESC[f]} | {NEEDS[f]} | " + " | ".join(cell(legs.get((g, f))) for g in GRAPHS) + " |")
    return md


def summary_side(best):
    if not best:
        return "none"
    pc = best[1]["latency_phase"]["per_call_ms_write_run_read"]
    return f"{best[0]} {fmt(pc['median'], 3)} ms (p10-p90 {fmt(pc['p10'])}-{fmt(pc['p90'])})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="TABLE")
    ap.add_argument("--prefix", default="s26")
    a = ap.parse_args()
    legs = load(a.dirs, a.prefix)
    out, full, summary = {"models": {}}, [], []
    full.append("| model | row | runtime and options | wrapper change needed | status | median ms (p10-p90) | "
                "load ms | first call ms | CPU threads used | agreement |")
    full.append("|---|---|---|---|---|---|---|---|---|---|")
    a_all = [f for f in ALL_FORMS if f.startswith("A")]
    b_all = [f for f in ALL_FORMS if not f.startswith("A")]  # every 2.3.2 row, X1t4 included
    for g in GRAPHS:
        full.extend(row(g, f, legs[(g, f)]) for f in ALL_FORMS if (g, f) in legs)
        A, _ = fastest(legs, g, a_all)
        B, _ = fastest(legs, g, b_all)
        rec = {"fastest_1_4_0": A and A[0], "median_1_4_0": A and med(A[1]), "fastest_2_3_2": B and B[0],
               "median_2_3_2": B and med(B[1])}
        rec["ratio_1_4_0_over_2_3_2"] = round(med(A[1]) / med(B[1]), 3) if A and B else None
        out["models"][g] = rec
        if A or B:
            ratio = rec["ratio_1_4_0_over_2_3_2"]
            summary.append(f"- {g}: 1.4.0 {summary_side(A)}; 2.3.2 {summary_side(B)}"
                           + (f"; time 1.4.0 \u00f7 2.3.2 = {ratio:.2f}." if ratio else "."))
    out["runs"] = {f"{g}_{f}": public_run(d) for (g, f), d in sorted(legs.items())}
    md = compact(legs) + [""] + summary + [""] + full
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()

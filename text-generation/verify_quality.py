#!/usr/bin/env python3
"""Local quality gate for a converted .litertlm — the `verify` stage of the
convert -> verify -> publish pipeline.

Asks 8 fixed, unambiguously-checkable questions on the Mac via `litert-mac-verify`
(no device push), then scores correctness + degeneracy with the SAME logic as
ios-llm-benchmark `scripts/quality_check.py`.

This is the publish guardrail: a quantization that collapses (wrong answers,
degenerate looping, special-token spam, empty output) must NOT reach
litert-community. MiniCPM5-1B int4 was garbage — this is what catches that.

Note vs the iOS QualityTask: that harness asks all 8 in ONE prompt (for
cross-runtime speed comparison at equal quality). Here each question is asked
SEPARATELY — a reasoning model (<think>) would otherwise burn the 256-token
budget thinking and never reach the later answers (a false fail, not a collapse).
Per-question isolates the model's actual correctness and still catches any
collapse (a broken quant degenerates on every question, not just one).

    python3 scripts/verify_quality.py <model.litertlm> [--min-correct N] [--json out.json]

Exit 0 = PASS (safe to publish), 1 = FAIL (do not publish), 2 = harness error.
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Default location of the prebuilt Mac verifier (release binary). Override with --verifier.
DEFAULT_VERIFIER = Path.home() / "code/litert-mac-verify/.build/release/litert-mac-verify"

# (label, question, answer-regex) — same 8 checks as quality_check.py CHECKS,
# asked one at a time. "Answer briefly." keeps reasoning models from rambling.
SUFFIX = " Answer briefly."
QUESTIONS = [
    ("17+25=42",        "What is 17 + 25?",                                  r"\b42\b"),
    ("capital=Tokyo",   "What is the capital of Japan?",                     r"tokyo"),
    ("opp(hot)=cold",   'What is the opposite of "hot"?',                    r"\bcold\b"),
    ("days/week=7",     "How many days are in a week?",                      r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?',            r"merci"),
    ("8*7=56",          "What is 8 times 7?",                                r"\b56\b"),
    ("0.9>0.11",        "Which is larger: 0.9 or 0.11?",                     r"0\.9"),
    ("rhyme=blue",      'Complete the rhyme: "Roses are red, violets are ___"', r"\bblue\b"),
]


def degenerate(text):
    """True if the output loops or is special-token spam (== quality_check.py)."""
    words = text.split()
    if len(words) >= 10:
        grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
        if grams and Counter(grams).most_common(1)[0][1] >= 3:
            return True
        if len(set(words)) / len(words) < 0.30:
            return True
    if len(text) >= 40 and len(set(text)) < 15:
        return True
    if text.count("<|") >= 5 or text.count("<pad>") >= 5:
        return True
    return False


def run_verifier(verifier, model, prompt, max_tokens):
    """Run litert-mac-verify; return (output_text, decode_tok_s). Raises on failure."""
    proc = subprocess.run(
        [str(verifier), str(model), prompt, "--max-tokens", str(max_tokens)],
        capture_output=True, text=True, timeout=600,
    )
    combined = proc.stdout + "\n" + proc.stderr
    m = re.search(r"^OUTPUT: \[(.*)\]$", combined, re.MULTILINE)
    if not m:
        raise RuntimeError(
            f"could not find OUTPUT line (exit={proc.returncode}).\n"
            f"--- tail ---\n{combined[-1500:]}"
        )
    text = m.group(1).replace("⏎", "\n")  # the tool replaces newlines with ⏎
    tok = None
    t = re.search(r"decode ([\d.]+) tok/s", combined)
    if t:
        tok = float(t.group(1))
    return text, tok


def strip_think(text):
    """Drop a <think>...</think> block so the check sees the final answer. Returns
    "" if a <think> opened but never closed (truncated mid-thought = no real answer)."""
    if "<think>" in text and "</think>" not in text:
        return ""  # think never closed -> the model emitted no final answer
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    ap = argparse.ArgumentParser(description="Local quality gate for a .litertlm")
    ap.add_argument("model", help="path to model.litertlm")
    ap.add_argument("--min-correct", type=int, default=6,
                    help="min correct answers (of 8) to PASS (default 6)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="generation budget per question (default 512; reasoning "
                         "models need room to close their <think> block)")
    ap.add_argument("--verifier", default=str(DEFAULT_VERIFIER),
                    help="path to litert-mac-verify binary")
    ap.add_argument("--json", help="write a JSON report to this path")
    ap.add_argument("--name", help="display name (default: parent dir name)")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        return 2
    if not Path(args.verifier).exists():
        print(f"ERROR: verifier not found: {args.verifier}\n"
              f"  build it: swift build -c release --package-path ~/code/litert-mac-verify",
              file=sys.stderr)
        return 2

    name = args.name or model.parent.name
    print(f"== quality gate: {name} ==")
    print(f"   model: {model}\n")

    results, toks = [], []
    for label, q, pat in QUESTIONS:
        try:
            raw, tok = run_verifier(args.verifier, model, q + SUFFIX, args.max_tokens)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
            return 2
        ans = strip_think(raw)
        ok = bool(re.search(pat, ans.lower()))
        # Degeneracy/collapse is judged on the final answer (not the reasoning):
        # an EMPTY answer (unclosed <think>, or special-token-only output) is a collapse;
        # otherwise apply the looping/spam heuristic. A short valid answer ("7") is fine.
        degen = (not ans.strip()) or degenerate(ans)
        if tok:
            toks.append(tok)
        results.append({"label": label, "question": q, "ok": ok,
                        "degenerate": degen, "answer": ans, "raw": raw, "tok_s": tok})
        shown = ans if ans.strip() else "(no answer — <think> truncated)"
        print(f"   [{'✓' if ok else '·'}]{' ⚠️degen' if degen else '       '} {label:16s} "
              f"-> {' '.join(shown.split())[:90]!r}")

    score = sum(r["ok"] for r in results)
    any_degen = any(r["degenerate"] for r in results)
    median_tok = sorted(toks)[len(toks) // 2] if toks else None
    passed = (score >= args.min_correct) and (not any_degen)

    print(f"\n   correct: {score}/8   degenerate: {'⚠️ YES' if any_degen else 'no'}"
          f"   decode~{median_tok:.0f} tok/s" if median_tok else
          f"\n   correct: {score}/8   degenerate: {'⚠️ YES' if any_degen else 'no'}")
    print(f"   VERDICT: {'✅ PASS — safe to publish' if passed else '❌ FAIL — do NOT publish'}"
          f"  (threshold {args.min_correct}/8, non-degenerate)")

    if args.json:
        report = {
            "name": name, "model": str(model),
            "score": score, "of": 8, "degenerate": any_degen,
            "median_decode_tok_s": median_tok,
            "min_correct": args.min_correct, "passed": passed,
            "questions": [{k: r[k] for k in ("label", "ok", "degenerate", "answer")}
                          for r in results],
        }
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"   wrote {args.json}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

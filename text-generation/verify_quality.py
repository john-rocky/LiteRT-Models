#!/usr/bin/env python3
"""Local quality gate for a converted .litertlm — the `verify` stage of the
convert -> verify -> publish pipeline.

Asks 8 fixed, unambiguously-checkable questions on the Mac via `litert-mac-verify`
(no device push), then scores each final answer (the reply after its reasoning,
strip_think): correctness with the 8 checks of ios-llm-benchmark
`scripts/quality_check.py`, except that the 0.9 vs 0.11 answer must say 0.9 is
the larger number (answer_ok), and degeneracy with the same heuristic.

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
import unicodedata
from collections import Counter
from pathlib import Path

# Default location of the prebuilt Mac verifier (release binary). Override with --verifier.
DEFAULT_VERIFIER = Path.home() / "code/litert-mac-verify/.build/release/litert-mac-verify"

# The 0.9 vs 0.11 row. Its regex also finds "0.9" in the wrong answer "0.11 is larger
# than 0.9.", so answer_ok() reads that row's answer with says_larger() instead.
LARGER_09 = r"0\.9"

# (label, question, answer-regex) — the 8 checks of quality_check.py CHECKS, asked
# one at a time and scored by answer_ok() on the final answer (strip_think()).
# "Answer briefly." keeps reasoning models from rambling.
SUFFIX = " Answer briefly."
QUESTIONS = [
    ("17+25=42",        "What is 17 + 25?",                                  r"\b42\b"),
    ("capital=Tokyo",   "What is the capital of Japan?",                     r"tokyo"),
    ("opp(hot)=cold",   'What is the opposite of "hot"?',                    r"\bcold\b"),
    ("days/week=7",     "How many days are in a week?",                      r"\bseven\b|\b7\b"),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?',            r"merci"),
    ("8*7=56",          "What is 8 times 7?",                                r"\b56\b"),
    ("0.9>0.11",        "Which is larger: 0.9 or 0.11?",                     LARGER_09),
    ("rhyme=blue",      'Complete the rhyme: "Roses are red, violets are ___"', r"\bblue\b"),
]


def _number(value):
    """A regex for one decimal as answers write it: 0.9 also as 0.90 and .9, never inside 10.9 or 0.95."""
    whole, frac = value.split(".")
    lead = "(?:0)?" if whole == "0" else whole
    return rf"(?<![\d.]){lead}\.{frac}0*(?!\d)"


def _plain(text):
    """Lowercase text without markdown and LaTeX marks, full-width forms folded (０．９ -> 0.9), a boxed
    result read as "answer: …", and without questions: a sentence that ends in "?" ("Which is larger: 0.9
    or 0.11?") claims nothing."""
    t = unicodedata.normalize("NFKC", text).lower()
    inner = r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}"
    while re.search(inner, t):
        t = re.sub(inner, r"\1", t)
    t = re.sub(r"\\boxed\{([^{}]*)\}", r" answer: \1 ", t)
    for a, b in (("\\geq", "≥"), ("\\ge", "≥"), ("\\gt", ">"), ("\\leq", "≤"), ("\\le", "≤"), ("\\lt", "<"),
                 ("&gt;", ">"), ("&lt;", "<")):
        t = t.replace(a, b)
    t = re.sub(r"[*`$\\]", " ", t)
    # sentences end at . ? ! 。 or a newline; a "." before a digit belongs to a number (0.9)
    sentences = re.findall(r"(?:[^.?!\n。]|\.(?=\d))*[.?!\n。]|(?:[^.?!\n。]|\.(?=\d))+$", t)
    return re.sub(r"[ \t]+", " ", "".join(s for s in sentences if not s.endswith("?")))


def larger_claims(answer, big, small):
    """[(position, number)] for every place the answer says which of the two numbers is larger, in the text
    says_larger() reads (lowercased, see _plain)."""
    t = _plain(answer)
    num = {big: _number(big), small: _number(small)}
    up = r"(?:larger|lager|bigger|greater|higher|more\s+than)"
    down = r"(?:smaller|lesser|less|lower)"
    # between a number and its verb: same sentence, no other number outside parentheses, no new clause
    # ("0.9 is larger than 0.11 because 9 is greater than 1" claims nothing about 0.11)
    gap = (r"(?:(?!\b(?:because|since|while|but|whereas|although|though)\b)[^.?!\n<>。\d(]"
           r"|\([^()\n]{0,30}\)){0,40}?")
    verb = r"(?:(?:is|was|are)(?:n't)?|'s|would be|will be|remains)"
    claims = []
    for x, y in ((big, small), (small, big)):
        X, Y = num[x], num[y]
        rules = (  # (regex, the number it says is larger, the number when a "not" turns it over)
            (rf"{X}{gap}\b{verb}\s+((?:\w+\s+){{0,2}}?){up}\b", x, y),
            (rf"{X}\s+{up}\s+than\b", x, None),
            (rf"{X}\s*\(\s*(?:the\s+)?{up}\s*\)", x, None),
            (rf"(?<!which is )(?<!which one is )\b{up}(?:\s+\w+){{0,2}}?\s*(?:is|:|=|would be)"
             rf"\s*(?:the\s+)?{X}(?!\s*(?:or|and|vs\.?|versus)\b)", x, None),
            (rf"\banswer\s*(?:is|:|=|would be)\s*(?:the\s+)?(?:number\s+)?{X}(?!\s*(?:or|and|vs\.?|versus)\b)", x, None),
            (rf"{X}\s*(?:>|≥|>=)\s*{Y}", x, None),
            (rf"{X}\s*(?:<|≤|<=)\s*{Y}", y, None),
            (rf"{X}{gap}\b{verb}\s+((?:\w+\s+){{0,2}}?){down}\b", y, x),
            (rf"{X}\s+{down}\s+than\b", y, None),
            (rf"{X}\s*\(\s*(?:the\s+)?{down}\s*\)", y, None),
            (rf"(?<!which is )\b(?:smaller|lesser)(?:\s+\w+){{0,2}}?\s*(?:is|:|=)\s*(?:the\s+)?{X}"
             rf"(?!\s*(?:or|and|vs\.?|versus)\b)", y, None),
            (rf"{X}\s*(?:の方|のほう|方)?\s*(?:が|は)\s*(?:より)?(?:大き|おおき)", x, None),
            (rf"{X}\s*(?:は|が)\s*{Y}\s*より(?:も)?\s*(?:大き|おおき)", x, None),
            (rf"{X}\s*(?:の方|のほう|方)?\s*(?:が|は)\s*(?:より)?(?:小さ|ちいさ)", y, None),
            (rf"{X}\s*(?:は|が)\s*{Y}\s*より(?:も)?\s*(?:小さ|ちいさ)", y, None),
            (rf"(?:大きい|大きな)(?:のは|方は|ほうは|数は|方が|ほうが|は|が|:)\s*{X}", x, None),
        )
        for rx, winner, if_not in rules:
            for m in re.finditer(rx, t):
                negated = if_not is not None and re.search(r"\bnot\b|n't", m.group(0))
                claims.append((m.start(), if_not if negated else winner))
    return sorted(claims)


def says_larger(answer, big, small):
    """True when the answer says `big` is the larger of the two numbers. The last claim decides:
    "A is larger / bigger / greater / higher / more than [B]", "A > B", "B < A", "the larger (one / number)
    is A", "the answer is A" (a boxed result too), "A (larger)", "B is smaller / less / lower", "B (smaller)",
    "the smaller is B", a "not" in such a claim turns it over, and the Japanese "A の方が大きい" /
    "A は B より大きい" / "B の方が小さい". An answer without a claim counts when it names only `big`
    ("0.9", "**0.9**"); naming both numbers without a claim is not an answer."""
    claims = larger_claims(answer, big, small)
    if claims:
        return claims[-1][1] == big
    t = _plain(answer)
    named = {v for v in (big, small) if re.search(_number(v), t)}
    return named == {big}


def answer_ok(pattern, answer):
    """True when the final answer is right: the row's regex found in the lowercased answer; the 0.9 vs 0.11
    row (pattern LARGER_09) counts only an answer that says 0.9 is larger (says_larger)."""
    if pattern == LARGER_09:
        return says_larger(answer, "0.9", "0.11")
    return bool(re.search(pattern, answer.lower()))


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


THOUGHT_TAGS = (("<think>", "</think>"), ("[thought]", "[/thought]"), ("[reasoning]", "[/reasoning]"))


def strip_think(text):
    """The final answer: the reply without its reasoning, so the check never reads words of the thought.
    Everything up to the last closing tag is reasoning (a template that opens <think> in the prompt leaves
    only </think> in the reply; the litert-lm CLI prints the thought channel as [thought] ... [/thought]).
    Returns "" when a thought opens and does not close (cut off mid-thought = no answer)."""
    for opening, closing in THOUGHT_TAGS:
        if closing in text:
            text = text.rsplit(closing, 1)[1]
        if opening in text:
            return ""
    return text.strip()


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
        ok = answer_ok(pat, ans)
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

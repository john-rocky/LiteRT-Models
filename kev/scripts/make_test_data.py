#!/usr/bin/env python3
"""Writes the Kev sample's bundled test data from the conversion run's reference data.

The reference data stays outside the module (see scripts/TEST_DATA.md). This script copies only
what may be redistributed: the SemIf records (MIT), the invented "own" records and the three
invented demo requests. It never writes the transfer-v4 rows (tv4, tv4x, tv4s) into the module.

    python3 scripts/make_test_data.py --kev-work /path/to/kev_work
        app/src/debug/assets/gate_fixtures.json, app/src/test/resources/head_fixture.{json,f32},
        the app's three example requests app/src/main/res/raw/example_*.json (the demo requests,
        byte for byte) and their expected rows and answers app/src/test/resources/examples_oracle.json
        (standard library + numpy)

    /path/to/venv/bin/python scripts/make_test_data.py --kev-work /path/to/kev_work --probes
        also app/src/test/resources/tokenizer_probes.json (and the same file as
        app/src/debug/assets/tokenizer_probes.json for the on-device gate), render_cases.json and
        python_numbers.json. Needs Python 3.12, transformers 5.17, tokenizers, pydantic and the
        author's kev package (kev_work/kev, added to sys.path here).

Sources under --kev-work (their sha256 is checked before anything is written):
    fixtures/requests.json   dfe55fb145df7a3967ed7213ae24d67b5d4ec42da0315fdb409d5e4b702e3a48
    oracle/oracle_0.8b.json  d3792f3fc92e5e6148ef3b927d389f02e9cbbe458787834e70138339c26e4c79
    oracle/hidden_0.8b.npz   9f0cafeb3ea4f984cb9529f1c6b00ca621de1d2cb5b96bbbc49be292ce3535fa
    demo/fixtures/demo_ticket_01.json    bc9dba71c5dd4e011f41d23eb5b48ba25adaa64a015f1160a7baa3ff94a1bafd
    demo/fixtures/demo_incident_02.json  abe9c38841bb54fc8f459d5f970c9dee179ec40e46dddadaebe8b54083ba9ab6
    demo/fixtures/demo_review_03.json    a33fa058c17e5ab1f4d1cf0261a5c1900a9050eef35663d064495d5b34f317f2
    demo/oracle/oracle_demo.json         6b00e472b41ebd9748d87e538074383c40395a86714c4aa10f73aa897713567e
                             (the author's fp32 model on the demo requests)
    demo/oracle/expected_v2_L512_cpu.json  438044bc0dc2593426c17111278b42e8677635bf2b5a5c7301ce227152f997cd
                             (the shipped L512 graph on a desktop CPU, head in numpy float32)
    hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d/
        tokenizer.json       06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523
                             (--probes; the tokenizer the author's checkpoint publishes)
    hf/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/
                             (--probes; what kev.checkpoint loads through AutoTokenizer)
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys

import numpy as np

MODULE = pathlib.Path(__file__).resolve().parents[1]
GATE_ASSET = MODULE / "app/src/debug/assets/gate_fixtures.json"
HEAD_JSON = MODULE / "app/src/test/resources/head_fixture.json"
HEAD_BIN = MODULE / "app/src/test/resources/head_fixture.f32"
PROBES = MODULE / "app/src/test/resources/tokenizer_probes.json"
PROBES_ASSET = MODULE / "app/src/debug/assets/tokenizer_probes.json"
EXAMPLES = {
    "demo_ticket_01": MODULE / "app/src/main/res/raw/example_ticket.json",
    "demo_incident_02": MODULE / "app/src/main/res/raw/example_incident.json",
    "demo_review_03": MODULE / "app/src/main/res/raw/example_review.json",
}
EXAMPLES_ORACLE = MODULE / "app/src/test/resources/examples_oracle.json"
RENDER_CASES = MODULE / "app/src/test/resources/render_cases.json"
PYTHON_NUMBERS = MODULE / "app/src/test/resources/python_numbers.json"

REQUESTS = "fixtures/requests.json"
ORACLE = "oracle/oracle_0.8b.json"
HIDDEN = "oracle/hidden_0.8b.npz"
DEMO_ORACLE = "demo/oracle/oracle_demo.json"
DEMO_GRAPH_CPU = "demo/oracle/expected_v2_L512_cpu.json"
KEV_SNAPSHOT = "hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d"
BASE_SNAPSHOT = "hf/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SHA256 = {
    REQUESTS: "dfe55fb145df7a3967ed7213ae24d67b5d4ec42da0315fdb409d5e4b702e3a48",
    ORACLE: "d3792f3fc92e5e6148ef3b927d389f02e9cbbe458787834e70138339c26e4c79",
    HIDDEN: "9f0cafeb3ea4f984cb9529f1c6b00ca621de1d2cb5b96bbbc49be292ce3535fa",
    "demo/fixtures/demo_ticket_01.json": "bc9dba71c5dd4e011f41d23eb5b48ba25adaa64a015f1160a7baa3ff94a1bafd",
    "demo/fixtures/demo_incident_02.json": "abe9c38841bb54fc8f459d5f970c9dee179ec40e46dddadaebe8b54083ba9ab6",
    "demo/fixtures/demo_review_03.json": "a33fa058c17e5ab1f4d1cf0261a5c1900a9050eef35663d064495d5b34f317f2",
    DEMO_ORACLE: "6b00e472b41ebd9748d87e538074383c40395a86714c4aa10f73aa897713567e",
    DEMO_GRAPH_CPU: "438044bc0dc2593426c17111278b42e8677635bf2b5a5c7301ce227152f997cd",
    KEV_SNAPSHOT + "/tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
}

GATE_SOURCES = ("semif", "own")
GATE_MAX_BYTES = 700_000
HEAD_MAX_BYTES = 300_000
WINDOWS = (512, 1024, 2048)
ORACLE_FIELDS = ("qid", "type", "keys", "row_ids", "decide_idx", "opt_idx", "probs", "answer")


def sha256(path):
    """Returns the hex sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources(kev_work, names):
    """Fails unless every named source has its expected sha256."""
    for name in names:
        actual = sha256(kev_work / name)
        if actual != SHA256[name]:
            sys.exit(f"{name}: sha256 {actual}, expected {SHA256[name]}")


def dump(value):
    """Compact JSON with the characters as they are (ensure_ascii=False)."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def window(length):
    """Smallest graph window that holds a row of `length` tokens."""
    return next(w for w in WINDOWS if length <= w)


def gate_items(records, oracle, sources, excluded):
    """The gate asset's items: request plus per-question oracle, for the given sources."""
    by_id = {}
    for question in oracle["questions"]:
        by_id.setdefault(question["id"], []).append(question)
    usage = {r["id"]: r["usage"]["input_tokens"] for r in oracle["requests"]}
    items = []
    for record in records:
        if record["source"] not in sources or record["id"] in excluded:
            continue
        questions = by_id[record["id"]]
        if [q["qid"] for q in questions] != list(record["request"]["questions"]):
            sys.exit(f"{record['id']}: oracle question order differs from the request")
        items.append({
            "id": record["id"],
            "source": record["source"],
            "request": record["request"],
            "input_tokens": usage[record["id"]],
            "questions": [{field: q[field] for field in ORACLE_FIELDS} for q in questions],
        })
    return items


def write_gate_asset(kev_work, records, oracle):
    """Writes gate_fixtures.json (semif + own), dropping the long own records only if too big."""
    excluded = []
    while True:
        items = gate_items(records, oracle, GATE_SOURCES, excluded)
        questions = [q for item in items for q in item["questions"]]
        windows = {str(w): sum(1 for q in questions if window(len(q["row_ids"])) == w) for w in WINDOWS}
        doc = {
            "version": 1,
            "description": "Kev-0.8B gate fixtures: SemIf authored144 (MIT) and invented records, with the "
                           "author's fp32 oracle per question (row ids, readout indices, probabilities, answer).",
            "sources": {name: SHA256[name] for name in (REQUESTS, ORACLE)},
            "temperature": oracle["temperature"],
            "records": len(items),
            "questions": len(questions),
            "by_source": {s: sum(1 for item in items if item["source"] == s) for s in GATE_SOURCES},
            "windows": windows,
            "excluded": excluded,
            "items": items,
        }
        text = dump(doc) + "\n"
        size = len(text.encode("utf-8"))
        if size <= GATE_MAX_BYTES:
            break
        long_own = [item["id"] for item in items if item["id"].startswith("own_long_")]
        if not long_own:
            sys.exit(f"gate asset is {size} bytes even without the long own records")
        print(f"gate asset {size} bytes > {GATE_MAX_BYTES}: dropping {long_own}")
        excluded = long_own
    GATE_ASSET.parent.mkdir(parents=True, exist_ok=True)
    GATE_ASSET.write_text(text, encoding="utf-8")
    print(f"{GATE_ASSET.relative_to(MODULE)}: {size} bytes, {doc['records']} records, "
          f"{doc['questions']} questions, windows {windows}")


def write_examples(kev_work):
    """The app's example requests: the three invented demo requests ({id, state, questions}) byte
    for byte, and per question the author's oracle and the shipped L512 graph's answer."""
    oracle = json.loads((kev_work / DEMO_ORACLE).read_text(encoding="utf-8"))
    graph = json.loads((kev_work / DEMO_GRAPH_CPU).read_text(encoding="utf-8"))
    if graph["oracle"]["file_sha256"] != SHA256[DEMO_ORACLE]:
        sys.exit(f"{DEMO_GRAPH_CPU} was computed against another oracle file")
    oracle_rows = {(q["id"], q["qid"]): q for q in oracle["questions"]}
    graph_rows = {(q["id"], q["qid"]): q for q in graph["questions"]}
    usage = {r["id"]: r["usage"]["input_tokens"] for r in oracle["requests"]}
    examples = []
    for record_id, path in EXAMPLES.items():
        source = f"demo/fixtures/{record_id}.json"
        data = (kev_work / source).read_bytes()
        request = json.loads(data)
        if request["id"] != record_id:
            sys.exit(f"{source} holds {request['id']}")
        questions = []
        for qid in request["questions"]:
            question, shipped = oracle_rows[(record_id, qid)], graph_rows[(record_id, qid)]
            if shipped["keys"] != question["keys"] or shipped["row_len"] != len(question["row_ids"]):
                sys.exit(f"{record_id}/{qid}: the graph's row differs from the oracle's")
            questions.append({
                **{field: question[field] for field in ORACLE_FIELDS},
                "graph_l512_cpu": {"probs": shipped["probs"], "answer": shipped["answer"]},
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        examples.append({"file": path.name, "id": record_id, "sha256": SHA256[source],
                         "input_tokens": usage[record_id], "questions": questions})
        print(f"{path.relative_to(MODULE)}: {source}, {len(questions)} questions")
    doc = {
        "version": 1,
        "description": "The app's three example requests (res/raw, the demo requests byte for byte): per "
                       "question the author's fp32 oracle (row ids, readout indices, probabilities, answer) "
                       "and graph_l512_cpu, the shipped L512 graph on a desktop CPU with the head in numpy "
                       "float32 (probabilities and the 4-decimal answer the app shows).",
        "sources": {name: SHA256[name] for name in (DEMO_ORACLE, DEMO_GRAPH_CPU)},
        "examples": examples,
    }
    EXAMPLES_ORACLE.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_ORACLE.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{EXAMPLES_ORACLE.relative_to(MODULE)}: {sum(len(e['questions']) for e in examples)} questions")


def write_head_fixture(kev_work, oracle):
    """Writes the first question of every own record: h_sel rows (float32 LE) and the oracle."""
    hidden = np.load(kev_work / HIDDEN)
    first = {}
    for question in oracle["questions"]:
        if question["source"] == "own":
            first.setdefault(question["id"], question)
    chunks, items, offset = [], [], 0
    for record_id, question in first.items():
        rows = np.asarray(hidden[f"{record_id}/{question['qid']}"], dtype="<f4")
        if rows.shape != (1 + len(question["keys"]), 1024):
            sys.exit(f"{record_id}: h_sel shape {rows.shape}")
        chunks.append(rows.tobytes())
        items.append({
            "id": record_id, "qid": question["qid"], "type": question["type"], "keys": question["keys"],
            "rows": rows.shape[0], "offset_floats": offset,
            "z_pre": question["z_pre"], "z_post": question["z_post"], "probs": question["probs"],
        })
        offset += rows.size
    data = b"".join(chunks)
    doc = {
        "version": 1,
        "description": "Kev-0.8B pointer-head fixture: hidden states [decide, option 1..K] after the final "
                       "RMSNorm (head_fixture.f32, float32 little-endian, row after row) with the author's oracle.",
        "sources": {name: SHA256[name] for name in (ORACLE, HIDDEN)},
        "hidden_size": 1024,
        "temperature": oracle["temperature"],
        "floats": offset,
        "items": items,
    }
    text = dump(doc) + "\n"
    size = len(data) + len(text.encode("utf-8"))
    if size > HEAD_MAX_BYTES:
        sys.exit(f"head fixture is {size} bytes > {HEAD_MAX_BYTES}")
    HEAD_BIN.parent.mkdir(parents=True, exist_ok=True)
    HEAD_BIN.write_bytes(data)
    HEAD_JSON.write_text(text, encoding="utf-8")
    print(f"head fixture: {len(items)} questions, {len(data)} + {len(text.encode('utf-8'))} bytes")


PROBE_TEXTS = [
    ("devanagari", "नमस्ते दुनिया"),
    ("devanagari", "हिन्दी में लिखा गया वाक्य।"),
    ("added_extra", "a<think>b</think>c"),
    ("added_extra", "<tool_response>ok</tool_response>"),
    ("added_extra", "<tts_pad><tts_text_bos><tts_text_eod><tts_text_bos_single>"),
    ("added_extra", "<|audio_start|>x<|audio_end|><|audio_pad|>"),
    ("added", "<|endoftext|>"),
    ("added", "before<|im_start|>user\nhi<|im_end|>after"),
    ("added", "<|fim_prefix|>state<|fim_middle|>q<|box_start|>o<|box_end|><|fim_suffix|>"),
    ("added", "a<tool_call>b</tool_call>c"),
    ("added", "<|unknown_name|> and <| spaced |> and <|a-b|>"),
    ("added", "<|<|endoftext|>|>"),
    ("emoji", "❤\ufe0f ok"),
    ("emoji", "👍🏽 thumbs"),
    ("emoji", "👨\u200d👩\u200d👧 family"),
    ("emoji", "🇯🇵🇺🇸 flags"),
    ("emoji", "1\ufe0f\u20e3 keycap"),
    ("emoji", "🤔💜 mixed"),
    ("nfc", "cafe\u0301"),
    ("nfc", "café"),
    ("nfc", "Å ngstrom"),
    ("nfc", "가 jamo"),
    ("nfc", "x\u0301y\u0308"),
    ("whitespace", "\r\n"),
    ("whitespace", "line one\r\nline two\n\n\nthree"),
    ("whitespace", "  leading and trailing  "),
    ("whitespace", "tab\tseparated\t\tvalues"),
    ("whitespace", "nbsp\u00a0ideographic\u3000line para end"),
    ("whitespace", "zero\u200bwidth"),
    ("whitespace", "next\u0085line"),
    ("whitespace", " \n "),
    ("whitespace", "a  b   c    d"),
    ("digits", "1234"),
    ("digits", "1234567890 3.14159 -12,345.67"),
    ("digits", "１２３ fullwidth"),
    ("digits", "Ⅻ ² ½ numbers"),
    ("contraction", "don't I'LL we'Re they've she'd it's"),
    ("contraction", "'ſtuff it'ſ ſ"),
    ("contraction", "x'K kelvin"),
    ("script", "日本語のテキストです。"),
    ("script", "한국어 문장"),
    ("script", "ภาษาไทย"),
    ("script", "שָׁלוֹם עולם"),
    ("script", "مرحبا بالعالم"),
    ("script", "Ünïcödé façade naïve"),
    ("script", "ﬁ ligature"),
    ("punctuation", "!!!??? ... —–- “quoted” ’s"),
    ("punctuation", "{\"a\": 1, \"b\": [true, null]}"),
    ("punctuation", "https://example.com/a?b=c&d=e#f user@example.com"),
    ("code", "def f(x):\n    return x * 2\n"),
    ("control", "\x00\x01\x1f end"),
    ("plain", "Hello world"),
    ("plain", ""),
    ("plain", "a" * 64),
]


def write_probes(kev_work):
    """Probe strings through the oracle's tokenizer path; the raw Kev tokenizer.json must agree."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(kev_work / "kev"))
    import tokenizers
    import transformers
    from kev.model import _SPECIAL_RE, user_tokens

    oracle_tokenizer = transformers.AutoTokenizer.from_pretrained(str(kev_work / BASE_SNAPSHOT))
    published = tokenizers.Tokenizer.from_file(str(kev_work / KEV_SNAPSHOT / "tokenizer.json"))
    cases, mismatches = [], []
    for index, (category, text) in enumerate(PROBE_TEXTS):
        raw = oracle_tokenizer(text, add_special_tokens=False).input_ids
        user = user_tokens(oracle_tokenizer, text)
        published_raw = published.encode(text, add_special_tokens=False).ids
        published_user = published.encode(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids
        if raw != published_raw or user != published_user:
            mismatches.append({"text": text, "raw": raw, "published_raw": published_raw})
        cases.append({"id": f"probe_{index:02d}", "category": category, "text": text, "raw_ids": raw, "user_ids": user})
    if mismatches:
        sys.exit(f"published tokenizer.json disagrees with AutoTokenizer: {mismatches}")
    doc = {
        "version": 1,
        "description": "Edge strings through AutoTokenizer (raw_ids: tokenizer(text, add_special_tokens=False); "
                       "user_ids: kev.model.user_tokens). The published Kev tokenizer.json gives the same ids.",
        "tokenizer_json_sha256": SHA256[KEV_SNAPSHOT + "/tokenizer.json"],
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "cases": cases,
    }
    text = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
    for path in (PROBES, PROBES_ASSET):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"{PROBES.relative_to(MODULE)} (+ debug asset): {len(cases)} probes, AutoTokenizer == published tokenizer.json")


RENDER_VALUES = [
    None, "", "plain text", "  leading spaces kept at top level", True, False, 0, 1, -7, 10**20,
    64.9, 1840.0, 0.1, 1e16, 1e-05, 0.0001, -0.0, 3.0e10, 123456789.123, 2.5e-07,
    [], {}, ["a", "b"], [1, 2.5, True, None],
    {"a": 1, "b": "x", "c": None, "d": False},
    {"outer": {"inner": {"deep": 1.5}, "list": [1, {"k": "v", "w": [True, "t"]}]}},
    [["nested", "list"], {"in": "list"}, [], {}],
    {"empty_dict": {}, "empty_list": [], "after": "x"},
    ["  indented item", "\n\nnewlines first", "\u3000ideographic space", "\x1cfile separator"],
    {"Ünïcödé key": "välue", "emoji 💜": ["🤔", "x"]},
    [{"name": "Rowan", "qty": 2, "price": 12.5}, {"name": "Ilse", "qty": 1, "price": 1840.0}],
]


def write_render_cases(kev_work):
    """The author's render / option_text / to_record on edge inputs, and requests pydantic rejects."""
    sys.path.insert(0, str(kev_work / "kev"))
    from kev.api import SystemOneRequest, option_text, render, to_record

    renders = [{"value": v, "indent": i, "text": render(v, i)} for v in RENDER_VALUES for i in (0, 1)]
    options = [{"name": n, "description": d, "text": option_text(n, d)}
               for n in ("no", "yes", "a") for d in (None, "", "desc", 0, False, {}, [], {"k": 1}, [1, "x"], 0.5)]
    requests = [
        {"state": "s", "questions": {"q": {"type": "noul"}}},
        {"state": "s", "questions": {"q": {"type": "noul", "criteria": {}}}},
        {"state": "s", "questions": {"q": {"type": "noul", "criteria": {"true": "it holds", "other": 1}}}},
        {"state": "s", "questions": {"q": {"type": "noul", "criteria": {"false": "", "true": None}}}},
        {"state": {"a": [1, 2]}, "questions": {"q": {"type": "choice", "instructions": {"task": "pick"},
                                                       "criteria": {"x": None, "y": "", "z": 0, "w": False}}}},
        {"state": ["one", {"two": 2.0}], "model": "kev-0.8b",
         "questions": {"s": {"type": "score", "instructions": "", "criteria": ["low", {"level": 2}, None, 3.5]},
                       "n": {"type": "noul", "instructions": None, "criteria": None}}},
        {"state": None, "questions": {"q": {"type": "choice", "criteria": {"only": "one option"}}}},
        {"state": 12, "extra": "ignored", "questions": {"q": {"type": "score", "criteria": [True]}}},
    ]
    invalid = [
        {"questions": {"q": {"type": "noul"}}},
        {"state": "s", "questions": {}},
        {"state": "s"},
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}},
        {"state": "s", "questions": {"q": {"type": "choice"}}},
        {"state": "s", "questions": {"q": {"type": "score", "criteria": []}}},
        {"state": "s", "questions": {"q": {"type": "score", "criteria": {"a": 1}}}},
        {"state": "s", "questions": {"q": {"type": "noul", "criteria": ["a"]}}},
        {"state": "s", "questions": {"q": {"type": "other"}}},
        {"state": "s", "questions": {"q": "not an object"}},
        {"state": "s", "model": 3, "questions": {"q": {"type": "noul"}}},
        {"state": "s", "model": None, "questions": {"q": {"type": "noul"}}},
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}}},
    ]
    records = []
    for request in requests:
        record, meta = to_record(SystemOneRequest.model_validate(request))
        records.append({"request": request, "record": record, "meta": meta})
    for request in invalid:
        try:
            SystemOneRequest.model_validate(request)
        except Exception:  # pydantic.ValidationError
            continue
        sys.exit(f"pydantic accepted a request listed as invalid: {request}")
    doc = {"version": 1, "description": "kev.api render / option_text / to_record on edge inputs; "
                                        "invalid = requests SystemOneRequest.model_validate rejects.",
           "render": renders, "option_text": options, "to_record": records, "invalid": invalid}
    RENDER_CASES.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{RENDER_CASES.relative_to(MODULE)}: {len(renders)} render, {len(options)} option_text, "
          f"{len(records)} to_record, {len(invalid)} invalid")


def write_python_numbers(kev_work):
    """CPython 3.12 sum / round / repr and the author's answer math on fixed and random inputs."""
    sys.path.insert(0, str(kev_work / "kev"))
    from kev.api import _normalize, choice_confidence, round_prob, score_confidence, to_answers

    if sys.version_info[:2] < (3, 12):
        sys.exit("python_numbers.json needs Python 3.12+ (compensated built-in sum)")
    rng = random.Random(20261003)
    f32 = lambda x: float(np.float32(x))
    sums = [[0.1] * 10, [1e16, 1.0, -1e16], [1.0, 1e100, 1.0, -1e100], [0.3, 0.6, 0.1], [-0.0], [5e-324, 5e-324]]
    sums += [[f32(rng.random()) for _ in range(rng.randint(2, 40))] for _ in range(40)]
    rounds = [0.12345, 0.12355, 0.00005, 0.99995, 0.5, 0.03125, 1 / 3, 2 / 3, 0.00015, 254.00005, -1e-9, 0.0]
    rounds += [f32(rng.random()) for _ in range(60)]
    reprs = [64.9, 1840.0, 0.1, 0.30000000000000004, 1e16, 1e15, 1e-05, 0.0001, 123456789.123, 2.5e-07,
             1e22, 1e23, 5e-324, 1.7976931348623157e308, 2.0 ** 52 + 0.5, 9007199254740993.0, -0.0, -1.5e-10]
    reprs += [f32(rng.random()) for _ in range(30)] + [rng.uniform(-1e6, 1e6) for _ in range(30)]
    distributions = [[1.0], [0.0, 0.0, 0.0], [0.5, 0.5], [0.2, 0.5, 0.3], [0.4, 0.2, 0.4], [1.0, 0.0, 0.0, 0.0]]
    for _ in range(40):
        raw = [rng.random() ** 3 for _ in range(rng.randint(2, 8))]
        total = sum(raw)
        distributions.append([f32(x / total) for x in raw])
    answers = []
    for p in distributions:
        k = len(p)
        meta = [{"id": "c", "type": "choice", "keys": [f"o{i}" for i in range(k)]},
                {"id": "s", "type": "score", "keys": [str(i) for i in range(k)],
                 "legend": {str(i): f"level {i}" for i in range(k)}}]
        if k == 2:
            meta.append({"id": "n", "type": "noul", "keys": ["false", "true"]})
        answers.append({"p": p, "meta": meta, "answers": to_answers([p] * len(meta), meta),
                        "normalize": _normalize(p), "choice_confidence": choice_confidence(p),
                        "score_confidence": score_confidence(p)})
    doc = {
        "version": 1, "python": sys.version.split()[0],
        "sum": [{"values": v, "sum": sum(v)} for v in sums],
        "round": [{"x": x, "round4": round(x, 4), "round_prob": round_prob(x)} for x in rounds],
        "repr": [{"x": x, "repr": repr(x)} for x in reprs],
        "answers": answers,
    }
    PYTHON_NUMBERS.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{PYTHON_NUMBERS.relative_to(MODULE)}: {len(sums)} sums, {len(rounds)} rounds, "
          f"{len(reprs)} reprs, {len(answers)} answer sets")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kev-work", required=True, type=pathlib.Path, help="the conversion run's kev_work")
    parser.add_argument("--probes", action="store_true", help="also write the AutoTokenizer / kev.api cases")
    args = parser.parse_args()
    kev_work = args.kev_work.resolve()
    demo = tuple(f"demo/fixtures/{record_id}.json" for record_id in EXAMPLES) + (DEMO_ORACLE, DEMO_GRAPH_CPU)
    check_sources(kev_work, (REQUESTS, ORACLE, HIDDEN) + demo + ((KEV_SNAPSHOT + "/tokenizer.json",) if args.probes else ()))
    records = json.loads((kev_work / REQUESTS).read_text(encoding="utf-8"))["records"]
    oracle = json.loads((kev_work / ORACLE).read_text(encoding="utf-8"))
    write_gate_asset(kev_work, records, oracle)
    write_head_fixture(kev_work, oracle)
    write_examples(kev_work)
    if args.probes:
        write_probes(kev_work)
        write_render_cases(kev_work)
        write_python_numbers(kev_work)


if __name__ == "__main__":
    main()

"""Export an HF model to .litertlm but FORCE a simple ChatML chat template.

The runtime applies the LlmMetadata's *structured* prompt_templates (the official
litert-community models use this — simple `<|im_start|>role\n` prefixes). Without
--use_jinja_template, litert-torch's `parse_chat_template` tries to extract that
structured form by applying the tokenizer's chat_template to sample messages — but
for COMPLEX templates (LFM2 namespaces, Qwen/MiniCPM tool+thinking logic) the
extraction fails and it falls back to embedding the raw jinja, which the runtime's
minimal jinja engine (minja) can't render → empty / garbage output on device.

Fix: monkeypatch the tokenizer's chat_template to a minimal ChatML template before
export, so parse_chat_template cleanly extracts the structured prefixes — matching
the official models. Usage:

    python export_simple_template.py <hf_model> <out_dir> <template.jinja> [quant_recipe]
"""

import sys

# --- scipy stub prelude (macOS / clipconv main-HEAD env): scipy 1.15.3's compiled
#     _propack fails to dlopen here, and transformers' D-FINE detection loss (pulled in
#     transitively when litert_torch calls AttentionInterface.register) imports
#     scipy.optimize. Neither is used by LLM conversion — stub the broken leaves so the
#     import chain succeeds; the REAL csgraph / sparse.linalg then load. No-op on a clean
#     scipy (e.g. .venv 1.17). Mirrors scripts/probe_convert.py. ---
import types as _types  # noqa: E402


class _StubLeaf:
  def __getattr__(self, n):
    return lambda *a, **k: None

  def __call__(self, *a, **k):
    return None


def _scipy_healthy():
  # Only stub when scipy is actually broken (clipconv env: scipy 1.15.3's _propack
  # fails to dlopen). On a clean scipy (.venv 1.17) the stub would ITSELF break the
  # scipy.sparse.csgraph / _svdp import chain that externalize_embedder pulls in
  # (the stub _propack lacks slansvd) — so skip stubbing when the real one works.
  try:
    import scipy.sparse.linalg._propack  # noqa: F401
    import scipy.optimize  # noqa: F401
    return True
  except Exception:
    return False


if not _scipy_healthy():
  _pp = _types.ModuleType("scipy.sparse.linalg._propack")
  _pp.__file__ = "<stub:scipy._propack>"
  _pp.__spec__ = None
  for _nm in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_pp, _nm, _StubLeaf())
  sys.modules["scipy.sparse.linalg._propack"] = _pp

  _opt = _types.ModuleType("scipy.optimize")
  _opt.__file__ = "<stub:scipy.optimize>"
  _opt.__spec__ = None
  _opt.linear_sum_assignment = lambda *a, **k: None
  sys.modules["scipy.optimize"] = _opt

import transformers  # noqa: E402

model_id = sys.argv[1]
out_dir = sys.argv[2]
template_path = sys.argv[3]
quant = sys.argv[4] if len(sys.argv) > 4 else "dynamic_wi4_afp32"

SIMPLE_TEMPLATE = open(template_path).read()

# Force every tokenizer loaded during export to use the simple ChatML template.
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained


def _patched_from_pretrained(*args, **kwargs):
  tok = _orig_from_pretrained(*args, **kwargs)
  try:
    tok.chat_template = SIMPLE_TEMPLATE
  except Exception as e:  # pylint: disable=broad-except
    print(f"WARN could not set chat_template: {e}")
  return tok


transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained

# Force a SentencePiece tokenizer in the .litertlm. export_tokenizer defaults to
# saving the HF tokenizer.json (→ HF_Tokenizer_Zlib), which the runtime
# MIS-TOKENIZES (the prompt comes out as garbage → degenerate output). The working
# official litert-community models embed SP_Tokenizer (HF→sentencepiece converted).
import os  # noqa: E402
import dataclasses  # noqa: E402
from litert_torch.generative.export_hf.core import export_lib  # noqa: E402
from litert_torch.generative.tools import (  # noqa: E402
    tokenizer_to_sentencepiece_lib as _tok_spm,
)


def _force_spm_export_tokenizer(source_model_artifacts, export_config, exported):
  tok = source_model_artifacts.tokenizer
  # BPE tokenizers (Qwen/Llama-BPE) expose a `vocab_file` pointing at vocab.json.
  # tokenizer_to_sentencepiece.convert() then tries to parse that JSON as a
  # sentencepiece ModelProto → "Wire format was corrupt". Only SP-native tokenizers
  # (Gemma/Llama-SP, vocab_file = *.model/*.spiece) belong on that fast path; for the
  # rest, clear vocab_file so convert() builds a real SP model from vocab+merges (the
  # path the working int8 export took when no vocab.json happened to be cached).
  vf = getattr(tok, "vocab_file", None)
  if vf and not str(vf).endswith((".model", ".spiece", ".spm")):
    tok.vocab_file = None
  spm = _tok_spm.convert(tok)
  path = os.path.join(export_config.work_dir, "tokenizer.spiece")
  with open(path, "wb") as f:
    f.write(spm)
  print("FORCED sentencepiece tokenizer")
  return dataclasses.replace(exported, tokenizer_model_path=path)


if os.environ.get("FORCE_SPM"):
  export_lib.export_tokenizer = _force_spm_export_tokenizer

# Register custom mixed-int4 recipes by NAME (export_lib does
# recipe_lib.__dict__[name]()). int4 default + keep the vocab embedding/lm_head
# (EMBEDDING_LOOKUP — the int4-killer for small models) at int8.
import copy  # noqa: E402
import ai_edge_quantizer.recipe as _aqr  # noqa: E402

_I4 = _aqr.dynamic_wi4_afp32()[0]
_I8 = copy.deepcopy(_I4)
_I8["op_config"]["weight_tensor_config"]["num_bits"] = 8


def _mk(ops_int8):
  rules = [_I4]
  for op in ops_int8:
    rr = copy.deepcopy(_I8)
    rr["operation"] = op
    rules.append(rr)
  return rules


_aqr.MIXED4 = lambda: _mk(["EMBEDDING_LOOKUP"])
_aqr.MIXED4B = lambda: _mk(["EMBEDDING_LOOKUP", "FULLY_CONNECTED"])

# Better int4: replace naive min-max with OCTAV (optimal-clipping, data-free) /
# MSE for the int4 weights, keeping the int8 embedding. Reduces PTQ degradation
# (e.g. GSM8K) without calibration data — the fix for "int4 is measurably worse than bf16".
_O4 = copy.deepcopy(_I4)
_O4["algorithm_key"] = _aqr.AlgorithmName.OCTAV
_M4 = copy.deepcopy(_I4)
_M4["algorithm_key"] = _aqr.AlgorithmName.MSE


def _mk_alg(int4_rule, ops_int8):
  rules = [int4_rule]
  for op in ops_int8:
    rr = copy.deepcopy(_I8)
    rr["operation"] = op
    rules.append(rr)
  return rules


_aqr.OCTAV4 = lambda: _mk_alg(_O4, ["EMBEDDING_LOOKUP"])
_aqr.MSE4 = lambda: _mk_alg(_M4, ["EMBEDDING_LOOKUP"])

# BLOCKWISE int4 (block size 32) — the granularity the official litert-community
# models use. `dynamic_wi4_afp32` defaults to CHANNELWISE int4, which catastrophically
# collapses small models (Qwen3-0.6B: 0% GSM8K, degenerate looping). Blockwise int4
# matches the official conversion (≈42% on 0.6B). Keep the vocab embedding at int8.
_B4 = copy.deepcopy(_I4)
_B4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
# Blockwise int4 + OCTAV optimal-clipping = best data-free int4.
_BO4 = copy.deepcopy(_O4)
_BO4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
_aqr.BMIX4 = lambda: _mk_alg(_B4, ["EMBEDDING_LOOKUP"])
_aqr.BMIX4B = lambda: _mk_alg(_B4, ["EMBEDDING_LOOKUP", "FULLY_CONNECTED"])
_aqr.BOCTAV4 = lambda: _mk_alg(_BO4, ["EMBEDDING_LOOKUP"])

# Blockwise-128 int4 — coarser blocks = 1/4 the scales = lighter dequant =
# faster GPU decode (the granularity the official Gemma block128 bundles use),
# at a small quality cost vs block32. The on-device speed/quality knob.
_B4_128 = copy.deepcopy(_I4)
_B4_128["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_128"
_aqr.BMIX4_128 = lambda: _mk_alg(_B4_128, ["EMBEDDING_LOOKUP"])

from litert_torch.generative.export_hf.export import export  # noqa: E402

# use_jinja_template defaults to True (→ embeds raw jinja, which the runtime's
# minja can't render → broken prompt). Force False so parse_chat_template extracts
# the STRUCTURED prompt_templates (simple ChatML prefixes) the runtime applies —
# matching the official litert-community models.
# "NONE" → no quantization (fp32 reference, for logit-parity isolation of the converter).
quant_recipe = None if quant.upper() in ("NONE", "FP32") else quant

export(
    model=model_id,
    output_dir=out_dir,
    prefill_lengths=[int(os.environ.get("PREFILL", "128"))],
    cache_length=int(os.environ.get("CACHE", "1024")),
    quantization_recipe=quant_recipe,
    use_jinja_template=False,
    experimental_use_mixed_precision=bool(os.environ.get("MIXED")),
    # EXTERNALIZE_EMBEDDER=1 splits the (tied) embedding into its own .litertlm
    # section — the generic equivalent of Gemma's PLE embedding-mmap. Keeps the main
    # TFLiteModel weights section under the iOS ~2GiB single-section mmap limit so
    # big (28-layer 3B) models load on iPhone. No effect on weights/parity.
    externalize_embedder=bool(os.environ.get("EXTERNALIZE_EMBEDDER")),
    trust_remote_code=True,
)
print("EXPORT_DONE")

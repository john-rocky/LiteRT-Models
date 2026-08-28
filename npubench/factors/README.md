# NPU factor experiments (S26 / SM8850)

The scripts and raw logs behind [`NPU_OP_FACTOR_REPORT.md`](../../NPU_OP_FACTOR_REPORT.md)
— why the Hexagon NPU loses on 9 of the 50 zoo models, established by
single-variable experiments rather than correlation.

| file | role |
|---|---|
| `opscan.py` | static op-composition scan of a `.tflite` (op histogram, act×act BMM shapes, FLOP estimate, DEQUANTIZE count) |
| `fold_deq.py` | folds fp16-weight DEQUANTIZE into fp32 consts — the identity transform that refuted the dtype hypothesis |
| `make_fp16.py` | the inverse: fp16-izes large consts + inserts DEQUANTIZE (synthetic dtype twin) |
| `probe_gen.py` | matched-compute transformer probes, T ∈ {512…2048} (sequence-shape curve) |
| `probe_style.py` | GELU-flavor / LayerNorm-implementation probes, incl. tanh-via-sigmoid decomposition |
| `build_dinov2_ab.py` | the flip experiment: real DINOv2-S with tanh / sigmoid / builtin-GELU, one variable |
| `build_tipsv2_erf.py` | second real-model flip: TIPSv2-B14-DPT with builtin GELU |
| `probe_batch2.py`–`probe_batch6.py` | later probe batches: rank-4 attention, conv rewrites, LLM prefill-T sweep + RMSNorm flavors, decode-GEMV dtype, fusion breakers + op-granularity curve (docstring in each) |
| `make_int8.py` | int8/int4 + DEQUANTIZE weight-storage twin (companion to `make_fp16.py`) |
| `cut_prefix.py` | truncate a `.tflite` to its first K ops — the bisect tool (memorize) |
| `build_whisper_pad.py` / `build_whisper_ctrl.py` | whisper-tiny encoder pad-flip pair (T=1536 vs 1500, bit-exact; the pad build documents two exactness traps: conv-tail mask, stored-vs-formula pos table) |
| `build_phase9_probes.sh` | Mac-side builder for every phase-9 probe (~3 GB into a target dir) |
| `phase9_chain.sh` | device driver for the phase-9 arms: push → compile pass → thermal-gated measure |
| `gated_measure.sh` | thermal-gated measurement driver (waits for status NONE per arm; screen off) |
| `phase*.log` | raw benchmark rows (npubench `#sweep`, N=50 medians, thermal in every row) |
| `opscan_all.json` | scan results for 28 of the 50 sweep models |

Probe caveat: the probe graphs carry a rank-5 tensor at the qkv permute, so
ML Drift (GPU) refuses them — GPU sensitivity was measured on the real-model
pairs instead. Python env: `~/venvs/ltconv040dev` (litert-torch 0.9.3,
ai-edge-litert 2.1.6).

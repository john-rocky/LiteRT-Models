# Building the Dia2-1B LiteRT graphs + assets

The app loads four `.tflite` graphs and six baked fp16 tables from the device `filesDir`. They are
far too big to bundle in the APK (~4.1 GB), so they are built here and pushed with
`install_to_device.sh`. Only the tokenizer files and the 13 kB voice prompt live in `assets/`.

## Prerequisites

```bash
pip install torch transformers ai-edge-litert litert-torch numpy
huggingface-cli download nari-labs/Dia2-1B --local-dir Dia2-1B
```

`build_dia2_mimi_decode.py` additionally needs the Mimi conversion helpers (`build_mimi.py`, which
bakes the causal convolutions and re-authors the decoder transformer). Point `MIMI_WORK` at the
directory that holds it.

## Build

```bash
export DIA2_OUT=out                     # where the graphs and tables are written
python build_dia2_temporal.py           # temporal transformer, packed KV-cache step graph
python build_dia2_depformer.py          # 3 weight-scheduled depformer step graphs
MIMI_WORK=/path/to/mimi python build_dia2_mimi_decode.py   # one-shot 256-frame decode
python export_dia2_assets.py            # baked fp16 tables + dia2_constants.json
python bake_prefix.py                   # voice prompt -> ../app/src/main/assets/
```

Each build script validates itself against the reference PyTorch model and prints a correlation.

| Script | Produces | Self-check |
|---|---|---|
| `build_dia2_temporal.py` | `dia2_temporal_fp32.tflite` (3.0 GB) | hidden / cb0 correlation vs `forward_step` |
| `build_dia2_depformer.py` | `dia2_depformer_wi{0,1,2}_fp32.tflite` | per-stage logit correlation |
| `build_dia2_mimi_decode.py` | `dia2_mimi_decode_t256.tflite` | shape + decode parity |
| `export_dia2_assets.py` | `*.f16` tables, `dia2_constants.json` | — |
| `bake_prefix.py` | `dia2_prefix.json` (13 kB) | prints frames / entries / new-word steps |

`dia2_mimi_dequant.tflite` (the RVQ decode step) is produced alongside the Mimi decoder.

## Install

```bash
./install_to_device.sh out/
```

The first app launch is expected to fail with "Load failed" until this has been run.

## Why a baked voice prompt?

Dia2 given no voice prefix **samples the speaker identity**, so the voice changes on every run.
Building a real prefix needs Whisper word timings and a Mimi *encoder*; both are host-only.
`bake_prefix.py` therefore precomputes the prompt (aligned Mimi codes, `new_word_steps`, prefix word
entries) into a small JSON, and the device only replays the warm-up — no Mimi encoder ships.

Word timings come from Dia2's own `GenerationResult.timestamps`, so Whisper is not required.

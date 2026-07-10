# Dia2-1B dialogue TTS on LiteRT CompiledModel (CPU)

Two-speaker **dialogue** text-to-speech running on-device with LiteRT `CompiledModel`. Dia2 is a
Moshi-style **RQ-Transformer**: once per 12.5 Hz frame a 30-layer *temporal* transformer emits a
word-timing action plus Mimi codebook 0, then a 3-layer *depformer* autoregressively fills the
remaining 31 codebooks for that same frame. The 32 codebooks are decoded to 24 kHz audio by Mimi.

This is the first dialogue TTS and the first RQ-Transformer in this zoo.

| | |
|---|---|
| Model | [Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B) (Nari Labs, Apache-2.0) |
| Backbone | 30-layer temporal transformer (GQA 16Q/8KV, RoPE) + 3-layer depformer |
| Codec | [kyutai/mimi](https://huggingface.co/kyutai/mimi), 32 quantizers |
| Frame rate | 12.5 Hz (1920 samples/frame) |
| Sample rate | 24 kHz mono |
| App | `com.dia2` — type a `[S1]`/`[S2]` script, generate, play |

## Device placement

Everything runs on **CPU (fp32)** as filed, because this sample pins LiteRT **2.1.3**. Two separate
reasons, both re-examined on 2026-07-10 — and the first one turned out not to be a wall at all:

* **Not the delegate.** An earlier version of this file said the Mali ML Drift delegate rejects the
  language models' KV-step `FULLY_CONNECTED` weight shapes. That rejection is real on LiteRT 2.1.3
  and **fixed in 2.1.5**. The depformer's own failure was in *our* graph: a rank-5 reshape inside the
  fused-QKV authoring (the GPU's maximum rank is 4). Slicing the last dimension into thirds instead
  gives **237/237 nodes delegated at 4–7 ms/stage**, and expanding the attention mask host-side from
  `[1,1,1,D]` to `[1,NH,1,D]` (the known BMM + broadcast-`ADD` bug) brings it to **corr 1.000000**
  against the desktop CPU reference. The depformer is GPU-ready; the 3.0 GB temporal graph has not
  been evaluated on GPU yet.
* **fp16 really does collapse these deep stacks** on ARM XNNPACK, so the CPU graphs ship as fp32.

| Graph | Shape | Notes |
|---|---|---|
| `dia2_temporal_fp32` | emb[1,1,1024] + RoPE + packed KV → hidden, action[2], cb0[2050] | 3.0 GB |
| `dia2_depformer_wi{0,1,2}` | one per weight set; the 31 stages share them on a schedule | 164 MB each |
| `dia2_mimi_dequant` | codes[1,32,1] → latent[1,512,1] | RVQ decode |
| `dia2_mimi_decode_t256` | latent[1,512,256] → audio[1,1,491520] | one-shot window |

The KV caches, RoPE, the embedding sums, the depformer's input/output projections and all sampling
live in Kotlin (`Dia2Synthesizer.kt`); the graphs are pure step functions.

## Three things that are easy to get wrong

**1. Both text streams carry real word tokens.** Channel 0 and channel 1 are *not* new-word/pad
markers. On a new word the main stream emits the word's first text token while the second stream
emits `NEW_WORD`; during the padding frames that follow, the main stream drains the rest of the word
and the second stream drains a two-word lookahead. Feeding markers instead produces fluent,
confident, completely wrong speech.

**2. The delay pattern must be undone before decoding.** Codebook `cb` lags the aligned timeline by
`AUDIO_DELAYS[cb]` frames (16 for cb0, 18 for the rest). Codes are stored at `audioCodes[cb][t+1]`,
so `aligned[cb][τ] = audioCodes[cb][delay[cb] + τ]` and the output is `maxDelay` frames shorter.
Skipping this yields muffled, unintelligible audio.

**3. Mimi decode has to be a single pass.** The decode path is upsample → **causal** decoder
transformer → SEANet, so its receptive field is unbounded. Decoding disjoint windows starts each one
with no history and costs ~13% relative error (corr 0.991 against a full-sequence torch decode). The
graph therefore spans 256 frames and the unused tail is left zeroed; causality makes that exact for
every real frame — **corr 0.999999**.

## The speaker is sampled — so the voice prompt is baked in

Dia2 given no voice prefix **samples the speaker identity**, so the voice changes on every run
(median F0 wanders over a ~120 Hz range). Classifier-free guidance does *not* fix this; it only
steadies the output level. The model's own remedy is a **voice prefix**, which normally needs
Whisper word timings and a Mimi *encoder* — both host-only.

`scripts/bake_prefix.py` therefore precomputes the prompt offline and ships it as a 13 kB JSON
(aligned Mimi codes, `new_word_steps`, prefix word entries). On device only the warm-up runs: the
temporal transformer replays the prompt to prime both KV caches — no Mimi encoder, no sampling, no
depformer. Measured on device, the generated speakers track their prompts (S1 214 Hz / S2 114 Hz,
against prompts of 247 Hz / 88 Hz; without a prefix S2 never drops below 214 Hz).

Classifier-free guidance (`cfg_scale = 2.0`, Dia2's default) is implemented faithfully: the guided
logits `uncond + scale·(cond − uncond)` only *select* the top-k candidate set, while the draw is a
temperature softmax over the **conditional** logits restricted to that set. That needs a second,
unconditional branch (text forced to `zero`/`pad`, same audio codes, its own KV cache) — hence two
temporal runs and two depformer runs per frame.

## Build and run

```bash
cd dia2/
python scripts/build_dia2_temporal.py
python scripts/build_dia2_depformer.py
python scripts/build_dia2_mimi_decode.py
python scripts/export_dia2_assets.py
python scripts/bake_prefix.py            # writes app/src/main/assets/dia2_prefix.json
./scripts/install_to_device.sh out/      # ~4.1 GB via adb
./gradlew :app:installDebug
```

## Performance and memory

On a Pixel 8a a 4-second utterance takes ~190 s: 71 warm-up frames (temporal only, ×2 guidance
branches) plus ~67 generated frames, each running 2 temporal steps and 2×31 depformer stages.

The process peaks at **~4.6 GB RSS** (3.0 GB fp32 temporal graph + 3×164 MB depformer + ~400 MB of
baked fp32 tables and KV caches) and settles around 3.2 GB. On an 8 GB phone that leaves little head
room — close other apps, or the kernel's low-memory killer may take the process mid-run.

## Validation

Every ported component was checked against the reference implementation on host before it reached
the device:

| Component | Check | Result |
|---|---|---|
| StateMachine (multiplex + lookahead) | vs recorded reference frame stream | 0/60 mismatches |
| Tokenizer + `parse_script` | vs reference entries | 10/10 exact |
| Depformer 31-stage KV glue | vs torch depformer | corr 1.0000, argmax 31/31 |
| Mimi decode | vs torch full-sequence decode | corr 0.999999 |
| Voice-prefix warm-up | vs reference warm-up + generation | 0 mismatches (71 + 63 frames) |

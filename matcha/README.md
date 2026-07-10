# Matcha-TTS on LiteRT CompiledModel GPU

On-device English text-to-speech. Type text, synthesize it on the **GPU**, play it back —
no network, no espeak. This is the **FFT-free** TTS lane: Matcha-TTS pairs a conditional
flow-matching (CFM) acoustic model with a **HiFi-GAN time-domain vocoder**, so there is **no
FFT/iSTFT anywhere** in the synthesis path and the whole thing rides the LiteRT `CompiledModel`
GPU (ML Drift) delegate. (Spectral vocoders — Kokoro/iSTFTNet/Vocos — are blocked on the missing
ML Drift FFT kernel; Matcha is the one that passes.)

| | |
|---|---|
| Model | Matcha-TTS (LJSpeech) + HiFi-GAN T2 v1 vocoder |
| Source | https://github.com/shivammehta25/Matcha-TTS (MIT) |
| Sample rate | 22.05 kHz, hop 256 |
| App | `com.matcha` — text in, audio out |

## Pipeline

Three heavy graphs on the **GPU**, the sequential/stochastic glue on the **host (CPU)**:

```
text --G2P(CPU)--> phoneme ids
     --host: embed + intersperse + pad-->        text_encoder(GPU)  -> mu, logw
     --host: durations + length-regulator-->     mu_y [1,80,T]
     --host: Euler ODE loop (N steps)-->          decoder(GPU) x N   (x_t, mu_y, sin_emb(t), mask) -> v
     --host: denormalize-->                       vocoder(GPU)       -> waveform [T*256]
```

- **`matcha_textenc_fp16.tflite`** (14.8 MB) — RoPE text encoder. in: `emb[1,256,192]`, `tmask[1,1,256]`; out: `mu[1,80,256]`, `logw[1,1,256]`.
- **`matcha_decoder_fp16.tflite`** (22.5 MB) — CFM flow-matching decoder, run once per ODE step. in: `x[1,80,512]`, `mu[1,80,512]`, `t_sin[1,160]`, `mask[1,1,512]`; out: `v[1,80,512]`. **Runs on the CompiledModel CPU delegate** — see "Decoder on CPU" below.
- **`matcha_vocoder_fp16.tflite`** (29.0 MB) — HiFi-GAN v1 generator (mel→wav). in: `mel[1,80,512]`; out: `wav[1,1,131072]`.
- **`dp_g2p_matcha_fp16.tflite`** (25.8 MB, **CPU** delegate) — DeepPhonemizer OOV fallback for the G2P.

Fixed shapes (`MAX_TEXT=256` phonemes, `MAX_MEL=512` frames ≈ 5.9 s). A **runtime float mask**
makes the padded positions a no-op (additive attention bias), so one compiled graph handles any
length without recompiling. `MatchaSynthesizer.kt` does the host orchestration; `MatchaG2P.kt`
the text→phoneme conversion.

## G2P (clean, espeak-free)

espeak is GPL, so it can't ship. Matcha-LJSpeech is trained on **espeak en-us IPA**, so the G2P
must reproduce that inventory. We use the hybrid from kokoro:

1. **275k-entry espeak-IPA dictionary** (`g2p_dict.txt.gz`, primary) — from OpenPhonemizer
   (Clear BSD), covers ~all common words with the correct espeak-IPA pronunciation.
2. **DeepPhonemizer** (OpenPhonemizer `best_model.pt`, MIT) for out-of-dictionary words,
   converted to a fixed `[1,96]` LiteRT graph (CPU delegate). Output IPA maps 1:1 onto the
   keithito 178-symbol set Matcha uses.

## GPU re-authoring (what makes it ML-Drift-clean)

All in `scripts/build_matcha.py` (random-weight op-checks + real-weight conversion, tflite-vs-torch
corr 1.000000 per graph, end-to-end waveform corr ≥0.99):

- `GroupNorm` → manual 4D mean/var (kills `GATHER_ND`)
- `Mish` → SELECT-free fp16-safe softplus `x·tanh(relu(x)+log1p(exp(-|x|)))`
- `ConvTranspose1d` → **ZeroStuffConvT1d** (DAC; kills `TRANSPOSE_CONV`, rejected by Mali)
- diffusers `Attention` → manual additive-masked attention; the **mask is a runtime input**
  (decoder adds the raw 0/1 mask = `AttnProcessor2_0`'s soft bias; text-enc adds `(mask-1)·1e4`)
- half-res mask `mask[:,:,::2]` → reshape-decimate (a step-2 slice lowers to `GATHER_ND`)
- `SinusoidalPosEmb` time embedding → host-side (weight-free sin/cos), `time_mlp` stays on GPU
- fp16 via AI Edge Quantizer `FLOAT_CASTING`

## Build & run

```bash
cd matcha/
./gradlew :app:installDebug
# the four tflites are pushed to the app's filesDir (too big to bundle):
./scripts/install_to_device.sh <dir-with-the-tflites>   # from HF or built by the scripts
```

The first launch fails with "model not found" until the install script has run. Get the tflites
from Hugging Face (`litert-community/Matcha-TTS`) or build them:

```bash
pip install --no-deps matcha-tts diffusers einops conformer deep-phonemizer
python scripts/convert_final.py 512        # the 3 GPU graphs (fp16)
python scripts/convert_g2p_matcha.py       # the G2P graph
```

## Decoder on CPU (ML Drift fusion bug)

The CFM decoder runs on the **CompiledModel CPU** delegate, not GPU. On the Pixel 8a Mali ML Drift
delegate the decoder's diffusers transformer blocks are **mis-fused at large activation magnitude**:
the up-path transformer (input |x|~60) collapses its residual (device output ±0.7 vs CPU ±60, corr
**0.006**), giving a NaN/garbled mel. Critically, the **same transformer block converted as a
standalone graph computes correctly on the GPU** (corr **0.984**) — so it is a *graph-fusion/scheduling*
bug, not a bad op (every individual op — GroupNorm-4D, Mish, SnakeBeta, ZeroStuffConvT1d, the manual
masked attention — verified correct on Mali via on-device tap dumps). It is **not** the row-147 fp16
variance-overflow class (the `SafeLayerNorm` scale-before-square fix does not help) nor C29 (the first
all-NaN mel *was* the multi-axis `mean((2,3))` → fixed with split `mean(3).mean(2)`).

The clean fix is host-side: the decoder on CPU is exact and keeps the whole pipeline **realtime
(RTF ~0.8 on Pixel 8a)** since the GPU vocoder dominates wall time anyway. text encoder + vocoder
stay on the GPU. (Alternative, not shipped: split the decoder into per-transformer CompiledModel
sub-graphs to dodge the fusion — feasible since standalone = corr 0.984, but more complex and not
numerically exact.)

## Notes

- RTF lever is the **vocoder** (runs once at `MAX_MEL`, dominates wall time); the ODE step count
  is cheap (N=2→10 barely changes latency), so keep N=10 for quality. Bucket the vocoder by length
  to cut padding waste if needed.
- `MatchaG2P` normalizes text host-side before phonemizing: **ALL-CAPS acronyms are spelled
  letter-by-letter** ("GPU" → "gee pee you", via an espeak letter-name IPA table) and **numbers are
  read as words** ("4090" → "four thousand ninety"; decimals → "point").

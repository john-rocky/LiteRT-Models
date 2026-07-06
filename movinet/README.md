# MoViNet-A0 — On-device streaming video action recognition (LiteRT GPU)

Real-time **video action recognition** running fully on the **LiteRT `CompiledModel`
GPU** delegate. This is the first *video-input* model in this zoo: instead of
classifying a single image, it recognises what *action* is happening across a
stream of camera frames, one frame at a time, with constant memory.

- **Model:** [MoViNet-A0](https://arxiv.org/abs/2103.11511) streaming variant
  (Google Research, Apache-2.0), trained on **Kinetics-600** (600 action classes).
- **Weights:** ported PyTorch checkpoint from
  [Atze00/MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch).
- **Input:** one RGB frame `[1, 3, 172, 172]` (NCHW, 0..1) per step.
- **Output:** `[1, 600]` logits over Kinetics-600, plus updated recurrent state.
- **Size:** 15 MB `.tflite`, ~3.75 M parameters.

## How the streaming works

MoViNet is a causal 3D CNN: each temporal convolution and each global-average-pool
keeps a small buffer of the recent past so the network can be fed **one frame at a
time** (constant memory, real-time), and its prediction sharpens as more frames of
the same action arrive.

The stock streaming graph carries that history in **5D** state tensors
`[1, T, H, W, C]`, which the GPU delegate cannot compile (all tensors must be ≤4D).
So the model here is re-authored as a **single-frame, 4D-only functional forward**
with every recurrent buffer threaded explicitly through the graph I/O
(**47 inputs / 28 outputs**):

| I/O slot        | count | shape           | meaning                              |
|-----------------|-------|-----------------|--------------------------------------|
| `input[0]`      | 1     | `[1,3,172,172]` | current RGB frame (NCHW, 0..1)       |
| `input[1..28]`  | 28    | `[1,C,H,W]`     | temporal-conv stream buffers (11 convs, kernel 3/5, oldest first) |
| `input[29..44]` | 16    | `[1,C,1,1]`     | streaming avg-pool running sums (15 SE + 1 head)    |
| `input[45]`     | 1     | `[1,1,1,1]`     | `inv_count` = 1 / current frame number             |
| `input[46]`     | 1     | `[1,1,1,1]`     | constant `1.0` (output decoupler, see below)        |
| `output[0]`     | 1     | `[1,600]`       | Kinetics-600 logits                  |
| `output[1..11]` | 11    | `[1,C,H,W]`     | current per-temporal-conv frame      |
| `output[12..27]`| 16    | `[1,C,1,1]`     | fresh per-frame spatial means        |

The **stream-buffer shift register and the pool running-sum accumulation are both
done host-side** (in Kotlin): the graph consumes the recurrent state but only
emits fresh tensors. Each frame the app runs once, then (a) shifts each conv's
stream buffer — drop oldest, append the emitted current frame — and (b)
accumulates `running_sum += emitted_mean`, feeding both back as inputs. Temporal
depthwise convs become a per-channel weighted sum of the buffered frames;
streaming pools become `avg = (running_sum + mean) * inv_count`. Every op lowers
to a GPU-supported primitive — **0 tensors of rank > 4, 0 banned ops, 0
composites** — and the converted graph matches the original PyTorch model
bit-for-bit (corr 0.99999999999, top-5 identical). On the *jumping-jacks*
reference clip, device GPU locks onto **"jumping jacks"** within a few frames.

### Why host-side state (Mali GPU delegate quirks)

Keeping the recurrent state in-graph tripped three silent Mali `CompiledModel`
bugs (all invisible on desktop CPU — device-verify!): an input passed *through* to
an output loses its compute-side use; a `state + tensor` output reads as zero; and
a conv-output tensor that is both consumed and emitted has its **emitted copy
corrupted (~2.5×)**, which then blows up in fp16 over frames. The fixes: emit only
fresh tensors and do all state plumbing host-side, and decouple each emitted
stream frame from its compute use with a multiply against the runtime `1.0` input.

## Build & run

```bash
cd movinet/
./gradlew :app:installDebug
```

The 15 MB `movinet_a0_stream.tflite` is bundled in `app/src/main/assets/` (build it
with `scripts/build_movinet.py`, see below — the file is not committed). Point the
back camera at someone performing an action; the top-5 Kinetics-600 predictions are
shown as bars at the bottom. Tap the screen to restart the classification window.

Because the pooling states are cumulative, the app resets them every 64 frames to
keep predictions responsive to the current action (and to keep the running sums
small — friendly to fp16 GPUs).

## Regenerate the model

```bash
pip install torch litert-torch fvcore einops
git clone https://github.com/Atze00/MoViNet-pytorch.git
MOVINET_PYTORCH=./MoViNet-pytorch python scripts/build_movinet.py
cp movinet_a0_stream.tflite app/src/main/assets/
```

`scripts/stream_model.py` holds the 4D re-authoring (`MoViNetA0Stream`);
`scripts/build_movinet.py` verifies parity and converts.

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- The tf-`same` residual average pooling is reformulated as
  `count_include_pad=True` + a constant boundary-correction mask so it lowers to
  `AVERAGE_POOL_2D + MUL` (no GPU-incompatible composite).

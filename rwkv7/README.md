# RWKV-7 World 0.1B — on-device text generation (LiteRT GPU)

The first autoregressive language model running its **full forward pass on the
LiteRT `CompiledModel` GPU delegate** (RNN mode, host-side state). RWKV-7 is an
RNN: one token per step with a fixed-size recurrent state, so the whole model
fits a single static GPU graph — no KV cache growth, no dynamic shapes, and no
CPU fallback for any op.

- Model: [RWKV-x070-World-0.1B-v2.8](https://huggingface.co/BlinkDL/rwkv-7-world)
  (Apache-2.0), L12 / D768 / 12 heads, vocab 65536.
- Graph: `x_emb[1,768], att_shift[12,768], ffn_shift[12,768], wkv[144,64,64]`
  → `logits[1,65536]` + the three updated states. 1863/1863 ops on GPU,
  1 partition, ~18 ms/token (fp16, Pixel 8a).
- Host side: token embedding row lookup (fp16 table, memory-mapped), greedy
  argmax, and state recycling between steps.
- Verified: 30-token greedy generation on device tracks desktop fp32 —
  28/30 tokens identical, the 2 divergences are fp32 near-ties (logit gap
  ≤ 0.04) with rank-2 picks; prefill logits corr 0.99995.

## Build & run

1. Produce the model artifacts (or download them from the HF card):

   ```bash
   cd scripts/
   # rwkv7_0.1b.pth + rwkv_vocab_v20230424.txt next to the script
   python build_rwkv7_step.py all   # parity → convert → fp16 → assets → validate
   ```

2. Stage the two large files (never committed / bundled):

   ```bash
   adb push scripts/rwkv7_step_fp16.tflite /data/local/tmp/
   adb push scripts/rwkv7_emb_fp16.bin    /data/local/tmp/
   ```

3. Build + install the app from Android Studio (open this directory), launch it
   once (it reports the missing files), then:

   ```bash
   ./scripts/install_to_device.sh
   ```

4. Relaunch. Type a prompt (or enable the chat template) and Generate.

## Files

| File | Role |
|------|------|
| `app/src/main/java/com/rwkv7/Rwkv7Generator.kt` | CompiledModel step loop, host-side state + embedding lookup |
| `app/src/main/java/com/rwkv7/RwkvTokenizer.kt` | RWKV World trie tokenizer port (fixture-tested vs the Python reference) |
| `app/src/main/java/com/rwkv7/MainActivity.kt` | Streaming completion/chat UI, greedy decoding |
| `scripts/build_rwkv7_step.py` | parity / convert / fp16 / assets / validate pipeline |
| `scripts/install_to_device.sh` | moves staged model files into the app's filesDir |

## GPU re-authorings (all exact)

- wkv7 recurrence at T=1 → plain 4D matmul/elementwise.
- `GroupNorm(heads)` → manual per-head mean/var.
- `F.normalize` → `x * rsqrt(sum(x²) + eps)`.
- `softplus` → `relu(z) + log1p(exp(-|z|))` (stock lowering emits
  GREATER+SELECT, rejected by the GPU delegate).
- Embedding lookup host-side (GATHER is GPU-banned); `ln0` stays in-graph.
- torch.export example inputs must be `.clone()`d (views trip the converter).

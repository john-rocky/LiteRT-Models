# Text Reranking — on-device RAG reranker (Qwen3-Reranker-0.6B, fully GPU)

On-device cross-encoder reranking for RAG: given a query and candidate documents, score each by
relevance and reorder them — running the 2025 SOTA
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (Apache-2.0)
**fully on LiteRT `CompiledModel` GPU** (ML Drift).

This is the reranking half of the on-device RAG stack — the retrieval half is
[`text-embedding/`](../text-embedding) (Qwen3-Embedding-0.6B). Embed → retrieve top-k → **rerank**.

![On-device RAG reranking on a Pixel 8a](hero.png)

Like the embedder, it is a **single forward pass** (no generation, no KV cache), so it is a plain
`.tflite` on the standard GPU path. It shares the exact same graph re-authoring — only the head
differs: after the final norm, a baked **2-logit head** (the tied-embedding rows for the tokens
`"no"`=2152 / `"yes"`=9693) produces `[1,L,2]`; the host takes the softmax at the last token →
**P(yes) = relevance**.

## Model

| | |
|---|---|
| Base | Qwen3-Reranker-0.6B (Qwen3 decoder, 28 layers, hidden 1024, GQA 16/8, RoPE, SwiGLU) |
| Task | pointwise reranking — P(yes) that a document answers the query |
| Precision | fp16, 882 MB `.tflite` (+ 310 MB embedding table) |
| Device | Pixel 8a / Tensor G3: **P(yes) parity ref 0.9995 / dev 0.9994**, all nodes on GPU |
| License | Apache-2.0 |

## How it runs

1. Build the Qwen3-Reranker prompt: `PREFIX` + `<Instruct>:… <Query>:… <Document>:…` + `SUFFIX`
   (the `PREFIX`/`SUFFIX` special tokens are pre-tokenized; the content is byte-level BPE on device).
2. **Embed lookup** — token ids → `inputs_embeds [1,256,1024]` from the bundled table (GATHER is
   GPU-banned, so host-side).
3. **CompiledModel GPU** runs the 28-layer transformer + the 2-logit head → `[1,256,2]`.
4. **Softmax** over `[no,yes]` at the last real token → `P(yes)`; sort documents by it.

The GPU-compatible re-authoring (host-embed, GQA cat-repeat, max-normalized SafeRMS for the deep-stack
fp16 overflow, baked RoPE / causal mask, baked yes/no head) is reproduced in [`conversion/`](conversion/).

## Model files

Staged into the app's private `filesDir` by `scripts/install_to_device.sh` (large-model pattern):
`qwen3rerank_gpu_fp16.tflite` (882 MB) + `embeddings_fp16.bin` (310 MB). The tokenizer
(`vocab.json`, `merges.txt`) and demo `docs.txt` are bundled in `assets/`. `.tflite`/`.bin` are never
committed; `conversion/` reproduces them.

## Build

```bash
cd text-reranking/
./gradlew :app:installDebug
scripts/install_to_device.sh
```

`minSdk 26`, `arm64-v8a`, LiteRT `CompiledModel` GPU.

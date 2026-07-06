# Text Embedding — on-device semantic search / RAG (Qwen3-Embedding-0.6B, fully GPU)

On-device text embeddings for semantic search and RAG retrieval, running the 2025 SOTA
[`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache-2.0)
**fully on LiteRT `CompiledModel` GPU** (ML Drift). Given a query and a set of documents, the app
embeds each with the model and ranks documents by cosine similarity — the retrieval half of a RAG
pipeline, entirely on-device.

This is the embedding counterpart to [`text-generation/`](../text-generation): that one runs
generative LLMs on the LiteRT-LM runtime (`.litertlm`); this one runs a single-forward embedding
model on the plain `CompiledModel` GPU path (`.tflite`), because last-token pooling needs no
generation loop and no KV cache.

![On-device semantic search on a Pixel 8a](hero.png)

## Model

| | |
|---|---|
| Base | Qwen3-Embedding-0.6B (Qwen3 decoder, 28 layers, hidden 1024, GQA 16/8, RoPE, SwiGLU) |
| Task | text embedding (last-token pooling), 1024-d, Matryoshka 32–1024 |
| Precision | fp16, 881 MB `.tflite` |
| Device | Pixel 8a / Tensor G3: GPU compile OK, **266 ms** / embedding, parity 0.9997 vs HF |
| License | Apache-2.0 |

## How it runs

1. **Tokenize** the text (Qwen byte-level BPE) on the host.
2. **Embed lookup** — token ids → `inputs_embeds [1,128,1024]` from the bundled embedding table
   (the model ties input/output embeddings). Lookup is a GATHER (GPU-banned), so it is done host-side,
   exactly like mel/log-mel preprocessing in the audio samples.
3. **CompiledModel GPU** runs the 28-layer transformer → `hidden_states [1,128,1024]`.
4. **Pool + normalize** — take the last real token, L2-normalize, optional Matryoshka truncation.
5. **Cosine similarity** ranks the document embeddings against the query embedding.

The GPU-compatible re-authoring (host-embed input, GQA cat-repeat, max-normalized SafeRMS for the
deep-stack fp16 overflow, RoPE/causal-mask constants) is documented and reproduced in
[`conversion/`](conversion/).

## Model files: how they get to the device

Both large assets are staged to the app's private `filesDir` via `scripts/install_to_device.sh`
(the large-model pattern used by `voiceassistant/`, `kokoro/`, etc.), then loaded with the
file-path `CompiledModel.create(path, options, null)` overload:

- `qwen3emb_gpu_fp16.tflite` — 881 MB transformer graph
- `embeddings.bin` — token embedding table for host-side lookup

The first launch fails with "model not found" until the install script has been run. `.tflite`
files are never committed; the conversion scripts in `conversion/` reproduce them.

## Build

```bash
cd text-embedding/
./gradlew :app:installDebug
scripts/install_to_device.sh    # stage the model + embedding table
```

`minSdk 26`, `arm64-v8a`, LiteRT `CompiledModel` GPU.

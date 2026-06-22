# Text Generation (LLM) — conversion recipes

How the four `.litertlm` LLMs in this repo were produced, with the **official**
[`litert-torch`](https://github.com/google-ai-edge/litert-torch) `export_hf` converter —
no fork, no custom graph code. Each is a dense decoder that rides the existing converter
and the [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime directly.

| Model | Base | Recipe | KV cache | externalize_embedder | Chat template |
|---|---|---|---|---|---|
| Falcon3-3B-Instruct | tiiuae/Falcon3-3B-Instruct | int4 **block128** | 2048 | no (≤2 GiB) | Falcon `<|user|>`/`<|assistant|>` |
| Llama-3.2-3B-Instruct | meta-llama/Llama-3.2-3B-Instruct | int4 **block32** | 4096 | **yes** | Llama-3 headers |
| Ministral-3-3B-Instruct-2512 | mistralai/Ministral-3-3B-Instruct-2512 | int4 **block32 + OCTAV** | 4096 | **yes** | Mistral `[INST]…[/INST]` |
| SmolLM3-3B | HuggingFaceTB/SmolLM3-3B | int4 **block32 + OCTAV** | 4096 | **yes** | bare ChatML |

## The three non-default choices (and why)

1. **Blockwise int4, not the named channelwise int4.** `litert-torch`'s named int4 recipe
   (`dynamic_wi4_afp32`) is **channelwise**, which collapses small LLMs (GSM8K ~46% vs bf16 ~75%
   on a 3B). Pass a **blockwise** recipe as a `recipe.json` instead — `recipes/int4_block32_octav.json`
   (block 32 + OCTAV optimal-clipping, the strongest data-free int4) or `recipes/int4_block128.json`
   (coarser = lighter dequant = faster GPU decode). Embeddings stay INT8. See
   [google-ai-edge/litert-torch#1068](https://github.com/google-ai-edge/litert-torch/issues/1068).
   The recipes are model-agnostic (`regex: ".*"`).
2. **`externalize_embedder=True` for ≥~2 GB models (iOS).** A 3B's weights are otherwise a single
   >2 GiB `TFLiteModel` section, which exceeds the iOS single-section `mmap` limit (engine create fails
   *"Failed to map section: Cannot allocate memory"*). Externalizing the (tied) embedding into its own
   section drops the main section <2 GiB (and dedups the tied matrix). Falcon3-3B (~1.7 GB) doesn't need it.
3. **A simple chat template** is forced so the runtime's minimal jinja (minja) can render it. Mistral's
   tekken tokenizer has no `<|im_end|>` (so Ministral must use `[INST]…[/INST]` + EOS `</s>`, **not**
   ChatML, or int4 runs away); SmolLM3's official template (`strftime_now`, tool blocks) doesn't render,
   so it ships bare ChatML. Templates are in `templates/`.

## Reproduce

`convert.py` is a thin wrapper over the official `export_hf` that (a) registers the blockwise recipes by
name, (b) forces the chosen `templates/*.jinja`, and (c) reads `CACHE` / `EXTERNALIZE_EMBEDDER` from the
env. Recipe names: `BMIX4` = block32, `BOCTAV4` = block32+OCTAV, `BMIX4_128` = block128.

```bash
pip install ai-edge-litert litert-torch transformers          # official tools

# Falcon3-3B — dense LlamaForCausalLM, block128, no externalize
CACHE=2048 python convert.py tiiuae/Falcon3-3B-Instruct \
    out/falcon3 templates/falcon_simple.jinja BMIX4_128

# Llama-3.2-3B — block32, externalize for iOS
EXTERNALIZE_EMBEDDER=1 CACHE=4096 python convert.py meta-llama/Llama-3.2-3B-Instruct \
    out/llama32 templates/llama_simple.jinja BMIX4

# Ministral-3-3B — multimodal source: extract the text decoder first (drops the Pixtral vision tower),
# then convert. Use the BF16 repo (the plain repo ships FP8, which won't load on CPU).
python extract_text_decoder.py                                # → src_models/ministral3-3b-text
EXTERNALIZE_EMBEDDER=1 CACHE=4096 python convert.py src_models/ministral3-3b-text \
    out/ministral3 templates/mistral_simple.jinja BOCTAV4

# SmolLM3-3B — dense SmolLM3 (GQA + NoPE), converts clean; bare-ChatML template
EXTERNALIZE_EMBEDDER=1 CACHE=4096 python convert.py HuggingFaceTB/SmolLM3-3B \
    out/smollm3 templates/smollm3_nothink.jinja BOCTAV4
```

Output is `out/<model>/model.litertlm` (tokenizer + template bundled).

### Pure-`export()` equivalent (official tools only, for Falcon3 / Llama where the jinja renders)

```python
from litert_torch.generative.export_hf.export import export
export(
    model="tiiuae/Falcon3-3B-Instruct",
    output_dir="out/falcon3",
    quantization_recipe="recipes/int4_block128.json",   # or recipes/int4_block32_octav.json
    cache_length=2048,
    # externalize_embedder=True,   # add for ≥2 GB models (Llama/Ministral/SmolLM3)
    trust_remote_code=True,
)
```

## Verify before publishing

`verify_quality.py` runs an 8-question non-degeneracy gate on the `.litertlm` via a LiteRT-LM runtime
(refuses obviously-broken quantizations):

```bash
python verify_quality.py out/ministral3/model.litertlm --json gate.json
```

Quality was further checked with GSM8K (n=100, greedy) vs the bf16 reference — int4 holds parity
(SmolLM3 0-pt, Ministral −4, Llama −5, Falcon ≈). See each model's section in the repo root `README.md`.

## License

Recipes/scripts: MIT (this repo). Each converted model inherits its base model's license
(Apache-2.0 for Ministral / SmolLM3; Falcon LLM License; Llama 3.2 Community License).

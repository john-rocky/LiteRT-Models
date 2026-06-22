"""Extract the text decoder from Ministral-3-3B (Mistral3ForConditionalGeneration)
into a standalone Ministral3ForCausalLM checkpoint that litert_torch's normal
TEXT_GENERATION path can convert.

Structure (transformers 5.12):
  Mistral3ForConditionalGeneration
    .model (Mistral3Model)
        .vision_tower            <- pixtral, DROP
        .multi_modal_projector   <- DROP
        .language_model (Ministral3Model)   <- KEEP -> causal.model
    .lm_head (nn.Linear)                     <- KEEP -> causal.lm_head
  (tie_word_embeddings=True: lm_head.weight == model.embed_tokens.weight)

    python scripts/extract_ministral3_text.py <src_dir> <out_dir>
"""

import gc
import sys

import torch
import transformers

src = sys.argv[1]
out = sys.argv[2]

print(f"Loading multimodal wrapper from {src} ...")
full = transformers.AutoModelForImageTextToText.from_pretrained(
    src, dtype=torch.bfloat16, low_cpu_mem_usage=True
)
full.eval()

text_cfg = full.config.text_config
print(f"text_config model_type={text_cfg.model_type} "
      f"hidden={text_cfg.hidden_size} layers={text_cfg.num_hidden_layers} "
      f"vocab={text_cfg.vocab_size} tie={text_cfg.tie_word_embeddings}")

print("Building standalone Ministral3ForCausalLM ...")
causal = transformers.Ministral3ForCausalLM(text_cfg)
causal.eval()

# language_model (Ministral3Model) -> causal.model (Ministral3Model): identical keys.
missing, unexpected = causal.model.load_state_dict(
    full.model.language_model.state_dict(), strict=False
)
print(f"  model load: missing={len(missing)} unexpected={len(unexpected)}")
if missing:
  print("   missing[:10]:", missing[:10])
if unexpected:
  print("   unexpected[:10]:", unexpected[:10])

causal.lm_head.load_state_dict(full.lm_head.state_dict())
causal.tie_weights()

# Sanity: a couple of weights should be finite and non-zero.
w = causal.model.embed_tokens.weight
print(f"  embed_tokens: shape={tuple(w.shape)} finite={torch.isfinite(w).all().item()} "
      f"absmean={w.abs().float().mean().item():.5f}")

del full
gc.collect()

print(f"Saving text-only checkpoint to {out} ...")
causal.save_pretrained(out, safe_serialization=True)

tok = transformers.AutoTokenizer.from_pretrained(src)
tok.save_pretrained(out)
print("EXTRACT_DONE")

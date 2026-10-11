# Host contract

This app implements the laya typed-decision contract for Phocinae-Largha-150M-v1 (the model's own runtime,
phocinae-server, builds the same sequence):

    CLS <type> question: <instructions> SEP [MASK] option … SEP <state> SEP

head budget 192 tokens (`rl_agent_config.json` head_max_len), window 256 / 384 / 512; options rendered as
`key: description` (choice), `level i: text` (score) and `false: … / true: …` (noul); a dict / list state serialized
as `json.dumps(state, ensure_ascii=False)`. The graph takes `inputs_embeds` (the host looks up the checkpoint's
float16 rows), `attention_mask` and `qtype_onehot` (choice, score, noul) and returns `token_logits` and `pooled_cls`;
the host gathers the option markers, divides by the question type's temperature from `rl_agent_config.json`
(choice 0.8660, score 0.8081, noul 0.6625) and takes a softmax; the act head takes `pooled_cls` and four features of
the untempered softmax. The full description with the measured numbers: the model card at
https://huggingface.co/litert-community/Phocinae-Largha-150M-LiteRT (section "Raw CompiledModel contract") and its
`host/phocinae_litert.py`, which is the Python reference of every step the Kotlin host repeats.

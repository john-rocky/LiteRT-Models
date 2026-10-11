# Component licenses and attribution

The sample's own license is [Apache-2.0](../LICENSE). Third-party license texts
below are retained with their source declarations; [NOTICE](../NOTICE) describes
the port and model-conversion changes.

| Component | License copy | Source |
|---|---|---|
| Phocinae-Largha-150M-v1 (model, tokenizer, config) | [Phocinae-APACHE-2.0.txt](Phocinae-APACHE-2.0.txt), [Phocinae-NOTICE.txt](Phocinae-NOTICE.txt) | [Model repository LICENSE / NOTICE](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1/tree/3807157407d14497a7ebd7114a3d71847b5b10a5) |
| Laya host protocol (the Kotlin host port) | [Laya-APACHE-2.0.txt](Laya-APACHE-2.0.txt) | Laya source LICENSE at commit 42626c348753fbb17572a813127df2278a1ec527; [pinned model declaration](https://huggingface.co/convaiinnovations/laya/blob/1c5edc17a7acd8701df6fc341c0d179f1c62c982/README.md) |
| ModernBERT | [ModernBERT-APACHE-2.0.txt](ModernBERT-APACHE-2.0.txt) | [Upstream LICENSE at c6d9423](https://github.com/AnswerDotAI/ModernBERT/blob/c6d942312f1b0b24d423628b8e477a3e97c7038f/LICENSE) |
| mmBERT-small | [mmBERT-MIT.txt](mmBERT-MIT.txt) | [Pinned model-card MIT declaration and author attribution](https://huggingface.co/jhu-clsp/mmBERT-small/blob/c5955035435e2bf121cde7f3c8863ef52ff35d82/README.md); [standard MIT text](https://opensource.org/license/mit) |
| Transformers 5.17.0 | [Transformers-APACHE-2.0.txt](Transformers-APACHE-2.0.txt) | License retained from the installed 5.17.0 distribution; [upstream project](https://github.com/huggingface/transformers) |
| tokenizers 0.23.2 | [Tokenizers-APACHE-2.0.txt](Tokenizers-APACHE-2.0.txt) | [Upstream LICENSE at v0.23.2](https://github.com/huggingface/tokenizers/blob/v0.23.2/LICENSE) |

The ModernBERT license retains its received `Copyright 2022 MosaicML Examples authors`
notice. The pinned mmBERT model repository declares MIT without a separate LICENSE/NOTICE
or copyright notice. Its standard permission text and model-card author attribution are
included; no copyright holder or year is invented.

The checkpoint is the Phocinae project's fine-tune of mmBERT-small, not a newly downloaded mmBERT
checkpoint; the MIT text covers the encoder it derives from.

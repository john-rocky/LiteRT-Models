# Exact graph rewrites
Use the pinned Python environment from `../requirements.txt`; invoke Python with `-B`.
`export.py` reads a local checkpoint and your own prepared NPZ example; no audio examples are bundled.
AR prefill selects its final row with `last_onehot`; packed-cache step inputs remain unchanged.
Acoustic condition uses semantic and frame one-hot matrices instead of runtime gathers.
Acoustic velocity replaces speaker expansion with an exact one-term matrix product.
Semantic encoding returns digit logits; integer FSQ indexing runs in the application.
Style attention and other attention products use rank-four matrix operands.
Export fp32 first, then request `--precision wfp16`; merged AR also supports native int8.
Run `python -B scripts/reexport_r6/export.py --help` from the module root for paths and options.

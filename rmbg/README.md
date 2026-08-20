
### Converting your own fine-tuned checkpoint

`RMBG_WEIGHTS=/path/to/model.safetensors` swaps in your own fine-tuned ISNet weights
(the model code still comes from the official `briaai/RMBG-1.4` snapshot; same
architecture required — default run reproduces the official ship exactly):

```bash
RMBG_WEIGHTS=/path/to/my_isnet.safetensors python scripts/convert_rmbg14.py
```

Mind the RMBG-1.4 license: fine-tunes inherit its non-commercial terms — for a
permissive alternative train on the DIS IS-Net recipe instead (see the `dis` module).

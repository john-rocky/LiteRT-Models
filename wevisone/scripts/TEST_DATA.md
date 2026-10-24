# Reference data for the JVM tests (`-Pwevisone.golden=<dir>`)

The tests in `app/src/test` compare the app's host code with the references of the conversion run. Without the directory only the committed public rows run (`app/src/test/resources/tok_rows_public.json`, which still needs the checkpoint's `tokenizer.json`, so those tests are skipped too). The directory holds:

| File | What | Made with |
|---|---|---|
| `tokenizer.json` | the checkpoint's tokenizer (tencent/WeVisOne-VL-2B @ 3dad9ce8, SHA-256 3bb17242…fc5e) | copied |
| `tok_rows.json` | 29 fixture rows: `instruction`, the reference `prompt_ids` (the HF processor with the fixed system turn, image at 512 × 512), the reference answer `new_ids` and `response` (fp32, greedy), the authors' parses `seg_boxes` and `shapes` (in bins) | transformers 5.12.1, the authors' package at 40f735b8 |
| `resize_index.json`, `img_<stem>.rgb`, `img_<stem>_512.rgb`, `img_<stem>_1024.rgb` | the fixture images as decoded RGB (row-major, 3 bytes a pixel) and Pillow's BICUBIC to 512² and 1024² | Pillow 12.2.0 |
| `box_bins.f32`, `box_f_bins.f32`, `box_widths.f32`, `box_ew.f32`, `box_f_ew.f32`, `box_index.json`, `box_row_<row>.f32` | the authors' `FourierPE1D` of every coordinate bin and every encoded box size, 0.2 log w + 0.5 for every box width, and six rows' 256 box features | torch 2.11.0 |
| `alpha_index.json`, `alpha_<row>_logits.f32`, `alpha_<row>_ref.f32` | six rows' matte logits at 1024² and the authors' sigmoid + bilinear to the photo size | torch 2.11.0 |

`.f32` files are little-endian float32.

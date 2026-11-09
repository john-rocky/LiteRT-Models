# SPEED_LADDER — speridlabs/idol + idol-turbo → LiteRT (litert-community/Idol)

runtime: litert
model: speridlabs/idol with the speridlabs/idol-turbo LoRA (5 steps), published as litert-community/Idol (+ litert-community/DINOv3-ViT-H16plus-Idol-LiteRT)
device: Galaxy S26 (SM-S942Q, RFGL80R6A6H) LiteRT 2.3.0, the sample app / Pixel 11 Pro (Tensor G6) LiteRT 2.3.0 / Mac M4 Max (Metal) LiteRT 2.3.0, Python runner
parity bar: CONTRACT.md, fixed before the first run: (a) per-part cosine vs the author's PyTorch pipeline ≥ 0.99 on every line; (b) 4-view render PSNR of the whole pipeline vs the author's ≥ 25 dB

## 梯子(ladder — 1 行 = 1 form。測った形は全部残す。ship = 出荷する形)
| # | form | backend | device | workload | metric | parity | status | evidence | ship |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fp16 weights, GPU FP16_WITH_FP32_ACCUM, 1024² input, 8,192 anchors, 5 turbo steps, 45 GPU graphs; BiRefNet on the LiteRT CPU | gpu | Galaxy S26, the app | one photo, the first after start on a cool phone | 303.2 s per photo (generate 291.4 s) | PASS (a) 21/21 lines (armchair), (b) 41.43–46.59 dB on 3 inputs | measured | contract/c_s26_cool.json, contract/a_armchair_s26.json, contract/b_armchair_s26.json | ship |
| 2 | the same form | gpu | Galaxy S26, the app | one photo on a phone already warm (thermal status 2) | 338.6–448.7 s per photo | PASS (same graphs as row 1) | measured | contract/c_s26.json, contract/e2e_s26_hold3.json | ship |
| 3 | the same graphs, Python runner (`python/run_idol.py`) | gpu | Mac M4 Max (Metal) | one photo | 63.8 s per photo, peak 22 GB | PASS (a) Mac lines, (b) 32.21–36.38 dB | measured | contract/a_v3_armchair_mac.json, contract/standin_v3, /Users/majimadaisuke/code/idol_work/hf_test/run.log | |
| 4 | the whole pipeline on the CPU | cpu | Galaxy S26 | one photo | — | — | pending: not run; only BiRefNet runs on the CPU today (8.5–17.7 s per photo) | contract/e_s26.json | |
| 5 | denoiser blocks on the Qualcomm HTP (AOT), the rest as row 1 | npu | Galaxy S26 | one photo | — | — | pending: not tried; the denoiser is 228.4 s of row 1's 303.2 s | contract/c_s26_cool.json | |

GPU clock: on the S26 the cap falls from 1,300 to 422 MHz within one photo, even from a cool phone (thermal status 1 → 3, `contract/c_s26_cool_thermal.json`, trace `contract/s26_thermal.csv`); a phone already warm gives row 2. Every lever that cuts GPU time also cuts the throttled tail.

## レバー(levers — 試すべき手を全部。1 行 = 1 手。n-a は「この model には当てはまらない」理由つき)
| lever | status | result / reason | evidence |
|---|---|---|---|
| quant | pending: int8 weights for the 26 denoiser graphs (228.4 s of 303.2 s) and the 8 Gaussian decoder graphs (36.0 s), checked against lines (a) and (b) | — | contract/c_s26_cool.json |
| precision | pending: GPU FP16 accumulation timed against FP16_WITH_FP32_ACCUM on the S26; the S26's (b) 41.43–46.59 dB vs Metal fp16's 32.21–36.38 dB is the margin it would spend | — | contract/b_armchair_s26.json |
| shape | pending: 512² input and 2,048 anchors (the window's ladder steps, D-908), each scored on lines (a) and (b) | — | CONTRACT.md |
| npu-path | pending: the denoiser blocks on the S26 HTP, AOT on Linux x86, delegate lines and partition count read from the log | — | contract/c_s26_cool.json |
| state | pending: fewer compiles per photo; the graph pool compiles 115 times per photo (69.8 s of the cool photo) because the 45 graphs do not fit in memory at once with the 2,500 MB reserve | — | contract/c_s26_cool.json |

## 例外(exception — 同じ device + workload でもっと速い形を出荷しないときだけ理由。無ければ「なし」)
なし

## 残り(remaining — まだ試していないレバーと見込み。無ければ「なし」。1 件でも書けば lint は HOLD)
- int8 denoiser and Gaussian decoder (quant): the denoiser is three quarters of the photo.
- 512² input and 2,048 anchors (shape): fewer tokens per denoiser call.
- The S26 HTP for the denoiser blocks (npu-path).
- Fewer compiles per photo (state): 115 compiles, 69.8 s.
- FP16 accumulation (precision).
- A CPU row for the whole pipeline (the ladder's cpu backend).

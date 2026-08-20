# RTMPose-family conversion (rtmpose / rtmface / rtmhand / rtmanimal / rtmw)

`build_rtm_multi.py` is the one converter for all five RTMPose zoo modules. It loads any
mmpose SimCC model via `init_model(cfg, ckpt)`, applies the GPU re-authorings (ScaleNorm →
SafeRMSNorm, GAU act@act BMM → broadcast-reduce, RTMW PixelShuffle → depth-to-space
ZeroStuffConvT2d — all numerically exact), converts with litert-torch, writes fp32 + fp16
tflites, op-checks GPU-cleanliness, and prints tflite-vs-torch corr.

Requirements: `pip install mmpose mmengine mmcv litert-torch ai-edge-quantizer ai-edge-litert`
(mmdet is stubbed out — not needed).

## Official ship reproductions

| module | RTM_CFG (relative to `mmpose/.mim/configs`) | RTM_CKPT | RTM_H×RTM_W |
|---|---|---|---|
| rtmpose (body) | `body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py` | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth` | 256×192 |
| rtmface | `face_2d_keypoint/rtmpose/wflw/rtmpose-m_8xb64-60e_wflw-256x256.py` | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-wflw_pt-aic-coco_60e-256x256-dc1dcdcf_20230228.pth` | 256×256 |
| rtmhand | `hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py` | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth` | 256×256 |
| rtmanimal | `animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py` | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth` | 256×256 |
| rtmw (whole-body) | `wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py` | `https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth` | 256×192 |

Example (rtmw):

```bash
RTM_CFG=wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py \
RTM_CKPT=https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth \
RTM_H=256 RTM_W=192 RTM_NAME=rtmw python build_rtm_multi.py
```

## Converting your own fine-tuned checkpoint

Every GPU patch is architecture-level and weight-independent, so any SimCC-head model
fine-tuned with the mmpose trainer converts the same way. `RTM_CFG` also accepts an
existing local file path — point it at your training config:

```bash
RTM_CFG=/path/to/your_finetune_config.py RTM_CKPT=/path/to/best_ckpt.pth \
RTM_H=256 RTM_W=192 RTM_NAME=mymodel python build_rtm_multi.py
```

The keypoint count flows from the head weights into the output shapes
`simcc_x [1, K, W·split]`, `simcc_y [1, K, H·split]` — update the app's keypoint
count/skeleton table to match. `RTM_H`/`RTM_W` must equal the config's `input_size`
(mmpose lists it as (w, h) — RTM_W is the first entry). Non-SimCC heads (heatmap-based)
are not covered.

# LiteRT-Models

This repo has more than 120 models converted to run on LiteRT, Google's on-device ML runtime for Android. LiteRT is the new name of TensorFlow Lite. Most models come with input and output shapes, preprocessing, the conversion script and a standalone Kotlin sample app.

<img src="https://github.com/user-attachments/assets/3c764bf0-32ac-4b65-814b-a73e1aba1cfc" width="100%">

Most of the `.tflite` models run every op on the phone's GPU through [`CompiledModel`](https://ai.google.dev/edge/litert). For example, DINOv2 ViT-S/14 at 448² runs in 53.6 ms on a Galaxy S26 GPU. The chat models run on [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), on a phone or a Mac.

**If you like this repository, please give it a star.**

## Run a model

Download a `.tflite` from [MODELS.md](MODELS.md), put it in `app/src/main/assets/`, and add the dependency from Google Maven:

```kotlin
implementation("com.google.ai.edge.litert:litert:2.1.3")
```

```kotlin
val options = CompiledModel.Options(Accelerator.GPU)
val model = CompiledModel.create(context.assets, "model.tflite", options, null)
val inputs = model.createInputBuffers()
inputs[0].writeFloat(floatArray)
val outputs = model.run(inputs)
val result = outputs[0].readFloat()
```

Each model's section in [MODELS.md](MODELS.md) says how to prepare `floatArray`. Reading the output waits for the GPU, so time the run and the read together. Models over about 150 MB load from a file path instead of the APK. From LiteRT 2.3.0, the GPU also needs `com.google.ai.edge.litert:litert-gpu` at the same version as `litert`.

## What's here

| Task | Models |
|---|---|
| Detection and tracking | [YOLO11n](MODELS.md#yolo11n), [YOLO26n](MODELS.md#yolo26n), <a name="rf-detr-nano"></a>[RF-DETR Nano](MODELS.md#rf-detr-nano), <a name="rt-detrv2-s"></a>[RT-DETRv2-S](MODELS.md#rt-detrv2-s), <a name="d-fine-s"></a>[D-FINE-S](MODELS.md#d-fine-s), [SSDLite320 MobileNetV3](MODELS.md#ssdlite320-mobilenetv3), [YOLO + DeepSORT](MODELS.md#yolo--deepsort-osnet), [Ultra-Fast-Lane-Detection](MODELS.md#ultra-fast-lane-detection), [TwinLiteNet](MODELS.md#twinlitenet), [M-LSD-tiny](MODELS.md#m-lsd-tiny), [YuNet](MODELS.md#yunet), [RTMPose-Face](MODELS.md#rtmpose-face-wflw-98-point-face-alignment), [SINet-V2](MODELS.md#sinet-v2), [DM-Count](MODELS.md#dm-count) |
| Segmentation and cutouts | [PIDNet-S](MODELS.md#pidnet-s-real-time-cityscapes), [YOLACT-ResNet50](MODELS.md#yolact-resnet50), [MobileSAM](MODELS.md#mobilesam), [SAM 2.1](MODELS.md#sam-21-hiera-tiny), [EdgeTAM](MODELS.md#edgetam-sam2), [EdgeTAM Video](MODELS.md#edgetam-video-sam2-tracking), [CLIPSeg](MODELS.md#clipseg), [Cloth Segmentation (U²-Net)](MODELS.md#cloth-segmentation-u-net), [RMBG-1.4](MODELS.md#rmbg-14-isnet), [ormbg](MODELS.md#ormbg-open-apache-20), [DIS](MODELS.md#dis-is-net-general-use), [MODNet](MODELS.md#modnet-trimap-free), [WeVisOne-VL-2B](wevisone/) |
| Pose and face | [YOLO26n-pose](MODELS.md#yolo26n-pose), [RTMPose-s](MODELS.md#rtmpose-s), [RTMW-m](MODELS.md#rtmw-m-whole-body-133-keypoints), [RTMPose-Hand](MODELS.md#rtmpose-hand-21-keypoints), [RTMPose-Animal](MODELS.md#rtmpose-animal-ap-10k-17-keypoints), [6DRepNet](MODELS.md#6drepnet), [L2CS-Net](MODELS.md#l2cs-net), [3DDFA_V2](MODELS.md#3ddfa_v2-3d-face-alignment), [BiSeNet](MODELS.md#bisenet-face-parsing), [HSEmotion](MODELS.md#hsemotion-facial-emotion-recognition), [Silent-Face](MODELS.md#silent-face-minifasnetv2) |
| Image enhancement and editing | [EDSR](MODELS.md#edsr-4), [Real-ESRGAN x4v3](MODELS.md#real-esrgan-x4v3), [GFPGAN v1.4](MODELS.md#gfpgan-v14-blind-face-restoration), [DehazeFormer-MCT](MODELS.md#dehazeformer-mct), [CPGA-Net](MODELS.md#cpga-net), [NAFNet](MODELS.md#nafnet-deblur), [LaMa-Dilated](MODELS.md#lama-dilated), [MI-GAN](MODELS.md#mi-gan-mobile-inpainting--object-removal), [Fast Neural Style](MODELS.md#fast-neural-style-4-styles), [U²-Net Portrait](MODELS.md#u-net-portrait), [DewarpNet](MODELS.md#dewarpnet) |
| Understanding images and video | [Vision-RWKV](MODELS.md#vision-rwkv-vrwkv-s), [PlantNet-300K](MODELS.md#plantnet-300k-1081-plant-species), [CLIP ViT-B/32](MODELS.md#clip-vit-b32), [Places365 ResNet18](MODELS.md#places365-resnet18-scene-recognition), [RAM++](MODELS.md#ram-recognize-anything-plus), [NIMA](MODELS.md#nima-neural-image-assessment), [XFeat](MODELS.md#xfeat-local-features), [UniSal](MODELS.md#unisal), [DINOv2 ViT-S/14](MODELS.md#dinov2-vit-s14), [PP-OCRv5](MODELS.md#pp-ocrv5), [MoViNet-A0](MODELS.md#movinet-a0-streaming-kinetics-600) |
| Depth and 3D | [Idol](MODELS.md#idol-speridlabsidol--idol-turbo), [MoGe-2 ViT-S](MODELS.md#moge-2-vit-s), [Depth Anything 3 ViT-S](MODELS.md#depth-anything-3-vit-s-small), [Metric3D v2 ViT-S](MODELS.md#metric3d-v2-vit-s), [TIPSv2-B/14 DPT](MODELS.md#tipsv2-b14-dpt-depth-normals-segmentation), [DSINE](MODELS.md#dsine) |
| Image generation | [iris-3b Turbo](MODELS.md#iris-3b-turbo), [Bonsai Image 4B](bonsai_image/) |
| Speech recognition | [Parakeet](MODELS.md#parakeet-fastconformer-ctc), [Whisper-tiny](MODELS.md#whisper-tiny), [wav2vec2-CTC](MODELS.md#wav2vec2-ctc-fully-gpu-single-pass), [Qwen3-ASR-1.7B](asr_litertlm/), [Qwen3-ASR-1.7B-15s](asr_litertlm/), [Fun-ASR-Nano-2512](asr_litertlm/), [Confucius4-R2T2](asr_litertlm/) |
| Text-to-speech | [Kokoro-82M](MODELS.md#kokoro-82m), [Matcha-TTS](MODELS.md#matcha-tts), [EMA Lightning](MODELS.md#ema-lightning-turkish-tts-86m), [Dia2-1B](MODELS.md#dia2-1b-dialogue), [VibeVoice-Realtime-0.5B](MODELS.md#vibevoice-realtime-05b), [Pocket TTS](MODELS.md#pocket-tts-kyutai), [KittenTTS nano 0.8](MODELS.md#kittentts-nano-08-dynamic-length), [Inflect-Nano-v2](MODELS.md#inflect-nano-v2-dynamic-length-exact-streaming), [Sopro v2 turbo](MODELS.md#sopro-v2-turbo-zero-shot-voice-cloning), [Audio8-TTS-Preview-0.6b](audio8_tts/), [Ming-3.0-tts-0.5B](ming3tts/) |
| Voice assistant, speaker labels and noise removal | [pyannote 3.1 stack](MODELS.md#pyannote-31-stack-segmentation--wespeaker), [Nemotron-3-Diarization](MODELS.md#nemotron-3-diarization-streaming-sortformer-8-speakers), [CMGAN](MODELS.md#cmgan-noise-suppression), [RE-USE](MODELS.md#re-use-universal-speech-restoration), [Whisper + SmolLM2 + Kokoro pipeline](MODELS.md#whisper--smollm2--kokoro-pipeline) |
| Audio and music | [wav2vec2 Keyword Spotting](MODELS.md#wav2vec2-keyword-spotting), [PANNs CNN14 Audio Tagging](MODELS.md#panns-cnn14-audio-tagging), [DAC 16kHz](MODELS.md#dac-16khz), [Mimi](MODELS.md#mimi-kyutai-2024), [TIGER-DnR](MODELS.md#tiger-dnr-dialog--effects--music), [CREPE](MODELS.md#crepe), [Basic Pitch](MODELS.md#basic-pitch-audio-to-midi) |
| Chat and text generation | [RWKV-7 World 0.1B](MODELS.md#rwkv-7-world-01b), [Falcon3-3B-Instruct](MODELS.md#falcon3-3b-instruct), [Llama-3.2-3B-Instruct](MODELS.md#llama-32-3b-instruct), [Ministral-3-3B-Instruct-2512](MODELS.md#ministral-3-3b-instruct-2512), [SmolLM3-3B](MODELS.md#smollm3-3b), [Qwen3.8-27B](MODELS.md#qwen38-27b-mac-chat-app) |
| Vision-language | [SmolVLM-256M](MODELS.md#smolvlm-256m), [SmolVLA](MODELS.md#smolvla-lerobotsmolvla_base) |
| Embeddings, reranking, entity extraction and text classification | [Qwen3-Embedding-0.6B](MODELS.md#qwen3-embedding-06b), [Qwen3-Reranker-0.6B](MODELS.md#qwen3-reranker-06b), [GLiNER2.5 Small](MODELS.md#gliner25-small-schema-based-entity-extraction), [GLiNER2.5 Multi](MODELS.md#gliner25-multi-multilingual-schema-based-entity-extraction-english--japanese-verified), [GLiFormer Large v1 — NER path](MODELS.md#gliformer-large-v1--ner-path-schema-prompted-entity-extraction-575m), [Laya Multilingual](MODELS.md#laya-multilingual-typed-text-decisions), [ModernBERT-Ja-310M Decision](MODELS.md#modernbert-ja-310m-decision-japanese-cross-encoder-typed-decisions), [GLiNER2.5-Decide](MODELS.md#gliner25-decide-zero-shot-text-classification), [GLiClass-Edge v3.0](MODELS.md#gliclass-edge-v30-zero-shot-text-classification) |
| Decision models: they answer yes-no, choice and score questions about text, JSON or a photo | [Julia-1](MODELS.md#julia-1-typed-decisions), [Phocinae-Largha-150M](MODELS.md#phocinae-largha-150m-typed-decisions), [Open Decision](MODELS.md#open-decision-deberta-v3-large-typed-decisions), [Kev-0.8B](MODELS.md#kev-08b-typed-decisions), [d1-omni-600M](MODELS.md#d1-omni-600m-typed-decisions-about-text-photos-and-speech), [d1-2B](MODELS.md#d1-2b-typed-reads-and-decisions-about-text-json-and-photos), [d1-Omni-2B](MODELS.md#d1-omni-2b-typed-reads-and-decisions-about-recordings-text-and-photos), [d1-3B](MODELS.md#d1-3b-typed-decisions-desktop), [clef-flash](MODELS.md#clef-flash-typed-decisions-desktop) |
| Time-series forecasting | [Timestral-209M](timestral/) |

[MODELS.md](MODELS.md) lists every model with its speed and download.

## Convert your own

To make a PyTorch model run fully on the GPU, start with [the conversion guide](docs/LITERT_CONVERSION_GUIDE.md). [`litert_gpu_toolkit/`](litert_gpu_toolkit/) has its rewrites as a Python package.

## Coming from TensorFlow Lite?

The `.tflite` files still work. The package names changed: [docs/litert-names.md](docs/litert-names.md) maps old to new.

## Projects using these models

- [PocketTTS-LiteRT](https://github.com/geneing/PocketTTS-LiteRT) runs Pocket TTS (Kyutai) on Android and builds on the [`pockettts/`](pockettts/) module.
- [dpm](https://github.com/OliverHenrichs/dpm), a tool for dancers to map the patterns they know, adapts the EdgeTAM video tracker in [`edgetam-video/`](edgetam-video/) to anonymize video.
- [ImageSplitter](https://github.com/mozico-565/ImageSplitter), an Android app that splits a photo into numbered parts for large prints, upscales with `real_esrgan_x4v3.tflite` from this repo's `v1` release.

## License

MIT for the sample apps. `kev/`, `julia1/`, `d1-3b/`, `clef-flash/` and `wevisone/` carry their own Apache-2.0 `LICENSE`, and `d1omni/` the LFM Open License v1.0, as it ports the model provider's code. Model licenses follow their original projects.

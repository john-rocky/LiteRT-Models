# NIMA — On-device image quality assessment (LiteRT GPU)

[NIMA (Neural Image Assessment)](https://github.com/idealo/image-quality-assessment) (idealo,
Apache-2.0) scores a photo's quality on a **1-10** scale. Two MobileNet models — **aesthetic**
(trained on AVA) and **technical** (trained on TID2013) — each predict a 10-bin score distribution;
the score is the distribution mean. Both run fully on the LiteRT CompiledModel **GPU**.

## On-device (Pixel 8a, Tensor G3 — verified)

| model | in → out | delegate | size |
|---|---|---|---|
| NIMA aesthetic | image [1,224,224,3] → dist [10] | **GPU** | 6.4 MB fp16 |
| NIMA technical | image [1,224,224,3] → dist [10] | **GPU** | 6.4 MB fp16 |

tflite-vs-Keras score parity: aesthetic **0.999998**, technical **0.999915**.

```
image →[resize 224² · MobileNet /127.5−1]→ [GPU MobileNet]→ softmax dist[10] →[Σ i·pᵢ]→ score 1-10
```

## How it converts

NIMA is `MobileNet(224², include_top=False, pooling='avg')` → Dense(10, softmax) — a pure CNN, so it
converts straight through `tf.lite` (fp16) with no re-authoring, and every op rides the GPU delegate
(the canonical MobileNet op set). The 10-bin distribution is the graph output; the 1-10 mean is
computed host-side. The idealo Keras-2 weights load cleanly in Keras 3.

## Build & run

```bash
python build_nima.py            # -> nima_{aesthetic,technical}_fp16.tflite  (needs ~/tfconv, TF 2.x)
# put the two .tflite in app/src/main/assets/  (bundled in the APK; also on Hugging Face)
./gradlew :app:installDebug
```

Launch the app — it scores the bundled sample on start; tap **Pick image** for your own photos.

Models are small enough to bundle in the APK (`app/src/main/assets/`). Also on Hugging Face:
`litert-community/NIMA-LiteRT`. Upstream:
[idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) (Apache-2.0).

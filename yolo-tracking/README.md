# YOLO + DeepSORT Tracking

Real-time multi-object tracking on Android using **YOLO11n** detection + **OSNet x0.25** appearance features + **DeepSORT** tracker, all running on **LiteRT CompiledModel GPU**.

## Architecture

```
Camera Frame (640x480)
    │
    ▼
┌─────────────────────────┐
│  YOLO11n (GPU)          │  Object detection
│  384x384 → [1,84,3024]  │  ~20ms on Pixel 8a
└───────────┬─────────────┘
            │ BBox crops
            ▼
┌─────────────────────────┐
│  OSNet x0.25 (GPU)      │  Re-ID feature extraction
│  256x128 → 512-dim      │  ~5ms per crop
└───────────┬─────────────┘
            │ Detections + embeddings
            ▼
┌─────────────────────────┐
│  DeepSORT (CPU/Kotlin)  │  Kalman filter + Hungarian matching
│  Cosine + Mahalanobis   │  + cascade matching
└───────────┬─────────────┘
            │ Track IDs
            ▼
   Visualization with trails
```

## Models

| Model | Input | Output | Size | Source |
|-------|-------|--------|------|--------|
| YOLO11n | 384x384 RGB | [1,84,3024] boxes+scores | 10 MB | `yolo/` module |
| OSNet x0.25 | 256x128 RGB | 512-dim embedding | ~1.4 MB | torchreid |

## Setup

### 1. Convert OSNet model

```bash
pip install torchreid litert-torch
cd scripts/
python convert_osnet.py
cp osnet_x0_25.tflite ../app/src/main/assets/
```

### 2. Copy YOLO model

```bash
cp ../../yolo/app/src/main/assets/yolo11n.tflite app/src/main/assets/
```

### 3. Build and install

Open `yolo-tracking/` in Android Studio, or:

```bash
./gradlew :app:installDebug
```

## DeepSORT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxAge` | 30 | Frames before a lost track is deleted |
| `maxCosineDist` | 0.4 | Appearance matching threshold |
| `maxIouDist` | 0.7 | IOU matching threshold (fallback) |
| `N_INIT` | 3 | Consecutive detections to confirm a track |
| `nnBudget` | 100 | Max stored features per track identity |

## How DeepSORT Works

1. **Kalman Filter** predicts each track's next position (constant velocity model)
2. **Matching Cascade** associates detections to confirmed tracks using appearance similarity (cosine distance on Re-ID embeddings), gated by Mahalanobis distance from the Kalman state
3. **IOU Matching** handles remaining unconfirmed tracks and recently-lost confirmed tracks
4. **Track Lifecycle**: Tentative → (3 consecutive hits) → Confirmed → (30 frames lost) → Deleted

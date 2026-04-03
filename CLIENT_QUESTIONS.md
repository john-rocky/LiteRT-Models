# クライアントへの確認事項

## 背景
当方の検証で、DepthAnything V2 (ViT-S/14, 518x518) のPixel 8a実測値：
- LiteRT GPU推論dispatch: **7-10ms** (非同期)
- **実際のGPU計算完了**: **345ms** (Lock待ち)
- データ転送(DMA): **0ms** (Mali共有メモリ)

`readFloat()`の340msは「データ転送」ではなく「GPU計算時間」そのものでした。

## 確認事項

### 1. 計測方法
「FP32でiOSの2倍程度で動いていた」とのことですが:
- **どの時間を計測しましたか？**
  - a) `compiledModel.run()` の戻り時間のみ (~7ms)
  - b) `run()` + `readFloat()` の合計 (~350ms)
  - c) カメラ入力からdepth表示までの全体
- **計測コード**を共有いただけますか？

### 2. デバイス
- 使用デバイスは？（Pixel 8a? Samsung S24? 他?）
- Androidバージョンは？

### 3. モデル
- 使用モデルファイル名は？
- 入力サイズは？ (518x518? 392x518? 256x256?)

### 4. 出力の取得方法
- `outputBuffers[0].readFloat()` を使用していますか？
- 他の方法で出力を取得していますか？

## 共有モデル

以下のモデルを提供可能です:

| ファイル | サイズ | 精度 | 入力 |
|---|---|---|---|
| `depth_anything_v2_keras.tflite` | 99MB | FP32 | 518x518 |
| `depth_anything_v2_keras_fp16w.tflite` | 50MB | FP16重み | 518x518 |
| `depth_anything_v2_keras_392x518.tflite` | 98MB | FP32 | 392x518 |
| `depth_anything_v2_keras_392x518_fp16w.tflite` | 49MB | FP16重み | 392x518 |
| `depth_anything_v2_nhwc_clamped.tflite` | 99MB | FP32+clamp | 392x518 |
| `depth_anything_v2_nhwc_clamped_fp16w.tflite` | 50MB | FP16重み+clamp | 392x518 |

### clampモデルについて
`onnx2tf`変換時にViTアテンションの中間値をクランプ。FP16 GPUで数値オーバーフローを防止する試み。ただし品質への影響あり(corr≈0.85 vs native Keras corr=0.9995)。

## 当方の検証結果サマリー

```
Pixel 8a (Tensor G3, Mali-G715):
  LiteRT ML Drift: 345ms/frame (GPU計算限界)
  NCNN Vulkan:     3330ms/frame
  NCNN CPU:        2675ms/frame
  → LiteRTが最速、readFloat=GPU計算待ちで回避不可
  → GL buffer zero-copy: Mali-G715で非対応 (cl_khr_gl_sharing なし)
  → OpenCL Texture: 唯一の外部バッファ型だがreadは同速度
```

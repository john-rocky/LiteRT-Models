# DepthAnything V2 リアルタイム化 — 最終技術レポート

## 決定的発見: ボトルネックはデータ転送ではなくGPU計算時間

```
C++ SDK計測結果 (Pixel 8a, Tensor G3, Mali-G715):
  write = 15ms   (CPU→GPU入力書き込み)
  run   = 4-8ms  (GPU推論dispatch — 非同期)
  lock  = 345ms  (GPU計算完了待ち — 実際の推論時間)
  copy  = 0ms    (Mali共有メモリ — DMAコピー不要!)
```

**readFloat 340msの正体は「GPU計算時間345ms」。** データ転送は0ms（Maliの共有メモリアーキテクチャ）。zero-copyは既に達成されている。バッファ最適化では一切高速化できない。

## 推論時間の真実

| 計測値 | 意味 |
|---|---|
| Kotlin `inf=7ms` | 非同期dispatch（GPUにキューイング） |
| Kotlin `readFloat=340ms` | GPU計算完了待ち + Java配列確保 |
| C++ `run=4ms` | 非同期dispatch |
| C++ `lock=345ms` | **GPU計算完了待ち（実際の推論時間）** |
| C++ `copy=0ms` | Mali共有メモリ、DMAなし |

DepthAnything V2 ViT-S/14 (518x518) の**実際のGPU推論時間は~345ms**。

## 試行した全アプローチ

### バッファタイプ (全てC++ SDKで検証)

| タイプ | CreateManaged | Run | 理由 |
|---|---|---|---|
| managed (default) | ✅ | ✅ read=369ms | GPU stall |
| **kOpenClTexture (12)** | ✅ | **✅** lock=345ms, copy=0ms | **唯一の外部サポート型** |
| kGlBuffer (6) | ✅ | ❌ "not supported" | cl_khr_gl_sharing非対応 |
| kOpenClBuffer (10) | ✅ | ❌ "not supported" | |
| kAhwb (2) | ✅ | ❌ "not supported" | |

**`GetOutputBufferRequirements().supportedTypes` = [12] (kOpenClTexture のみ)**

### FP32/FP16

| 方式 | FP32設定 | 結果 |
|---|---|---|
| Kotlin CompiledModel | `GpuOptions.Precision.FP32` ✅ | 正常（全シーン） |
| C++ dlopen | 設定不可（関数未エクスポート） | FP16 → ViTオーバーフロー |
| **C++ SDK** | **`GpuOptions::SetPrecision(kFp32)` ✅** | **正常** |

### EGLコンテキスト

| 方式 | 結果 |
|---|---|
| dlopen + 手動struct | ABI間違い → EGL共有失敗 |
| dlopen + ABI修正 (24byte, tag 6/7) | GpuEnvironmentCreateに渡す → "Reusing provided EGL" |
| **C++ SDK `Environment::Create({})`** | **自動管理 ✅** |

### GLSurfaceView表示

| 設定 | 結果 |
|---|---|
| デフォルト | SurfaceView behind window → 黒 |
| setZOrderOnTop + OPAQUE | 黒 |
| setZOrderOnTop + TRANSLUCENT | 白（EGL破壊時）/ 表示（EGL正常時） |
| LiteRT init on bg thread | GLは正常だがFP16 → ゴミ出力 |
| LiteRT init on GL thread | GPU推論OK、GL表示壊れる |

### Zero-Copy試行

| 方式 | 結果 |
|---|---|
| SSBO `CreateTensorBufferFromGlBuffer` | CreateOK, Run status=3 |
| AHB `CreateTensorBufferFromAhwb` | CreateOK, Run status=3 |
| OpenCL buffer CreateManaged | CreateOK, Run "not supported" |
| GL buffer CreateManaged | CreateOK, Run "not supported" |
| **OpenCL texture CreateManaged** | **CreateOK, Run OK!** → lock=345ms, copy=0ms |
| Kotlin TensorBuffer Lock from C++ | status=3 (JNI wrapper) |
| Kotlin CompiledModel handle in C++ | クラッシュ (JNI wrapper) |
| Kotlin nativeRun from C++ | "events attached" エラー |

## NPU/TPU可能性

| パス | 実現性 | 理由 |
|---|---|---|
| Tensor G3 Darwinn TPU | **❌ 不可能** | 公開APIなし、INT8のみ、GELU/LayerNorm/BatchMatMul非対応 |
| NNAPI | △ 非推奨 | Android 15で非推奨、GPU/CPUにルーティングされるだけ |
| GPU (Mali-G715) | ✅ 現行 | 345ms/frame = ~3fps |
| CPU (XNNPACK) | △ フォールバック | ~500-1000ms |

**Tensor G3のDarwinn TPUは第三者開発者には非公開。** Google自社アプリ（カメラ、音声認識）のみ使用。

## 結論: Pixel 8aではDepthAnything V2のリアルタイムは不可能

**根本原因: Mali-G715のGPU計算能力不足 (345ms/frame for ViT-S)**

これはデータ転送の問題ではなく、GPU演算速度の限界。

## リアルタイム達成への道

### 1. デバイス変更 (最も確実)
Samsung S24/S25+ (Snapdragon 8 Gen3/Elite, Adreno 750/830)
- cl_khr_gl_sharing対応 → GL buffer zero-copy可
- Adreno GPUはViT推論がMaliより高速
- 公式サンプルで17ms (セグメンテーション) の実績

### 2. NCNN + Vulkan (Pixel 8aで試す価値あり)
- Vulkan computeはOpenCLと異なるGPUパイプライン
- DepthAnything V2のAndroidデモ既存 (ncnn-android-depth_anything)
- Mali-G715のVulkan性能はOpenCLと異なる可能性

### 3. モデル軽量化
- ViT-Tiny (5Mパラメータ, ViT-Sの1/5)
- MobileNetベースの深度推定モデル
- INT8量子化 (FP16精度問題を回避)

### 4. クライアント確認
- 「FP32でiOSの2倍程度」の詳細: デバイス、計測方法、モデル
- dispatch時間(7ms)と実計算時間(345ms)の混同の可能性

## 技術スタック

```
LiteRT: v2.1.3 (最新)
C++ SDK: litert_cc_sdk.zip (FP32 + OpenCLTexture対応)
モデル: depth_anything_v2_keras.tflite (518x518, 99MB, NHWC)
デバイス: Pixel 8a (Tensor G3, Mali-G715, Android 15/16)
```

## ファイル

### C++ SDK パイプライン
- `cpp/depth_pipeline_v2.cpp` — 公式パターンベース、FP32 + OpenCLTexture
- `cpp/litert_sdk/litert_cc_sdk/` — LiteRT C++ SDK v2.1.3
- `cpp/CMakeLists.txt` — SDK統合ビルド

### Kotlin
- `DepthEstimator.kt` — V2 CompiledModel (FP32, readFloat 340ms)
- `InterpreterDepthEstimator.kt.v1-only` — V1 Interpreter (GPU非対応)
- `DepthGLSurfaceView.kt` — GLSurfaceView (表示不安定)

### レガシー (dlopen方式)
- `cpp/depth_pipeline.cpp` — dlopen + 手動struct (参考用)
- `cpp/litert_c_api.h` — C API型定義 (参考用)

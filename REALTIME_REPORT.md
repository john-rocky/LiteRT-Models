# DepthAnything V2 Android Real-time Performance Report

## Current State

### Benchmark (single image, no camera)

| Stage | Time |
|-------|------|
| model.run() (GPU dispatch) | 6-9ms |
| **readFloat() (GPU→CPU transfer)** | **321ms** |

benchmark appの計測は`model.run()`のみ。`readFloat()`を含めた実測は**~330ms/frame**。

### Real-time camera pipeline (measured)

```
camera->bitmap:  169ms
preprocess:      104ms  (resize + ImageNet normalize)
inference:        38ms  (GPU dispatch + sync)
postprocess:     355ms  (readFloat=321ms, colormap=28ms, draw=5ms)
───────────────────────
total:           ~666ms  = ~1.5 FPS
```

## Root Cause

**CompiledModel API (LiteRT 2.x) の`TensorBuffer.readFloat()`が遅い。**

- `readFloat()`はJNI経由でGPU完了待ち + GPU→CPU転送 + Java配列確保・コピーを行う
- 518x518 = 268,324 floatsの読み出しに321ms
- ByteBuffer直接アクセスやpre-allocateバッファへの読み込みはAPIに存在しない
- `runAsync`も存在しない

ベンチマークの「9ms」は`model.run()`（GPUへのdispatch）だけの計測で、出力読み出しを含んでいなかった。

## Comparison with YOLO real-time app

YOLO Flutter appは旧Interpreter API + GpuDelegateを使用:

| | CompiledModel (ML Drift) | 旧Interpreter + GpuDelegate |
|---|---|---|
| Inference API | `model.run(in, out)` + `out.readFloat()` | `interpreter.run(input, output)` |
| Output読み出し | JNI経由で新規float[]確保 (321ms) | pre-allocateした配列に直接書き込み (~0ms) |
| GPU engine | ML Drift (高速推論) | 旧GPU delegate (低速推論) |
| DepthAnything推論 | 6-9ms dispatch | 575ms |
| 合計 | ~330ms | ~575ms |

どちらも real-time (30FPS) には届かない。

## Options

### A. 出力解像度を下げたモデルを作成

headのupsample(→518x518)を省略し、小さい出力で`readFloat()`を高速化。

| 出力 | float数 | read推定 | 合計推定 | FPS |
|------|---------|---------|---------|-----|
| 518x518 | 268K | 321ms | 666ms | 1.5 |
| 296x296 | 88K | ~105ms | ~260ms | ~4 |
| 148x148 | 22K | ~26ms | ~50ms | ~20 |

148x148出力でも表示時にbilinear拡大するため、depth overlayとしては十分な品質。内部計算はフル解像度のまま。

### B. 旧Interpreter API + GPU delegate

575ms inference、output readは高速。合計~600ms。Aより遅い。

### C. CompiledModel API改善を待つ

GoogleがTensorBufferにByteBuffer直接アクセスやrunAsyncを追加すれば解決するが、Googleは却下済み（issue #3006, "does not align with our immediate plans"）。

### D. LiteRT C++ NDK + OpenGL SSBO (本質的解決策)

C++ APIにはKotlinにない zero-copy機能がある:
- `TensorBuffer::CreateFromGlBuffer()` — OpenGL SSBOをzero-copyで出力バッファに
- `RunAsync()` — 非同期推論
- GPU内で推論→colormap→描画を完結、CPU読み戻しゼロ

```
Camera → OpenGL texture
  → LiteRT C++ RunAsync (input SSBO → output SSBO)  // 6ms
  → GLSL compute shader (Inferno colormap)            // <1ms
  → GLSurfaceView render                              // <1ms
= 合計 ~8ms (120+ FPS理論値)
```

prebuilt C++ SDKあり（ソースビルド不要）。JNIブリッジ + OpenGL compute shader実装が必要。工数: 1-2週間。

## References

- [LiteRT Issue #3176 - TensorBuffer read/write really slow](https://github.com/google-ai-edge/LiteRT/issues/3176)
- [LiteRT Issue #3006 - Zero-copy for Kotlin (rejected)](https://github.com/google-ai-edge/LiteRT/issues/3006)
- [LiteRT C++ SDK](https://ai.google.dev/edge/litert/next/android_cpp_sdk)
- [LiteRT GPU zero-copy docs](https://ai.google.dev/edge/litert/next/gpu)

# Depth Pipeline Status

## Current: Kotlin DepthEstimator + ImageView (working baseline)

```
Camera (CameraX) → CPU preprocess (17ms)
  → LiteRT CompiledModel GPU inference (9ms)
  → readFloat() GPU→CPU transfer (320ms) ← BOTTLENECK
  → CPU Inferno colormap (5ms)
  → ImageView display
  = ~350ms/frame = ~3 FPS
```

## What was tried for C++ NDK zero-copy pipeline

### ✅ Working
- LiteRT C API via `dlopen("libLiteRt.so")` — all symbols loaded
- `CreateManagedTensorBufferFromRequirements` — correct buffer creation
- GPU inference via C API — correct depth values (when FP32)
- CPU colormap (Inferno LUT) in C++

### ❌ Blockers

#### 1. GLSurfaceView rendering broken by LiteRT
- LiteRT `GpuEnvironmentCreate` / `CreateCompiledModel` corrupt GLSurfaceView's EGL context
- `eglMakeCurrent` restore doesn't fully fix rendering
- Shared EGL context creation on GL thread also breaks GLSurfaceView
- `setZOrderOnTop(true)` + TRANSLUCENT showed depth intermittently (+ white flicker)
- Tested: OPAQUE, TRANSLUCENT, shared context, independent context, save/restore — none fully work

#### 2. SSBO zero-copy inference
- `CreateTensorBufferFromGlBuffer` succeeds
- `RunCompiledModel` with GL buffer tensors returns **status=3** (runtime failure)

#### 3. FP32 precision
- LiteRT 2.1.3 doesn't export `LrtCreateGpuOptions` / `LrtSetGpuAcceleratorCompilationOptionsPrecision`
- Independent EGL context on bg thread → FP16 → ViT attention overflow → output=0.088684 (garbage)
- Shared EGL context from GLSurfaceView → FP32 → correct output, but breaks GL rendering
- Kotlin CompiledModel API handles FP32 correctly via `GpuOptions.Precision.FP32`

### Recommended next steps for real-time
1. **AHardwareBuffer approach** — `LiteRtCreateTensorBufferFromAhwb` (symbol exists in libLiteRt.so)
2. **Legacy TFLite Interpreter + GPU delegate** — `TfLiteGpuDelegateBindBufferToTensor` for SSBO binding (proven in MediaPipe)
3. **TextureView instead of GLSurfaceView** — integrates into view hierarchy, no Z-order issues

## Device
Pixel 8a — Tensor G3, Mali-G715, Android 15, LiteRT 2.1.3

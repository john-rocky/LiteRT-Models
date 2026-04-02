# Depth Pipeline — Full Technical Status

## Repository
`sample-app/` in this repo. Pixel 8a (Tensor G3, Mali-G715, Android 15/16, LiteRT 2.1.3).

---

## Current Working Setup (3fps)

```
Camera (CameraX RGBA_8888, 1080x1080, rot=90)
  → CPU preprocess: resize 518x518, ImageNet normalize (17ms)
  → Kotlin CompiledModel.run() GPU inference (9ms)
  → readFloat() GPU→CPU transfer (320ms) ← 97% of frame time
  → CPU Inferno colormap (5ms)
  → ImageView display
  = ~350ms/frame = ~3 FPS
```

### Key files
- `MainActivity.kt` — CameraX + ImageView, no Compose
- `DepthEstimator.kt` — Kotlin LiteRT CompiledModel API, FP32 GPU
- `Colormap.kt` — 256-entry Inferno LUT
- `depth_anything_v2_keras.tflite` — 518x518 NHWC native Keras model (99MB, corr=0.9995)

### Why 3fps
`readFloat()` = 320ms. This is GPU→CPU DMA transfer of 518×518×4=1MB.
The Kotlin `TensorBuffer.readFloat()` API has no zero-copy alternative (Google rejected: LiteRT issue #3006).

---

## C++ NDK Pipeline Attempt (experimental, unused)

### Goal
Eliminate `readFloat()` by keeping depth output on GPU:
```
Camera → SSBO → LiteRT GPU inference → SSBO → compute shader colormap → GLSurfaceView
= target ~10ms/frame = 100+ FPS
```

### Files (in repo, not active)
- `app/src/main/cpp/litert_c_api.h` — LiteRT C API type declarations + dlsym loader
- `app/src/main/cpp/depth_pipeline.cpp` — Full C++ pipeline (JNI, EGL, LiteRT, shaders)
- `app/src/main/cpp/CMakeLists.txt` — NDK build config (GLESv3, EGL, dl)
- `NativeDepthPipeline.kt` — JNI bridge class
- `DepthGLSurfaceView.kt` — GLSurfaceView + Renderer
- `build.gradle.kts` — has NDK/CMake config (externalNativeBuild)

### What works in C++
1. **LiteRT C API via dlopen** — `dlopen("libLiteRt.so", RTLD_NOW)` loads runtime from Maven AAR at link-free. All core symbols found via dlsym.
2. **Tensor buffer creation** — `LiteRtCreateManagedTensorBufferFromRequirements()` correctly creates buffers matching compiled model's requirements.
3. **GPU inference** — `LiteRtRunCompiledModel()` returns status=0 with correct depth values when EGL context has FP32.
4. **CPU colormap** — Inferno LUT in C++, percentile-based normalization (2nd/98th), exponential smoothing.

### Critical bugs / enum values discovered
- **`kLiteRtElementTypeFloat32 = 1`** (not 2). Value 2 = Int32. Caused `CreateManagedTensorBuffer` to return "invalid argument".
- **`LiteRtEnvOption` struct**: `{ LiteRtEnvOptionTag tag; union { void* ptr; uint64_t u64; } value; }` — option tags include `kLiteRtEnvOptionTagEglDisplay=14`, `kLiteRtEnvOptionTagEglContext=15`.
- **`LiteRtCreateManagedTensorBufferFromRequirements`** takes 4 params: `(env, tensor_type*, requirements, buffer*)` — NOT 2.

### Unresolved Blockers

#### Blocker 1: GLSurfaceView + LiteRT EGL Conflict
**Problem:** `LiteRtGpuEnvironmentCreate()` and `LiteRtCreateCompiledModel()` permanently corrupt GLSurfaceView's EGL context. Even `eglMakeCurrent` restore doesn't fix rendering.

**Tested combinations (all failed for rendering):**

| Setup | Result |
|---|---|
| LiteRT init on GL thread, `eglMakeCurrent` restore | Black screen (EGL corrupted) |
| LiteRT init on GL thread + TRANSLUCENT + setZOrderOnTop | Intermittent depth + white flicker |
| LiteRT init on bg thread (shared EGL context) | Correct inference but shared context creation breaks GLSurfaceView |
| LiteRT init on bg thread (independent EGL context) | GL rendering works but inference outputs garbage (FP16) |
| Diagnostic: skip all LiteRT init, just glClear(red) | **Red shows** — confirms GLSurfaceView works when LiteRT doesn't touch EGL |

**Root cause:** LiteRT's GPU environment creates its own EGL context/display and makes it current. This side-effects GLSurfaceView's internal EGL management even after restore. No workaround found with GLSurfaceView.

**Potential solutions not tried:**
- TextureView (integrates into view hierarchy, no separate surface)
- Manual EGL + SurfaceView (full control, like terryky/android_tflite NativeActivity approach)
- MediaPipe's `GlCalculatorHelper` pattern

#### Blocker 2: SSBO Zero-Copy Inference (status=3)
**Problem:** `LiteRtRunCompiledModel()` with GL buffer tensor buffers returns `kLiteRtStatusErrorRuntimeFailure` (3).

```cpp
// This succeeds:
LiteRtCreateTensorBufferFromGlBuffer(env, &type, GL_SHADER_STORAGE_BUFFER, ssbo_id, size, 0, nullptr, &buffer);

// This fails with status=3:
LiteRtRunCompiledModel(compiled_model, 0, 1, &input_buf, 1, &output_buf);
```

**Possible causes:**
- LiteRT 2.1.3 on Tensor G3 doesn't support GL buffer I/O for this model
- CL-GL interop not properly initialized
- The model's internal representation incompatible with GL buffers

**Alternative approaches not tried:**
- `LiteRtCreateTensorBufferFromAhwb()` — AHardwareBuffer (symbol exists in libLiteRt.so)
- Legacy TFLite Interpreter API + `TfLiteGpuDelegateBindBufferToTensor()` (proven in MediaPipe/terryky samples, must bind BEFORE `ModifyGraphWithDelegate`)

#### Blocker 3: FP32 Precision on Background Thread
**Problem:** LiteRT v2.1.3 doesn't export `LrtCreateGpuOptions` / `LrtSetGpuAcceleratorCompilationOptionsPrecision`. Cannot set FP32 via C API.

**Behavior:**
- GLSurfaceView's EGL context → GPU defaults to FP32 → correct output (4.19, 3.78, etc.)
- Independent EGL context (bg thread) → GPU defaults to FP16 → ViT attention overflow → constant output 0.088684
- Shared EGL context from GLSurfaceView → FP32 works BUT breaks GLSurfaceView rendering

**Why FP32 matters:** DepthAnything V2 ViT-S has 6-head attention with 64-dim heads. FP16 accumulation overflows at ~26 bits, producing garbage depth output. Documented in INVESTIGATION.md.

---

## Symbol Table (libLiteRt.so v2.1.3, arm64-v8a)

### Available (confirmed via llvm-nm)
```
LiteRtCreateEnvironment, LiteRtDestroyEnvironment
LiteRtGpuEnvironmentCreate
LiteRtCreateModelFromFile, LiteRtCreateModelFromBuffer, LiteRtDestroyModel
LiteRtCreateOptions, LiteRtDestroyOptions, LiteRtSetOptionsHardwareAccelerators
LiteRtAddOpaqueOptions
LiteRtCreateCompiledModel, LiteRtDestroyCompiledModel
LiteRtRunCompiledModel, LiteRtRunCompiledModelAsync
LiteRtGetCompiledModelInputBufferRequirements, LiteRtGetCompiledModelOutputBufferRequirements
LiteRtCreateManagedTensorBuffer, LiteRtCreateManagedTensorBufferFromRequirements
LiteRtCreateTensorBufferFromGlBuffer, LiteRtCreateTensorBufferFromAhwb
LiteRtDestroyTensorBuffer, LiteRtLockTensorBuffer, LiteRtUnlockTensorBuffer
LiteRtGetTensorBufferRequirementsBufferSize
```

### NOT available (dlsym fails)
```
LrtCreateGpuOptions — needed for FP32 precision setting
LrtDestroyGpuOptions
LrtSetGpuAcceleratorCompilationOptionsPrecision
```

### Also present (legacy TFLite C API)
```
TfLiteInterpreterCreate, TfLiteInterpreterGetInputTensor, etc.
```

---

## Research: Proven Real-time Approaches

### 1. Legacy TFLite Interpreter + GL Delegate SSBO Binding
```cpp
TfLiteDelegate* delegate = TfLiteGpuDelegateCreate(&options);
TfLiteGpuDelegateBindBufferToTensor(delegate, ssbo_id, tensor_index);  // BEFORE ModifyGraphWithDelegate
interpreter->ModifyGraphWithDelegate(delegate);
// Now Invoke() reads/writes SSBOs directly
```
**Source:** terryky/android_tflite, MediaPipe tflite_inference_calculator.cc

### 2. MediaPipe TFLiteGPURunner
```cpp
builder->SetInputObjectDef(i, GetSSBOObjectDef(channels));
builder->SetOutputObjectDef(i, GetSSBOObjectDef(channels));
builder->Build(&runner_);
runner_->SetInputObject(0, OpenGlBuffer{ssbo_id});
runner_->Run();
```

### 3. AHardwareBuffer → GL SSBO Bridge
```cpp
AHardwareBuffer_allocate(&desc, &ahwb);  // BLOB format, GPU_DATA_BUFFER usage
EGLClientBuffer native = eglGetNativeClientBufferANDROID(ahwb);
glBufferStorageExternalEXT(GL_SHADER_STORAGE_BUFFER, 0, bytes, native, flags);
```
**Source:** tensorflow/lite/delegates/gpu/async_buffers.cc

### Key Rule
> `ModifyGraphWithDelegate()` and all `Invoke()`/`Run()` calls must happen from the **same EGL context thread**.

---

## Model Details
- **File:** `depth_anything_v2_keras.tflite` (99MB)
- **Input:** [1, 518, 518, 3] NHWC Float32, ImageNet normalized (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- **Output:** [1, 518, 518, 1] Float32 (relative depth, higher=closer)
- **Architecture:** DINOv2 ViT-S/14 backbone + DPT neck + conv head, ReLU output
- **Conversion:** Native Keras (not onnx2tf), corr=0.9995 vs PyTorch
- **GPU:** 9ms inference on Pixel 8a Tensor G3 (ML Drift)
- **FP16 incompatible:** ViT attention overflow at FP16 precision

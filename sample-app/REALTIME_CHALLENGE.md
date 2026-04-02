# DepthAnything V2 リアルタイム化の課題

## 現状
- GPU推論: **7-10ms** (Tensor G3 ML Drift, 609/609ノードGPU実行)
- readFloat: **230-340ms** (GPU→CPUデータ転送)
- 合計: **~250ms = 4fps** (392x518モデル)

推論自体は十分速い。**readFloat()のGPU→CPU転送が全体の92%**を占める。

## 試したこと

### 1. C++ NDK SSBO zero-copy
GPU上のSSBO(OpenGL Shader Storage Buffer)に推論結果を直接書き込み、compute shaderでレンダリング。readFloat不要。
- **結果**: `LiteRtCreateTensorBufferFromGlBuffer`でバッファ作成は成功(status=0)するが、`RunCompiledModel`がstatus=3(runtime failure)を返す。
- **原因**: LiteRT 2.1.3のOpenCLデリゲートがGL bufferテンソルI/Oをサポートしていない。

### 2. AHardwareBuffer zero-copy
GL bufferの代わりにAndroid HardwareBufferを使用。
- **結果**: `LiteRtCreateTensorBufferFromAhwb`でバッファ作成は成功(status=0)するが、同様に`RunCompiledModel`がstatus=3。
- **原因**: 外部提供バッファ全般が`RunCompiledModel`で非対応。

### 3. C++からKotlin TensorBufferを直接Lock
Kotlin `readFloat()`のJNIオーバーヘッドを回避するため、C++から`LiteRtLockTensorBuffer`を直接呼び出し。
- **結果**: status=3。KotlinのTensorBufferハンドルはJNIラッパーで、C APIのraw pointerと互換性がない。
- **副次的発見**: readFloat無しで`run()`を呼ぶと294msブロック → **readFloatの340msはJNIオーバーヘッドではなくGPU stall+DMA転送**と確認。

### 4. Kotlin CompiledModelハンドルをC++で使用
Kotlin CompiledModel(FP32)のnativeハンドルを抽出し、C++の`RunCompiledModel`に渡す。
- **結果**: クラッシュ。KotlinのCompiledModelハンドルはJNIラッパーで、C APIの`LiteRtCompiledModel`と型が異なる。

### 5. C++からKotlinの`nativeRun` JNIメソッドを呼ぶ
Kotlin CompiledModel(FP32)のnativeRunをC++からリフレクションで呼び、C++で作成したテンソルバッファを渡す。
- **結果**: "Output buffers cannot have events attached"エラー。C++で作成したバッファにGPUイベントが付与されており、KotlinのnativeRunが拒否。
- `ClearTensorBufferEvent`を呼んでも解決せず。

### 6. EGLコンテキスト共有
GLSurfaceViewのEGLコンテキストをLiteRTに渡し、同一GPU上でSSBO共有。
- **結果**: `LiteRtEnvOption`のABI修正(24バイト構造体、タグ値6/7)後、`GpuEnvironmentCreate`に渡すことで"Reusing provided EGL environment"に成功。
- ただしSSBO `RunCompiledModel`は依然status=3。

### 7. FP32精度設定
C++ APIでは`LrtCreateGpuOptions`/`LrtSetGpuAcceleratorCompilationOptionsPrecision`が未エクスポート(v2.1.3)。FP32を設定できず、FP16デフォルト → ViTアテンションオーバーフロー → 複雑シーンで出力がゴミ(0.088684固定値)。
- Kotlin APIのみ`GpuOptions.Precision.FP32`を設定可能。

### 8. GLSurfaceView + LiteRT EGLコンテキスト競合
LiteRTのGPU初期化(`GpuEnvironmentCreate`/`CreateCompiledModel`)がGLSurfaceViewのEGLコンテキストを破壊。
- EGL save/restore、別スレッド実行、共有コンテキスト等を試すも、安定した表示を実現できず。
- **唯一の安定表示**: ImageView + Bitmap(従来方式)。

### 9. `run(inputBuffers)`方式
`compiledModel.run(inputBuffers)`で内部最適化されたoutputバッファを返す方式をテスト。
- **結果**: readFloat時間は変わらず(228ms)。

## 確認済み事実
- LiteRT **v2.1.3が最新版** (2026-03-17リリース)
- Tensor G3のML Drift delegateは**GELU/BATCH_MATMUL/LAYER_NORM対応**(609/609ノード全GPU)
- readFloat 340msは**出力サイズに比例**: 518x518→340ms、392x518→230ms
- `TensorBufferType`にはOpenClBuffer, GlBuffer, Ahwb等のenum値が定義されているが、**Kotlin APIからこれらの型でバッファを作成する方法がない**
- readFloat以外の出力取得メソッドはKotlin APIに**存在しない**

## 未解決
- クライアントが「FP32でiOSの2倍程度で動いていた」と報告 → readFloat 340msを回避する方法が存在する可能性
- `readFloat`内部の時間内訳(GPU待ち vs DMA vs Java配列確保)が不明

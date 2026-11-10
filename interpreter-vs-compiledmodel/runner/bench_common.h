// bench_common.h: the interface between ftfl_bench.cc (arguments, fixtures, timing, dumps) and the two runtime paths
// (tfl_path.cc = the TensorFlow Lite C API that react-native-fast-tflite calls, on LiteRT 1.4.0's
// libtensorflowlite_jni.so; lrt_path.cc = LiteRT 2.x CompiledModel on libLiteRt.so). Each path dlopens its own
// runtime library, so one process holds one runtime. Element type codes follow TfLiteType, which LiteRtElementType
// mirrors (1 float32, 2 int32, 3 uint8, 4 int64, 6 bool, 7 int16, 9 int8, 10 float16, 11 float64).
#ifndef FTFL_BENCH_COMMON_H_
#define FTFL_BENCH_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct TensorInfo {
  std::string name;
  int dtype = 0;
  size_t bytes = 0;
  std::vector<int> dims;
};

struct LoadOpts {
  std::string model_path;
  std::string libdir;         // the stage: runtime libraries, accelerators, dispatch and compiler plugin
  std::string cache_dir;      // NPU JIT cache (LiteRT path only), empty = none
  std::string accel;          // cpu | gpu | npu | npu+cpu (tflite: npu = the NNAPI delegate)
  std::string gpu_precision;  // default | fp32 | fp16 | fp16acc32
  std::string tflite_lib;     // tflite path: the library that carries the TFLite C API
  std::string tflite_gpu_lib; // tflite path: the GPU delegate library
  int cpu_threads = 0;        // 0 = not set: the runtime's own default
  int qnn_burst = 0;          // litert path, NPU: 1 = the "qualcomm" payload htp_performance_mode = 2 (burst)
  int cpu_fallback = 0;       // litert path, GPU: 1 = accelerators GPU | CPU (the ops the GPU does not take run on the
                              // CPU, as the TFLite GPU delegate leaves them to the interpreter); 0 = GPU only
  int xnnpack = 0;            // tflite path, CPU: 1 = add the XNNPACK delegate (TfLiteXNNPackDelegateCreate) to the
                              // interpreter options; num_threads = --cpu-threads when set, else the library's default
  int host_buffers = 0;       // litert path: 1 = host-memory tensor buffers of each tensor's own size
                              // (LiteRtCreateManagedTensorBuffer) instead of the compiled model's buffer requirements
};

class Backend {
 public:
  virtual ~Backend() = default;
  // Model, options, accelerator, compile / interpreter, tensors or buffers. False with *err on any failure.
  virtual bool Load(const LoadOpts& o, std::string* err) = 0;
  virtual const std::vector<TensorInfo>& Inputs() const = 0;
  // In the primary subgraph's output order, so both paths list the same tensors in the same order.
  virtual const std::vector<TensorInfo>& Outputs() const = 0;
  virtual bool Write(const std::vector<std::vector<uint8_t>>& in, std::string* err) = 0;
  virtual bool Run(std::string* err) = 0;
  // Raw bytes of every output into the caller's buffers (resized once), in Outputs() order.
  virtual bool Read(std::vector<std::vector<uint8_t>>* out, std::string* err) = 0;
  // One line of facts about the backend (library, accelerator, delegation), printed after the load.
  virtual std::string Info() const = 0;
};

Backend* MakeTfliteBackend();
Backend* MakeLiteRtBackend();

#endif  // FTFL_BENCH_COMMON_H_

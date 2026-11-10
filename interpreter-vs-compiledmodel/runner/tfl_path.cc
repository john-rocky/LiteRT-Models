// tfl_path.cc: the TensorFlow Lite C API path, called the way react-native-fast-tflite 3.0.1 calls it on Android
// (cpp/HybridTfliteModule.cpp, cpp/HybridTfliteModel.cpp, cpp/TfliteHelpers.cpp at feeb7677):
//   TfLiteModelCreate(bytes) -> TfLiteInterpreterOptionsCreate -> [TfLiteInterpreterOptionsAddDelegate] ->
//   TfLiteInterpreterCreate -> TfLiteInterpreterAllocateTensors; per call TfLiteTensorCopyFromBuffer ->
//   TfLiteInterpreterInvoke -> memcpy(TfLiteTensorData).
// fast-tflite sets no thread count; --cpu-threads N adds TfLiteInterpreterOptionsSetNumThreads (a column of its own).
// Delegates as fast-tflite builds them: GPU = TfLiteGpuDelegateV2Create(TfLiteGpuDelegateOptionsV2Default()),
// NNAPI = TfLiteNnapiDelegateCreate(TfLiteNnapiDelegateOptionsDefault()). --gpu-precision fp16 changes the GPU
// options to priority1 = MIN_LATENCY with precision loss allowed (a column of its own, not fast-tflite's call).
// The library is dlopened (RTLD_LOCAL), never linked: libtensorflowlite_jni.so of LiteRT 1.4.0 by default, or any
// library that exports the same C API (--tflite-lib libLiteRt.so: LiteRT 2.x behind the same calls, CPU only).
// Headers: LiteRT 1.4.0 AAR headers/ (types and prototypes only; every call goes through a dlsym'd pointer).
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "bench_common.h"
#include "tflite/c/c_api.h"
#include "tflite/delegates/gpu/delegate.h"
#include "tflite/delegates/nnapi/nnapi_delegate_c_api.h"

namespace {

#define TFL_FN(name) decltype(&::name) name = nullptr;
struct TflApi {
  TFL_FN(TfLiteVersion)
  TFL_FN(TfLiteModelCreate)
  TFL_FN(TfLiteModelDelete)
  TFL_FN(TfLiteInterpreterOptionsCreate)
  TFL_FN(TfLiteInterpreterOptionsDelete)
  TFL_FN(TfLiteInterpreterOptionsSetNumThreads)
  TFL_FN(TfLiteInterpreterOptionsAddDelegate)
  TFL_FN(TfLiteInterpreterCreate)
  TFL_FN(TfLiteInterpreterDelete)
  TFL_FN(TfLiteInterpreterAllocateTensors)
  TFL_FN(TfLiteInterpreterGetInputTensorCount)
  TFL_FN(TfLiteInterpreterGetInputTensor)
  TFL_FN(TfLiteInterpreterGetOutputTensorCount)
  TFL_FN(TfLiteInterpreterGetOutputTensor)
  TFL_FN(TfLiteInterpreterInvoke)
  TFL_FN(TfLiteTensorType)
  TFL_FN(TfLiteTensorNumDims)
  TFL_FN(TfLiteTensorDim)
  TFL_FN(TfLiteTensorByteSize)
  TFL_FN(TfLiteTensorData)
  TFL_FN(TfLiteTensorName)
  TFL_FN(TfLiteTensorCopyFromBuffer)
  // the NNAPI delegate lives in the same library as the C API (1.4.0); absent from libLiteRt.so
  TFL_FN(TfLiteNnapiDelegateOptionsDefault)
  TFL_FN(TfLiteNnapiDelegateCreate)
  TFL_FN(TfLiteNnapiDelegateDelete)
};
struct GpuApi {
  TFL_FN(TfLiteGpuDelegateOptionsV2Default)
  TFL_FN(TfLiteGpuDelegateV2Create)
  TFL_FN(TfLiteGpuDelegateV2Delete)
};
#undef TFL_FN

// The XNNPACK delegate's options struct changes between TFLite versions (later sources put runtime_flags right after
// num_threads) and the 1.4.0 AAR ships no header for it. So the options live in a buffer larger than any version's
// struct, TfLiteXNNPackDelegateOptionsDefault() fills it, and only num_threads (offset 0 in every version) is written.
// On AArch64 a struct over 16 bytes is returned through the caller's buffer (x8), so the larger declared return type
// receives the library's smaller struct; the bytes past it are never read by TfLiteXNNPackDelegateCreate.
struct XnnOptsBuf { alignas(16) unsigned char b[512]; };
typedef XnnOptsBuf (*XnnDefaultFn)();
typedef TfLiteDelegate* (*XnnCreateFn)(const void*);
typedef void (*XnnDeleteFn)(TfLiteDelegate*);

template <typename T>
bool Sym(void* h, const char* name, T* out, std::string* missing) {
  *out = reinterpret_cast<T>(::dlsym(h, name));
  if (*out == nullptr) { if (missing) *missing += std::string(missing->empty() ? "" : ",") + name; return false; }
  return true;
}

std::string LibPath(const std::string& dir, const std::string& name) {
  return name.find('/') != std::string::npos || dir.empty() ? name : dir + "/" + name;
}

class TfliteBackend : public Backend {
 public:
  ~TfliteBackend() override {
    if (interp_ && api_.TfLiteInterpreterDelete) api_.TfLiteInterpreterDelete(interp_);
    if (xnn_delegate_ && xnn_delete_) xnn_delete_(xnn_delegate_);
    if (gpu_delegate_ && gpu_.TfLiteGpuDelegateV2Delete) gpu_.TfLiteGpuDelegateV2Delete(gpu_delegate_);
    if (nnapi_delegate_ && api_.TfLiteNnapiDelegateDelete) api_.TfLiteNnapiDelegateDelete(nnapi_delegate_);
    if (model_ && api_.TfLiteModelDelete) api_.TfLiteModelDelete(model_);
    // the libraries stay loaded until the process ends (a GPU driver thread may still hold code in them)
  }

  bool Load(const LoadOpts& o, std::string* err) override {
    lib_name_ = o.tflite_lib.empty() ? "libtensorflowlite_jni.so" : o.tflite_lib;
    std::string path = LibPath(o.libdir, lib_name_);
    lib_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib_) { *err = "dlopen " + path + ": " + ::dlerror(); return false; }
    std::string missing;
    bool ok = true;
    ok &= Sym(lib_, "TfLiteVersion", &api_.TfLiteVersion, &missing);
    ok &= Sym(lib_, "TfLiteModelCreate", &api_.TfLiteModelCreate, &missing);
    ok &= Sym(lib_, "TfLiteModelDelete", &api_.TfLiteModelDelete, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterOptionsCreate", &api_.TfLiteInterpreterOptionsCreate, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterOptionsDelete", &api_.TfLiteInterpreterOptionsDelete, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterOptionsSetNumThreads", &api_.TfLiteInterpreterOptionsSetNumThreads, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterOptionsAddDelegate", &api_.TfLiteInterpreterOptionsAddDelegate, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterCreate", &api_.TfLiteInterpreterCreate, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterDelete", &api_.TfLiteInterpreterDelete, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterAllocateTensors", &api_.TfLiteInterpreterAllocateTensors, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterGetInputTensorCount", &api_.TfLiteInterpreterGetInputTensorCount, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterGetInputTensor", &api_.TfLiteInterpreterGetInputTensor, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterGetOutputTensorCount", &api_.TfLiteInterpreterGetOutputTensorCount, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterGetOutputTensor", &api_.TfLiteInterpreterGetOutputTensor, &missing);
    ok &= Sym(lib_, "TfLiteInterpreterInvoke", &api_.TfLiteInterpreterInvoke, &missing);
    ok &= Sym(lib_, "TfLiteTensorType", &api_.TfLiteTensorType, &missing);
    ok &= Sym(lib_, "TfLiteTensorNumDims", &api_.TfLiteTensorNumDims, &missing);
    ok &= Sym(lib_, "TfLiteTensorDim", &api_.TfLiteTensorDim, &missing);
    ok &= Sym(lib_, "TfLiteTensorByteSize", &api_.TfLiteTensorByteSize, &missing);
    ok &= Sym(lib_, "TfLiteTensorData", &api_.TfLiteTensorData, &missing);
    ok &= Sym(lib_, "TfLiteTensorName", &api_.TfLiteTensorName, &missing);
    ok &= Sym(lib_, "TfLiteTensorCopyFromBuffer", &api_.TfLiteTensorCopyFromBuffer, &missing);
    if (!ok) { *err = "missing in " + lib_name_ + ": " + missing; return false; }
    accel_ = o.accel;
    if (o.accel == "npu+cpu") { *err = "accel npu+cpu has no TFLite C API form (use npu = the NNAPI delegate)"; return false; }
    if (o.accel == "npu") {  // fast-tflite's 'nnapi'
      std::string m2;
      bool n_ok = Sym(lib_, "TfLiteNnapiDelegateOptionsDefault", &api_.TfLiteNnapiDelegateOptionsDefault, &m2);
      n_ok = Sym(lib_, "TfLiteNnapiDelegateCreate", &api_.TfLiteNnapiDelegateCreate, &m2) && n_ok;
      n_ok = Sym(lib_, "TfLiteNnapiDelegateDelete", &api_.TfLiteNnapiDelegateDelete, &m2) && n_ok;
      if (!n_ok) { *err = "NNAPI delegate missing in " + lib_name_ + ": " + m2; return false; }
    }
    if (o.accel == "gpu") {
      gpu_lib_name_ = o.tflite_gpu_lib.empty() ? "libtensorflowlite_gpu_jni.so" : o.tflite_gpu_lib;
      std::string gp = LibPath(o.libdir, gpu_lib_name_);
      gpu_lib_ = ::dlopen(gp.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!gpu_lib_) { *err = "dlopen " + gp + ": " + ::dlerror(); return false; }
      std::string m3;
      bool g_ok = Sym(gpu_lib_, "TfLiteGpuDelegateOptionsV2Default", &gpu_.TfLiteGpuDelegateOptionsV2Default, &m3);
      g_ok = Sym(gpu_lib_, "TfLiteGpuDelegateV2Create", &gpu_.TfLiteGpuDelegateV2Create, &m3) && g_ok;
      g_ok = Sym(gpu_lib_, "TfLiteGpuDelegateV2Delete", &gpu_.TfLiteGpuDelegateV2Delete, &m3) && g_ok;
      if (!g_ok) { *err = "GPU delegate missing in " + gpu_lib_name_ + ": " + m3; return false; }
    }
    if (o.xnnpack && o.accel != "cpu") { *err = "--xnnpack 1 is a CPU form (accel " + o.accel + ")"; return false; }
    if (o.host_buffers) { *err = "--host-buffers is a litert path flag"; return false; }
    XnnDefaultFn xnn_default = nullptr;
    if (o.xnnpack) {
      std::string m4;
      bool x_ok = Sym(lib_, "TfLiteXNNPackDelegateOptionsDefault", &xnn_default, &m4);
      x_ok = Sym(lib_, "TfLiteXNNPackDelegateCreate", &xnn_create_, &m4) && x_ok;
      x_ok = Sym(lib_, "TfLiteXNNPackDelegateDelete", &xnn_delete_, &m4) && x_ok;
      if (!x_ok) { *err = "XNNPACK delegate missing in " + lib_name_ + ": " + m4; return false; }
    }
    if (o.accel != "cpu" && o.accel != "gpu" && o.accel != "npu") { *err = "accel " + o.accel; return false; }

    // the model as fast-tflite gets it: the file's bytes in memory, kept alive for the model's life
    FILE* f = std::fopen(o.model_path.c_str(), "rb");
    if (!f) { *err = "open " + o.model_path; return false; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    model_bytes_.resize(sz > 0 ? (size_t)sz : 0);
    bool rd = model_bytes_.empty() || std::fread(model_bytes_.data(), 1, model_bytes_.size(), f) == model_bytes_.size();
    std::fclose(f);
    if (!rd) { *err = "read " + o.model_path; return false; }
    model_ = api_.TfLiteModelCreate(model_bytes_.data(), model_bytes_.size());
    if (!model_) { *err = "TfLiteModelCreate failed"; return false; }
    TfLiteInterpreterOptions* opts = api_.TfLiteInterpreterOptionsCreate();
    if (!opts) { *err = "TfLiteInterpreterOptionsCreate failed"; return false; }
    if (o.cpu_threads > 0) api_.TfLiteInterpreterOptionsSetNumThreads(opts, o.cpu_threads);
    if (o.xnnpack) {
      std::memset(&xnn_opts_, 0, sizeof xnn_opts_);
      xnn_opts_ = xnn_default();
      int32_t nt = 0;
      std::memcpy(&nt, xnn_opts_.b, sizeof nt);
      info_ += " xnnpack_delegate(default_num_threads=" + std::to_string(nt);
      if (o.cpu_threads > 0) { nt = o.cpu_threads; std::memcpy(xnn_opts_.b, &nt, sizeof nt); }
      info_ += ",num_threads=" + std::to_string(nt) + ")";
      xnn_delegate_ = xnn_create_(&xnn_opts_);
      if (!xnn_delegate_) { api_.TfLiteInterpreterOptionsDelete(opts); *err = "TfLiteXNNPackDelegateCreate returned null"; return false; }
      api_.TfLiteInterpreterOptionsAddDelegate(opts, xnn_delegate_);
    }
    if (o.accel == "gpu") {
      TfLiteGpuDelegateOptionsV2 g = gpu_.TfLiteGpuDelegateOptionsV2Default();
      if (o.gpu_precision == "fp16") {
        g.is_precision_loss_allowed = 1;
        g.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        g.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        g.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
      } else if (o.gpu_precision == "fp32") {
        g.is_precision_loss_allowed = 0;
        g.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
      } else if (o.gpu_precision != "default") {
        api_.TfLiteInterpreterOptionsDelete(opts);
        *err = "gpu precision " + o.gpu_precision + " has no TFLite GPU delegate V2 form";
        return false;
      }
      info_ += " gpu_options(precision_loss_allowed=" + std::to_string(g.is_precision_loss_allowed) +
               ",priority1=" + std::to_string(g.inference_priority1) + ",priority2=" + std::to_string(g.inference_priority2) +
               ",max_partitions=" + std::to_string(g.max_delegated_partitions) +
               ",experimental_flags=" + std::to_string((long long)g.experimental_flags) + ")";
      gpu_delegate_ = gpu_.TfLiteGpuDelegateV2Create(&g);
      if (!gpu_delegate_) { api_.TfLiteInterpreterOptionsDelete(opts); *err = "TfLiteGpuDelegateV2Create returned null"; return false; }
      api_.TfLiteInterpreterOptionsAddDelegate(opts, gpu_delegate_);
    } else if (o.accel == "npu") {
      TfLiteNnapiDelegateOptions n = api_.TfLiteNnapiDelegateOptionsDefault();
      info_ += std::string(" nnapi_options(accelerator_name=") + (n.accelerator_name ? n.accelerator_name : "(none)") +
               ",execution_preference=" + std::to_string((int)n.execution_preference) + ")";
      nnapi_delegate_ = api_.TfLiteNnapiDelegateCreate(&n);
      if (!nnapi_delegate_) { api_.TfLiteInterpreterOptionsDelete(opts); *err = "TfLiteNnapiDelegateCreate returned null"; return false; }
      api_.TfLiteInterpreterOptionsAddDelegate(opts, nnapi_delegate_);
    }
    interp_ = api_.TfLiteInterpreterCreate(model_, opts);
    api_.TfLiteInterpreterOptionsDelete(opts);
    if (!interp_) { *err = "TfLiteInterpreterCreate returned null (fast-tflite: \"Failed to create TFLite interpreter!\")"; return false; }
    TfLiteStatus st = api_.TfLiteInterpreterAllocateTensors(interp_);
    if (st != kTfLiteOk) { *err = "TfLiteInterpreterAllocateTensors status " + std::to_string((int)st); return false; }
    int32_t n_in = api_.TfLiteInterpreterGetInputTensorCount(interp_);
    for (int32_t i = 0; i < n_in; ++i) {
      TfLiteTensor* t = api_.TfLiteInterpreterGetInputTensor(interp_, i);
      if (!t) { *err = "input tensor " + std::to_string(i) + " is null"; return false; }
      in_t_.push_back(t);
      inputs_.push_back(Describe(t));
    }
    int32_t n_out = api_.TfLiteInterpreterGetOutputTensorCount(interp_);
    for (int32_t i = 0; i < n_out; ++i) {
      const TfLiteTensor* t = api_.TfLiteInterpreterGetOutputTensor(interp_, i);
      if (!t) { *err = "output tensor " + std::to_string(i) + " is null"; return false; }
      out_t_.push_back(t);
      outputs_.push_back(Describe(t));
    }
    return true;
  }

  const std::vector<TensorInfo>& Inputs() const override { return inputs_; }
  const std::vector<TensorInfo>& Outputs() const override { return outputs_; }

  bool Write(const std::vector<std::vector<uint8_t>>& in, std::string* err) override {
    if (in.size() != in_t_.size()) { *err = "input count"; return false; }
    for (size_t i = 0; i < in.size(); ++i) {
      TfLiteStatus st = api_.TfLiteTensorCopyFromBuffer(in_t_[i], in[i].data(), in[i].size());
      if (st != kTfLiteOk) { *err = "TfLiteTensorCopyFromBuffer input " + std::to_string(i) + " status " + std::to_string((int)st); return false; }
    }
    return true;
  }

  bool Run(std::string* err) override {
    TfLiteStatus st = api_.TfLiteInterpreterInvoke(interp_);
    if (st != kTfLiteOk) { *err = "TfLiteInterpreterInvoke status " + std::to_string((int)st); return false; }
    return true;
  }

  bool Read(std::vector<std::vector<uint8_t>>* out, std::string* err) override {
    out->resize(out_t_.size());
    for (size_t i = 0; i < out_t_.size(); ++i) {
      const void* p = api_.TfLiteTensorData(out_t_[i]);
      if (!p) { *err = "TfLiteTensorData output " + std::to_string(i) + " is null"; return false; }
      (*out)[i].resize(outputs_[i].bytes);
      std::memcpy((*out)[i].data(), p, outputs_[i].bytes);
    }
    return true;
  }

  std::string Info() const override {
    return "runtime=tflite_c_api lib=" + lib_name_ + " version=" + (api_.TfLiteVersion ? api_.TfLiteVersion() : "?") +
           " accel=" + accel_ + (gpu_lib_ ? " gpu_lib=" + gpu_lib_name_ : std::string()) + info_ +
           " delegation=see_log(Replacing N out of M node(s))";
  }

 private:
  TensorInfo Describe(const TfLiteTensor* t) {
    TensorInfo ti;
    const char* nm = api_.TfLiteTensorName(t);
    ti.name = nm ? nm : "";
    ti.dtype = (int)api_.TfLiteTensorType(t);
    ti.bytes = api_.TfLiteTensorByteSize(t);
    int nd = api_.TfLiteTensorNumDims(t);
    for (int d = 0; d < nd; ++d) ti.dims.push_back(api_.TfLiteTensorDim(t, d));
    return ti;
  }

  void* lib_ = nullptr;
  void* gpu_lib_ = nullptr;
  std::string lib_name_, gpu_lib_name_, accel_, info_;
  TflApi api_;
  GpuApi gpu_;
  std::vector<uint8_t> model_bytes_;
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreter* interp_ = nullptr;
  TfLiteDelegate* gpu_delegate_ = nullptr;
  TfLiteDelegate* nnapi_delegate_ = nullptr;
  TfLiteDelegate* xnn_delegate_ = nullptr;
  XnnCreateFn xnn_create_ = nullptr;
  XnnDeleteFn xnn_delete_ = nullptr;
  XnnOptsBuf xnn_opts_;
  std::vector<TfLiteTensor*> in_t_;
  std::vector<const TfLiteTensor*> out_t_;
  std::vector<TensorInfo> inputs_, outputs_;
};

}  // namespace

Backend* MakeTfliteBackend() { return new TfliteBackend(); }

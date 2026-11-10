// lrt_path.cc: the LiteRT 2.x CompiledModel path (C API), the way a wrapper would call it after moving off the
// Interpreter: environment (runtime / dispatch / compiler-plugin dirs = the stage, JIT cache dir for the NPU) ->
// model from the file's bytes -> options (hardware accelerators; "xnnpack" num_threads only with --cpu-threads;
// "gpu_options" precision only when --gpu-precision is not default; "qualcomm" burst only with --qnn-burst 1) ->
// LiteRtCreateCompiledModel -> managed tensor buffers from the compiled model's requirements; per call lock / copy /
// unlock each input, LiteRtRunCompiledModel, lock / copy / unlock each output.
// The CPU options' identifier "xnnpack" is the one of LiteRT v2.2.0-v2.3.2 (litert/c/options/litert_cpu_options.cc:131).
// Every call goes through a dlsym'd pointer (libLiteRt.so is dlopened, never linked); outputs of any element type are
// listed in the primary subgraph's output order (mapped from the signature by tensor name) so they line up with the
// TFLite path. With --host-buffers 1 the buffers are host memory of each tensor's own size instead.
// Outputs whose shape the flatbuffer leaves open (TFLite_Detection_PostProcess) take their layout from
// LiteRtGetCompiledModelOutputTensorLayouts after the compile.
// Headers: LiteRT v2.3.0 litert_cc_sdk.zip (the C API of v2.3.2 is the same: compare v2.3.0...v2.3.2 changes one
// comment in litert/c/internal/litert_runtime_c_api.h).
#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bench_common.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace {

#define LRT_FN(name) decltype(&::name) name = nullptr;
struct LrtApi {
  LRT_FN(LiteRtCreateEnvironment)
  LRT_FN(LiteRtDestroyEnvironment)
  LRT_FN(LiteRtCreateModelFromBuffer)
  LRT_FN(LiteRtDestroyModel)
  LRT_FN(LiteRtGetNumModelSignatures)
  LRT_FN(LiteRtGetModelSignature)
  LRT_FN(LiteRtGetSignatureKey)
  LRT_FN(LiteRtGetSignatureSubgraph)
  LRT_FN(LiteRtGetNumSignatureInputs)
  LRT_FN(LiteRtGetNumSignatureOutputs)
  LRT_FN(LiteRtGetSignatureInputTensorByIndex)
  LRT_FN(LiteRtGetSignatureOutputTensorByIndex)
  LRT_FN(LiteRtGetNumSubgraphOutputs)
  LRT_FN(LiteRtGetSubgraphOutput)
  LRT_FN(LiteRtGetTensorName)
  LRT_FN(LiteRtGetRankedTensorType)
  LRT_FN(LiteRtCreateOptions)
  LRT_FN(LiteRtDestroyOptions)
  LRT_FN(LiteRtSetOptionsHardwareAccelerators)
  LRT_FN(LiteRtCreateOpaqueOptions)
  LRT_FN(LiteRtAddOpaqueOptions)
  LRT_FN(LiteRtCreateCompiledModel)
  LRT_FN(LiteRtDestroyCompiledModel)
  LRT_FN(LiteRtCompiledModelIsFullyAccelerated)
  LRT_FN(LiteRtGetCompiledModelInputBufferRequirements)
  LRT_FN(LiteRtGetCompiledModelOutputBufferRequirements)
  LRT_FN(LiteRtGetCompiledModelOutputTensorLayouts)
  LRT_FN(LiteRtGetTensorBufferRequirementsBufferSize)
  LRT_FN(LiteRtCreateManagedTensorBufferFromRequirements)
  LRT_FN(LiteRtCreateManagedTensorBuffer)
  LRT_FN(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes)
  LRT_FN(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType)
  LRT_FN(LiteRtGetTensorBufferType)
  LRT_FN(LiteRtGetTensorBufferSize)
  LRT_FN(LiteRtDestroyTensorBuffer)
  LRT_FN(LiteRtLockTensorBuffer)
  LRT_FN(LiteRtUnlockTensorBuffer)
  LRT_FN(LiteRtRunCompiledModel)
};
#undef LRT_FN

void DeleteCStringPayload(void* p) { delete[] static_cast<char*>(p); }

size_t ElemBytes(int t) {
  switch (t) {
    case kLiteRtElementTypeFloat32: case kLiteRtElementTypeInt32: case kLiteRtElementTypeUInt32: return 4;
    case kLiteRtElementTypeFloat16: case kLiteRtElementTypeBFloat16: case kLiteRtElementTypeInt16:
    case kLiteRtElementTypeUInt16: return 2;
    case kLiteRtElementTypeInt64: case kLiteRtElementTypeUInt64: case kLiteRtElementTypeFloat64: return 8;
    default: return 1;
  }
}

size_t NumElems(const LiteRtLayout& l) {
  size_t n = 1;
  for (unsigned i = 0; i < l.rank; ++i) n *= (size_t)(l.dimensions[i] > 0 ? l.dimensions[i] : 0);
  return n;
}

#define LRT_OK(expr, what)                                                   \
  do {                                                                       \
    LiteRtStatus _st = (expr);                                               \
    if (_st != kLiteRtStatusOk) { *err = std::string(what) + " status " + std::to_string((int)_st); return false; } \
  } while (0)

class LiteRtBackend : public Backend {
 public:
  ~LiteRtBackend() override {
    if (!lib_) return;
    for (auto b : outs_) if (b) api_.LiteRtDestroyTensorBuffer(b);
    for (auto b : ins_) if (b) api_.LiteRtDestroyTensorBuffer(b);
    if (cm_) api_.LiteRtDestroyCompiledModel(cm_);
    if (opts_) api_.LiteRtDestroyOptions(opts_);
    if (model_) api_.LiteRtDestroyModel(model_);
    if (env_) api_.LiteRtDestroyEnvironment(env_);
  }

  bool Load(const LoadOpts& o, std::string* err) override {
    std::string path = o.libdir.empty() ? "libLiteRt.so" : o.libdir + "/libLiteRt.so";
    lib_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib_) { *err = "dlopen " + path + ": " + ::dlerror(); return false; }
    std::string missing;
#define GET(name) api_.name = reinterpret_cast<decltype(api_.name)>(::dlsym(lib_, #name)); if (!api_.name) missing += std::string(missing.empty() ? "" : ",") + #name;
    GET(LiteRtCreateEnvironment) GET(LiteRtDestroyEnvironment) GET(LiteRtCreateModelFromBuffer) GET(LiteRtDestroyModel)
    GET(LiteRtGetNumModelSignatures) GET(LiteRtGetModelSignature) GET(LiteRtGetSignatureKey) GET(LiteRtGetSignatureSubgraph)
    GET(LiteRtGetNumSignatureInputs) GET(LiteRtGetNumSignatureOutputs) GET(LiteRtGetSignatureInputTensorByIndex)
    GET(LiteRtGetSignatureOutputTensorByIndex) GET(LiteRtGetNumSubgraphOutputs) GET(LiteRtGetSubgraphOutput)
    GET(LiteRtGetTensorName) GET(LiteRtGetRankedTensorType) GET(LiteRtCreateOptions) GET(LiteRtDestroyOptions)
    GET(LiteRtSetOptionsHardwareAccelerators) GET(LiteRtCreateOpaqueOptions) GET(LiteRtAddOpaqueOptions)
    GET(LiteRtCreateCompiledModel) GET(LiteRtDestroyCompiledModel) GET(LiteRtCompiledModelIsFullyAccelerated)
    GET(LiteRtGetCompiledModelInputBufferRequirements) GET(LiteRtGetCompiledModelOutputBufferRequirements)
    GET(LiteRtGetCompiledModelOutputTensorLayouts) GET(LiteRtGetTensorBufferRequirementsBufferSize)
    GET(LiteRtCreateManagedTensorBufferFromRequirements) GET(LiteRtDestroyTensorBuffer) GET(LiteRtLockTensorBuffer)
    GET(LiteRtUnlockTensorBuffer) GET(LiteRtRunCompiledModel)
    GET(LiteRtCreateManagedTensorBuffer) GET(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes)
    GET(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType) GET(LiteRtGetTensorBufferType)
    GET(LiteRtGetTensorBufferSize)
#undef GET
    if (!missing.empty()) { *err = "missing in libLiteRt.so: " + missing; return false; }

    // --- environment: the stage holds the GPU accelerator, the Qualcomm dispatch and compiler plugin, QAIRT ---
    std::vector<LiteRtEnvOption> env_opts;
    auto add_env = [&](LiteRtEnvOptionTag tag, const std::string& v) {
      LiteRtEnvOption e{}; e.tag = tag; e.value.type = kLiteRtAnyTypeString; e.value.str_value = v.c_str(); env_opts.push_back(e);
    };
    add_env(kLiteRtEnvOptionTagRuntimeLibraryDir, o.libdir);
    add_env(kLiteRtEnvOptionTagDispatchLibraryDir, o.libdir);
    add_env(kLiteRtEnvOptionTagCompilerPluginLibraryDir, o.libdir);
    if (!o.cache_dir.empty()) { ::mkdir(o.cache_dir.c_str(), 0775); add_env(kLiteRtEnvOptionTagCompilerCacheDir, o.cache_dir); }
    LRT_OK(api_.LiteRtCreateEnvironment((int)env_opts.size(), env_opts.data(), &env_), "LiteRtCreateEnvironment");

    LiteRtHwAcceleratorSet hw;
    if (o.accel == "cpu") hw = kLiteRtHwAcceleratorCpu;
    else if (o.accel == "gpu") hw = o.cpu_fallback ? (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu)
                                                   : kLiteRtHwAcceleratorGpu;
    else if (o.accel == "npu") hw = kLiteRtHwAcceleratorNpu;
    else if (o.accel == "npu+cpu") hw = (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu);
    else { *err = "accel " + o.accel; return false; }
    if (o.host_buffers && o.accel.rfind("npu", 0) == 0) { *err = "--host-buffers is for cpu / gpu"; return false; }
    accel_ = o.accel;
    info_ += " hw_accelerators=" + std::to_string((int)hw);
    if (o.host_buffers) info_ += " host_buffers=1";

    FILE* f = std::fopen(o.model_path.c_str(), "rb");
    if (!f) { *err = "open " + o.model_path; return false; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    model_bytes_.resize(sz > 0 ? (size_t)sz : 0);
    bool rd = model_bytes_.empty() || std::fread(model_bytes_.data(), 1, model_bytes_.size(), f) == model_bytes_.size();
    std::fclose(f);
    if (!rd) { *err = "read " + o.model_path; return false; }
    LRT_OK(api_.LiteRtCreateModelFromBuffer(env_, model_bytes_.data(), model_bytes_.size(), &model_), "LiteRtCreateModelFromBuffer");

    LiteRtParamIndex nsig = 0;
    LRT_OK(api_.LiteRtGetNumModelSignatures(model_, &nsig), "LiteRtGetNumModelSignatures");
    if (nsig < 1) { *err = "no signature"; return false; }
    LiteRtSignature sig = nullptr;
    LRT_OK(api_.LiteRtGetModelSignature(model_, 0, &sig), "LiteRtGetModelSignature");
    const char* key = nullptr;
    LRT_OK(api_.LiteRtGetSignatureKey(sig, &key), "LiteRtGetSignatureKey");
    sig_key_ = key ? key : "";
    LiteRtParamIndex n_in = 0, n_out = 0;
    LRT_OK(api_.LiteRtGetNumSignatureInputs(sig, &n_in), "LiteRtGetNumSignatureInputs");
    LRT_OK(api_.LiteRtGetNumSignatureOutputs(sig, &n_out), "LiteRtGetNumSignatureOutputs");
    std::vector<LiteRtRankedTensorType> in_types(n_in), out_types(n_out);
    for (LiteRtParamIndex i = 0; i < n_in; ++i) {
      LiteRtTensor t = nullptr;
      LRT_OK(api_.LiteRtGetSignatureInputTensorByIndex(sig, i, &t), "LiteRtGetSignatureInputTensorByIndex");
      LRT_OK(api_.LiteRtGetRankedTensorType(t, &in_types[i]), "LiteRtGetRankedTensorType(input)");
      const char* nm = nullptr; api_.LiteRtGetTensorName(t, &nm);
      TensorInfo ti; ti.name = nm ? nm : ""; ti.dtype = (int)in_types[i].element_type;
      ti.bytes = NumElems(in_types[i].layout) * ElemBytes(ti.dtype);
      for (unsigned d = 0; d < in_types[i].layout.rank; ++d) ti.dims.push_back(in_types[i].layout.dimensions[d]);
      inputs_.push_back(ti);
    }
    std::vector<std::string> sig_out_names(n_out);
    for (LiteRtParamIndex o2 = 0; o2 < n_out; ++o2) {
      LiteRtTensor t = nullptr;
      LRT_OK(api_.LiteRtGetSignatureOutputTensorByIndex(sig, o2, &t), "LiteRtGetSignatureOutputTensorByIndex");
      LRT_OK(api_.LiteRtGetRankedTensorType(t, &out_types[o2]), "LiteRtGetRankedTensorType(output)");
      const char* nm = nullptr; api_.LiteRtGetTensorName(t, &nm);
      sig_out_names[o2] = nm ? nm : "";
    }
    // the subgraph's output order (= TfLiteInterpreterGetOutputTensor's), matched by tensor name
    LiteRtSubgraph sub = nullptr;
    LRT_OK(api_.LiteRtGetSignatureSubgraph(sig, &sub), "LiteRtGetSignatureSubgraph");
    LiteRtParamIndex n_sub_out = 0;
    LRT_OK(api_.LiteRtGetNumSubgraphOutputs(sub, &n_sub_out), "LiteRtGetNumSubgraphOutputs");
    std::vector<std::string> sub_names(n_sub_out);
    for (LiteRtParamIndex i = 0; i < n_sub_out; ++i) {
      LiteRtTensor t = nullptr;
      LRT_OK(api_.LiteRtGetSubgraphOutput(sub, i, &t), "LiteRtGetSubgraphOutput");
      const char* nm = nullptr; api_.LiteRtGetTensorName(t, &nm);
      sub_names[i] = nm ? nm : "";
    }
    order_.clear();  // order_[k] = the signature output index of subgraph output k
    bool mapped = n_sub_out == n_out;
    for (LiteRtParamIndex k = 0; mapped && k < n_sub_out; ++k) {
      int found = -1;
      for (LiteRtParamIndex s = 0; s < n_out; ++s) if (sig_out_names[s] == sub_names[k]) { found = (int)s; break; }
      if (found < 0) mapped = false; else order_.push_back((size_t)found);
    }
    if (!mapped) { order_.clear(); for (LiteRtParamIndex s = 0; s < n_out; ++s) order_.push_back(s); order_note_ = " ORDER_UNMAPPED(signature order kept)"; }

    // --- options ---
    LRT_OK(api_.LiteRtCreateOptions(&opts_), "LiteRtCreateOptions");
    LRT_OK(api_.LiteRtSetOptionsHardwareAccelerators(opts_, hw), "LiteRtSetOptionsHardwareAccelerators");
    if (o.cpu_threads > 0) {
      if (!AddToml("xnnpack", "num_threads = " + std::to_string(o.cpu_threads) + "\n", err)) return false;
      info_ += " xnnpack_toml(num_threads=" + std::to_string(o.cpu_threads) + ")";
    }
    if (o.accel == "gpu" && o.gpu_precision != "default") {
      int prec = o.gpu_precision == "fp16" ? (int)kLiteRtDelegatePrecisionFp16
               : o.gpu_precision == "fp32" ? (int)kLiteRtDelegatePrecisionFp32
               : o.gpu_precision == "fp16acc32" ? (int)kLiteRtDelegatePrecisionFp16WithFp32Accum : -1;
      if (prec < 0) { *err = "gpu precision " + o.gpu_precision; return false; }
      if (!AddToml("gpu_options", "precision = " + std::to_string(prec) + "\n", err)) return false;
      info_ += " gpu_options(precision=" + std::to_string(prec) + ")";
    } else if (o.accel == "gpu") {
      info_ += " gpu_options(none: the accelerator's default precision)";
    }
    if ((hw & kLiteRtHwAcceleratorNpu) && o.qnn_burst) {
      if (!AddToml("qualcomm", "htp_performance_mode = 2\n", err)) return false;
      info_ += " qualcomm_toml(htp_performance_mode=2)";
    }

    // --- compile, then buffers from the compiled model's requirements ---
    LRT_OK(api_.LiteRtCreateCompiledModel(env_, model_, opts_, &cm_), "LiteRtCreateCompiledModel");
    bool fully = false;
    if (api_.LiteRtCompiledModelIsFullyAccelerated(cm_, &fully) == kLiteRtStatusOk) fully_ = fully ? "true" : "false";
    bool open_shape = false;  // an output the flatbuffer leaves without a shape (set by its op's Prepare)
    for (LiteRtParamIndex o2 = 0; o2 < n_out; ++o2)
      if (out_types[o2].layout.rank == 0 || NumElems(out_types[o2].layout) == 0) open_shape = true;
    if (open_shape) {
      std::vector<LiteRtLayout> layouts(n_out);
      LiteRtStatus ls = api_.LiteRtGetCompiledModelOutputTensorLayouts(cm_, 0, n_out, layouts.data(), true);
      info_ += " output_layouts=from_compiled_model(status " + std::to_string((int)ls) + ")";
      if (ls == kLiteRtStatusOk) for (LiteRtParamIndex o2 = 0; o2 < n_out; ++o2) out_types[o2].layout = layouts[o2];
    }
    ins_.assign(n_in, nullptr);
    outs_.assign(n_out, nullptr);
    for (LiteRtParamIndex i = 0; i < n_in; ++i) {
      LiteRtTensorBufferRequirements req = nullptr;
      LRT_OK(api_.LiteRtGetCompiledModelInputBufferRequirements(cm_, 0, i, &req), "LiteRtGetCompiledModelInputBufferRequirements");
      if (o.host_buffers)
        LRT_OK(api_.LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeHostMemory, &in_types[i], inputs_[i].bytes, &ins_[i]),
               "LiteRtCreateManagedTensorBuffer(input, host memory)");
      else
        LRT_OK(api_.LiteRtCreateManagedTensorBufferFromRequirements(env_, &in_types[i], req, &ins_[i]), "LiteRtCreateManagedTensorBufferFromRequirements(input)");
      info_ += " in" + std::to_string(i) + "(" + DescribeReq(req) + "," + DescribeBuf(ins_[i]) + ")";
    }
    std::vector<TensorInfo> sig_outputs(n_out);
    for (LiteRtParamIndex o2 = 0; o2 < n_out; ++o2) {
      LiteRtTensorBufferRequirements req = nullptr;
      LRT_OK(api_.LiteRtGetCompiledModelOutputBufferRequirements(cm_, 0, o2, &req), "LiteRtGetCompiledModelOutputBufferRequirements");
      size_t rbytes = 0; api_.LiteRtGetTensorBufferRequirementsBufferSize(req, &rbytes);
      TensorInfo ti; ti.name = sig_out_names[o2]; ti.dtype = (int)out_types[o2].element_type;
      ti.bytes = NumElems(out_types[o2].layout) * ElemBytes(ti.dtype);
      if (o.host_buffers)
        LRT_OK(api_.LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeHostMemory, &out_types[o2], ti.bytes, &outs_[o2]),
               "LiteRtCreateManagedTensorBuffer(output, host memory)");
      else
        LRT_OK(api_.LiteRtCreateManagedTensorBufferFromRequirements(env_, &out_types[o2], req, &outs_[o2]), "LiteRtCreateManagedTensorBufferFromRequirements(output)");
      info_ += " out" + std::to_string(o2) + "(" + DescribeReq(req) + "," + DescribeBuf(outs_[o2]) + ")";
      for (unsigned d = 0; d < out_types[o2].layout.rank; ++d) ti.dims.push_back(out_types[o2].layout.dimensions[d]);
      if (ti.bytes == 0 || (rbytes > 0 && ti.bytes > rbytes)) { *err = "output " + ti.name + " bytes " + std::to_string(ti.bytes) + " vs buffer " + std::to_string(rbytes); return false; }
      sig_outputs[o2] = ti;
    }
    for (size_t k = 0; k < order_.size(); ++k) outputs_.push_back(sig_outputs[order_[k]]);
    return true;
  }

  const std::vector<TensorInfo>& Inputs() const override { return inputs_; }
  const std::vector<TensorInfo>& Outputs() const override { return outputs_; }

  bool Write(const std::vector<std::vector<uint8_t>>& in, std::string* err) override {
    if (in.size() != ins_.size()) { *err = "input count"; return false; }
    for (size_t i = 0; i < in.size(); ++i) {
      if (in[i].size() != inputs_[i].bytes) { *err = "input " + std::to_string(i) + " bytes"; return false; }
      void* host = nullptr;
      LRT_OK(api_.LiteRtLockTensorBuffer(ins_[i], &host, kLiteRtTensorBufferLockModeWrite), "LiteRtLockTensorBuffer(input)");
      std::memcpy(host, in[i].data(), in[i].size());
      LRT_OK(api_.LiteRtUnlockTensorBuffer(ins_[i]), "LiteRtUnlockTensorBuffer(input)");
    }
    return true;
  }

  bool Run(std::string* err) override {
    LRT_OK(api_.LiteRtRunCompiledModel(cm_, 0, ins_.size(), ins_.data(), outs_.size(), outs_.data()), "LiteRtRunCompiledModel");
    return true;
  }

  bool Read(std::vector<std::vector<uint8_t>>* out, std::string* err) override {
    out->resize(order_.size());
    for (size_t k = 0; k < order_.size(); ++k) {
      LiteRtTensorBuffer b = outs_[order_[k]];
      void* host = nullptr;
      LRT_OK(api_.LiteRtLockTensorBuffer(b, &host, kLiteRtTensorBufferLockModeRead), "LiteRtLockTensorBuffer(output)");
      (*out)[k].resize(outputs_[k].bytes);
      std::memcpy((*out)[k].data(), host, outputs_[k].bytes);
      LRT_OK(api_.LiteRtUnlockTensorBuffer(b), "LiteRtUnlockTensorBuffer(output)");
    }
    return true;
  }

  std::string Info() const override {
    return "runtime=litert_compiled_model lib=libLiteRt.so accel=" + accel_ + " signature=" + sig_key_ +
           " fully_accelerated=" + fully_ + info_ + order_note_;
  }

 private:
  bool AddToml(const char* identifier, const std::string& toml, std::string* err) {
    char* payload = new char[toml.size() + 1];
    std::memcpy(payload, toml.data(), toml.size());
    payload[toml.size()] = '\0';
    LiteRtOpaqueOptions opaque = nullptr;
    LiteRtStatus st = api_.LiteRtCreateOpaqueOptions(identifier, payload, DeleteCStringPayload, &opaque);
    if (st != kLiteRtStatusOk) { delete[] payload; *err = std::string("LiteRtCreateOpaqueOptions ") + identifier + " status " + std::to_string((int)st); return false; }
    LRT_OK(api_.LiteRtAddOpaqueOptions(opts_, opaque), "LiteRtAddOpaqueOptions");
    return true;
  }

  // "req_bytes=N,req_types=A/B": the requirement's buffer size and its supported LiteRtTensorBufferType codes in order
  // (1 host memory, 2 AHWB, 10 OpenCL buffer, 14 OpenCL buffer packed, ...: litert_tensor_buffer_types.h).
  std::string DescribeReq(LiteRtTensorBufferRequirements req) const {
    size_t rb = 0; int nt = 0;
    api_.LiteRtGetTensorBufferRequirementsBufferSize(req, &rb);
    api_.LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(req, &nt);
    std::string s = "req_bytes=" + std::to_string(rb) + ",req_types=";
    for (int j = 0; j < nt; ++j) {
      LiteRtTensorBufferType t = kLiteRtTensorBufferTypeUnknown;
      api_.LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(req, j, &t);
      s += (j ? "/" : "") + std::to_string((int)t);
    }
    return s;
  }
  // "buf_type=T,buf_bytes=N": the type and size of the buffer the runner made.
  std::string DescribeBuf(LiteRtTensorBuffer b) const {
    LiteRtTensorBufferType t = kLiteRtTensorBufferTypeUnknown; size_t sz = 0;
    api_.LiteRtGetTensorBufferType(b, &t);
    api_.LiteRtGetTensorBufferSize(b, &sz);
    return "buf_type=" + std::to_string((int)t) + ",buf_bytes=" + std::to_string(sz);
  }

  void* lib_ = nullptr;
  LrtApi api_;
  LiteRtEnvironment env_ = nullptr;
  LiteRtModel model_ = nullptr;
  LiteRtOptions opts_ = nullptr;
  LiteRtCompiledModel cm_ = nullptr;
  std::vector<uint8_t> model_bytes_;
  std::vector<LiteRtTensorBuffer> ins_, outs_;
  std::vector<size_t> order_;
  std::vector<TensorInfo> inputs_, outputs_;
  std::string accel_, sig_key_, fully_ = "?", info_, order_note_;
};

}  // namespace

Backend* MakeLiteRtBackend() { return new LiteRtBackend(); }

// ftfl_bench: one native runner for both sides of the comparison on a phone (LiteRT 1.4.0's TFLite C API against
// LiteRT 2.x CompiledModel). Its contract with the phone script: the flags below, a "pid=" line first,
// "compile_end model[0]" after the load, "summary" and "done wall_ms=" at the end, dumps <out>/model0/<id3>.f32 of
// out_elems float32 each.
//   --runtime tflite : the TensorFlow Lite C API as react-native-fast-tflite calls it (tfl_path.cc), on LiteRT 1.4.0's
//                      libtensorflowlite_jni.so (+ libtensorflowlite_gpu_jni.so); accel cpu | gpu | npu (= NNAPI).
//   --runtime litert : LiteRT CompiledModel (lrt_path.cc) on libLiteRt.so; accel cpu | gpu | npu | npu+cpu.
// One process holds one runtime (each path dlopens its library RTLD_LOCAL; nothing is linked), so runs of both
// runtimes can alternate on one phone.
// Per fixture (the parity pass) and per round (the latency phase) one call = write every input + run + read every
// output back into host buffers, the span of fast-tflite's runSync(); the run alone is timed too. The dump of a
// fixture is every output in the primary subgraph's output order, converted to float32 and concatenated.
// Thread facts (the flag is never trusted): Threads of /proc/self/status after the load, after the warm-up and after
// the rounds, CPU time over wall time across the rounds, and the tasks that used CPU. CPU frequency ceilings
// (policy*/scaling_max_freq over cpuinfo_max_freq) are read before the rounds and after each round, outside the
// timed span; the lowest ratio is printed (runs under 0.85 are not used).
//
// usage: ftfl_bench --runtime tflite|litert --accel cpu|gpu|npu|npu+cpu [--gpu-precision default|fp32|fp16|fp16acc32]
//                   [--burst 0|1] --libdir DIR [--cache-dir DIR] --fixtures DIR --count K [--inputs in0,...]
//                   [--warmup W] [--rounds R] [--ab-fixture ID] [--dump full|none] --out DIR --model FILE
//                   [--cpu-threads N] [--tflite-lib NAME] [--tflite-gpu-lib NAME] [--qnn-burst 0|1]
//                   [--cpu-fallback 0|1] [--xnnpack 0|1] [--host-buffers 0|1] [--form TAG]
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bench_common.h"

namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
long long WallMs() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000; }
double ProcessCpuMs() { struct timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

int ProcThreads() {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return -1;
  char line[256];
  int n = -1;
  while (std::fgets(line, sizeof line, f)) if (std::strncmp(line, "Threads:", 8) == 0) { n = std::atoi(line + 8); break; }
  std::fclose(f);
  return n;
}

std::map<std::string, long> TaskTicks() {  // utime + stime per task of this process
  std::map<std::string, long> out;
  DIR* d = ::opendir("/proc/self/task");
  if (!d) return out;
  while (struct dirent* e = ::readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string p = std::string("/proc/self/task/") + e->d_name + "/stat";
    FILE* f = std::fopen(p.c_str(), "r");
    if (!f) continue;
    char buf[1024];
    size_t r = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[r] = 0;
    const char* q = std::strrchr(buf, ')');
    if (!q) continue;
    long u = 0, s = 0;
    int field = 3;
    const char* t = q + 1;
    while (*t && field <= 15) {
      while (*t == ' ') ++t;
      if (field == 14) u = std::strtol(t, nullptr, 10);
      if (field == 15) s = std::strtol(t, nullptr, 10);
      while (*t && *t != ' ') ++t;
      ++field;
    }
    out[e->d_name] = u + s;
  }
  ::closedir(d);
  return out;
}

long ReadLong(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return -1;
  long v = -1;
  if (std::fscanf(f, "%ld", &v) != 1) v = -1;
  std::fclose(f);
  return v;
}

// lowest scaling_max_freq / cpuinfo_max_freq over the cpufreq policies, and the policies as "name=cur/max"
double CeilingRatio(std::string* detail) {
  double low = 2.0;
  std::string det;
  DIR* d = ::opendir("/sys/devices/system/cpu/cpufreq");
  if (!d) { if (detail) *detail = "unreadable"; return -1.0; }
  std::vector<std::string> names;
  while (struct dirent* e = ::readdir(d)) if (std::strncmp(e->d_name, "policy", 6) == 0) names.push_back(e->d_name);
  ::closedir(d);
  std::sort(names.begin(), names.end());
  for (const auto& n : names) {
    std::string base = "/sys/devices/system/cpu/cpufreq/" + n;
    long cur = ReadLong(base + "/scaling_max_freq"), mx = ReadLong(base + "/cpuinfo_max_freq");
    if (cur <= 0 || mx <= 0) continue;
    double r = (double)cur / (double)mx;
    if (r < low) low = r;
    det += (det.empty() ? "" : ",") + n + "=" + std::to_string(cur) + "/" + std::to_string(mx);
  }
  if (detail) *detail = det;
  return low > 1.5 ? -1.0 : low;
}

double Pct(std::vector<double> v, double p) {  // nearest rank
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t k = (size_t)std::ceil(p / 100.0 * v.size());
  if (k < 1) k = 1;
  if (k > v.size()) k = v.size();
  return v[k - 1];
}
double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

const char* DtypeName(int t) {
  switch (t) {
    case 1: return "float32"; case 2: return "int32"; case 3: return "uint8"; case 4: return "int64"; case 6: return "bool";
    case 7: return "int16"; case 9: return "int8"; case 10: return "float16"; case 11: return "float64"; default: return "other";
  }
}
const char* DtypeExt(int t) {
  switch (t) { case 1: return ".f32"; case 2: return ".i32"; case 4: return ".i64"; case 10: return ".f16"; default: return ".u8"; }
}

float HalfToFloat(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
  if (e == 0) {
    if (m == 0) bits = s;
    else { int k = -1; do { ++k; m <<= 1; } while (!(m & 0x400)); m &= 0x3ff; bits = s | ((uint32_t)(127 - 15 - k) << 23) | (m << 13); }
  } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
  else bits = s | ((e + 127 - 15) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// every output's raw bytes -> float32 values, appended in order (values as stored: no dequantization)
void ToFloat(const std::vector<TensorInfo>& outs, const std::vector<std::vector<uint8_t>>& raw, std::vector<float>* f) {
  f->clear();
  for (size_t i = 0; i < outs.size(); ++i) {
    const uint8_t* p = raw[i].data();
    size_t n = raw[i].size();
    switch (outs[i].dtype) {
      case 1: for (size_t k = 0; k + 4 <= n; k += 4) { float v; std::memcpy(&v, p + k, 4); f->push_back(v); } break;
      case 2: for (size_t k = 0; k + 4 <= n; k += 4) { int32_t v; std::memcpy(&v, p + k, 4); f->push_back((float)v); } break;
      case 4: for (size_t k = 0; k + 8 <= n; k += 8) { int64_t v; std::memcpy(&v, p + k, 8); f->push_back((float)v); } break;
      case 9: for (size_t k = 0; k < n; ++k) f->push_back((float)(int8_t)p[k]); break;
      case 10: for (size_t k = 0; k + 2 <= n; k += 2) { uint16_t v; std::memcpy(&v, p + k, 2); f->push_back(HalfToFloat(v)); } break;
      case 7: for (size_t k = 0; k + 2 <= n; k += 2) { int16_t v; std::memcpy(&v, p + k, 2); f->push_back((float)v); } break;
      default: for (size_t k = 0; k < n; ++k) f->push_back((float)p[k]); break;
    }
  }
}

bool ReadBytes(const std::string& path, std::vector<uint8_t>* v) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  v->assign(sz > 0 ? (size_t)sz : 0, 0);
  bool ok = v->empty() || std::fread(v->data(), 1, v->size(), f) == v->size();
  std::fclose(f);
  return ok;
}

bool WriteFloats(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  return ok;
}

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> r;
  size_t p = 0;
  while (p <= s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    r.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  return r;
}

std::string Dims(const std::vector<int>& d) {
  std::string s = "[";
  for (size_t i = 0; i < d.size(); ++i) s += (i ? "," : "") + std::to_string(d[i]);
  return s + "]";
}

[[noreturn]] void Fatal(const std::string& what) {
  std::printf("FATAL %s\n", what.c_str());
  std::fprintf(stderr, "ERROR: %s\n", what.c_str());
  std::fflush(stdout);
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string runtime = "litert", fixtures, out_dir, inputs_arg, ab_fixture = "000", dump = "full", form;
  LoadOpts o;
  o.accel = "cpu";
  o.gpu_precision = "default";
  long warmup = 5, rounds = 50, count = 0;
  int burst = 0;
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", what); std::exit(1); }
      return argv[++i];
    };
    if (f == "--runtime") runtime = next("--runtime");
    else if (f == "--accel") o.accel = next("--accel");
    else if (f == "--gpu-precision") o.gpu_precision = next("--gpu-precision");
    else if (f == "--burst") burst = std::atoi(next("--burst"));  // the phone script always passes 1; recorded only
    else if (f == "--libdir") o.libdir = next("--libdir");
    else if (f == "--cache-dir") o.cache_dir = next("--cache-dir");
    else if (f == "--fixtures") fixtures = next("--fixtures");
    else if (f == "--count") count = std::strtol(next("--count"), nullptr, 10);
    else if (f == "--inputs") inputs_arg = next("--inputs");
    else if (f == "--warmup") warmup = std::strtol(next("--warmup"), nullptr, 10);
    else if (f == "--rounds") rounds = std::strtol(next("--rounds"), nullptr, 10);
    else if (f == "--ab-fixture") ab_fixture = next("--ab-fixture");
    else if (f == "--dump") dump = next("--dump");
    else if (f == "--out") out_dir = next("--out");
    else if (f == "--model") o.model_path = next("--model");
    else if (f == "--cpu-threads") o.cpu_threads = std::atoi(next("--cpu-threads"));
    else if (f == "--tflite-lib") o.tflite_lib = next("--tflite-lib");
    else if (f == "--tflite-gpu-lib") o.tflite_gpu_lib = next("--tflite-gpu-lib");
    else if (f == "--qnn-burst") o.qnn_burst = std::atoi(next("--qnn-burst"));
    else if (f == "--cpu-fallback") o.cpu_fallback = std::atoi(next("--cpu-fallback"));
    else if (f == "--xnnpack") o.xnnpack = std::atoi(next("--xnnpack"));
    else if (f == "--host-buffers") o.host_buffers = std::atoi(next("--host-buffers"));
    else if (f == "--form") form = next("--form");
    else { std::fprintf(stderr, "unknown flag %s\n", f.c_str()); return 1; }
  }
  if (o.model_path.empty() || o.libdir.empty() || fixtures.empty() || out_dir.empty() || count < 1) {
    std::fprintf(stderr, "need --model, --libdir, --fixtures, --count, --out\n");
    return 1;
  }
  if (runtime != "tflite" && runtime != "litert") { std::fprintf(stderr, "--runtime tflite|litert\n"); return 1; }
  if (dump != "full" && dump != "none") { std::fprintf(stderr, "--dump full|none\n"); return 1; }
  ::mkdir(out_dir.c_str(), 0775);
  std::printf("pid=%d runtime=%s accel=%s gpu_precision=%s cpu_threads=%d qnn_burst=%d cpu_fallback=%d xnnpack=%d host_buffers=%d burst_flag=%d "
              "form=%s fixtures=%ld warmup=%ld rounds=%ld ab_fixture=%s libdir=%s cache_dir=%s dump=%s tflite_lib=%s wall_ms=%lld\n",
              (int)getpid(), runtime.c_str(), o.accel.c_str(), o.gpu_precision.c_str(), o.cpu_threads, o.qnn_burst, o.cpu_fallback,
              o.xnnpack, o.host_buffers, burst,
              form.empty() ? "-" : form.c_str(), count, warmup, rounds, ab_fixture.c_str(), o.libdir.c_str(),
              o.cache_dir.empty() ? "-" : o.cache_dir.c_str(), dump.c_str(), o.tflite_lib.empty() ? "-" : o.tflite_lib.c_str(),
              WallMs());
  std::fflush(stdout);

  std::unique_ptr<Backend> be(runtime == "tflite" ? MakeTfliteBackend() : MakeLiteRtBackend());
  std::string err;
  std::printf("compile_start model[0] wall_ms=%lld\n", WallMs());
  std::fflush(stdout);
  auto l0 = Clock::now();
  if (!be->Load(o, &err)) Fatal("load: " + err);
  double load_ms = Ms(l0, Clock::now());
  std::printf("compile_end model[0] wall_ms=%lld\n", WallMs());
  int th_load = ProcThreads();
  const auto& ins = be->Inputs();
  const auto& outs = be->Outputs();
  size_t out_elems = 0;
  for (const auto& t : outs) {
    size_t eb = t.dtype == 1 || t.dtype == 2 ? 4 : t.dtype == 4 ? 8 : (t.dtype == 10 || t.dtype == 7) ? 2 : 1;
    out_elems += t.bytes / eb;
  }
  std::printf("model[0]=%s load_ms=%.1f threads_after_load=%d inputs=%zu outputs=%zu out_elems=%zu %s\n",
              o.model_path.c_str(), load_ms, th_load, ins.size(), outs.size(), out_elems, be->Info().c_str());
  for (size_t i = 0; i < ins.size(); ++i)
    std::printf("model[0] input[%zu] name=%s dtype=%s dims=%s bytes=%zu\n", i, ins[i].name.c_str(), DtypeName(ins[i].dtype),
                Dims(ins[i].dims).c_str(), ins[i].bytes);
  for (size_t i = 0; i < outs.size(); ++i)
    std::printf("model[0] output[%zu] name=%s dtype=%s dims=%s bytes=%zu\n", i, outs[i].name.c_str(), DtypeName(outs[i].dtype),
                Dims(outs[i].dims).c_str(), outs[i].bytes);
  std::fflush(stdout);

  std::vector<std::string> names = inputs_arg.empty() ? std::vector<std::string>() : Split(inputs_arg);
  std::vector<std::vector<uint8_t>> in_data(ins.size()), raw;
  auto load_fixture = [&](const std::string& id) -> bool {
    for (size_t i = 0; i < ins.size(); ++i) {
      std::string nm = i < names.size() ? names[i] : "in" + std::to_string(i);
      std::string p = fixtures + "/" + id + "_" + nm + DtypeExt(ins[i].dtype);
      if (!ReadBytes(p, &in_data[i]) || in_data[i].size() != ins[i].bytes) {
        std::printf("FIXTURE_ERROR %s (%zu bytes, want %zu)\n", p.c_str(), in_data[i].size(), ins[i].bytes);
        return false;
      }
    }
    return true;
  };
  double last_run_ms = 0;
  auto call = [&]() -> double {  // write + run + read back; the run alone in last_run_ms
    auto t0 = Clock::now();
    if (!be->Write(in_data, &err)) Fatal("write: " + err);
    auto t1 = Clock::now();
    if (!be->Run(&err)) Fatal("run: " + err);
    auto t2 = Clock::now();
    if (!be->Read(&raw, &err)) Fatal("read: " + err);
    auto t3 = Clock::now();
    last_run_ms = Ms(t1, t2);
    return Ms(t0, t3);
  };

  // --- phase 1: the parity pass, every fixture once; fixture 000's first call is the first inference ---
  std::string mdir = out_dir + "/model0";
  ::mkdir(mdir.c_str(), 0775);
  double first_ms = -1, first_run_ms = -1;
  std::vector<float> fl;
  char idbuf[16];
  size_t nonfinite_total = 0;
  for (long fx = 0; fx < count; ++fx) {
    std::snprintf(idbuf, sizeof idbuf, "%03ld", fx);
    std::string id = idbuf;
    if (!load_fixture(id)) return 3;
    double t = call();
    if (fx == 0) { first_ms = t; first_run_ms = last_run_ms; t = call(); }
    ToFloat(outs, raw, &fl);
    size_t nf = 0;
    double l2 = 0, mx = 0;
    for (float v : fl) { if (!std::isfinite(v)) { ++nf; continue; } l2 += (double)v * v; mx = std::max(mx, (double)std::fabs(v)); }
    nonfinite_total += nf;
    if (fl.size() != out_elems) Fatal("output elements " + std::to_string(fl.size()) + " != " + std::to_string(out_elems));
    if (dump == "full" && !WriteFloats(mdir + "/" + id + ".f32", fl)) { std::printf("WRITE_ERROR %s/%s.f32\n", mdir.c_str(), id.c_str()); return 3; }
    std::printf("parity model[0] fixture %s ms=%.3f run_ms=%.3f nonfinite=%zu out[0..3]=%.6g,%.6g,%.6g,%.6g l2=%.6g maxabs=%.6g\n",
                id.c_str(), t, last_run_ms, nf, fl.size() > 0 ? fl[0] : 0.f, fl.size() > 1 ? fl[1] : 0.f, fl.size() > 2 ? fl[2] : 0.f,
                fl.size() > 3 ? fl[3] : 0.f, std::sqrt(l2), mx);
  }
  std::printf("parity_summary model[0] %s fixtures=%ld nonfinite_values=%zu first_call_ms=%.3f first_run_ms=%.3f load_ms=%.1f outdir=%s\n",
              o.model_path.c_str(), count, nonfinite_total, first_ms, first_run_ms, load_ms, mdir.c_str());
  std::fflush(stdout);

  // --- phase 2: steady latency on one fixture ---
  if (rounds > 0) {
    if (!load_fixture(ab_fixture)) return 3;
    std::printf("latency_start wall_ms=%lld\n", WallMs());
    for (long w = 0; w < warmup; ++w) { double t = call(); std::printf("warmup %ld %.3f ms run_ms=%.3f\n", w, t, last_run_ms); }
    int th_calls = ProcThreads();
    std::string cdet0;
    double ceil_low = CeilingRatio(&cdet0);
    std::printf("ceiling_before %.4f %s\n", ceil_low, cdet0.c_str());
    std::vector<double> ms, run_ms;
    std::map<std::string, long> ticks0 = TaskTicks();
    double cpu_sum = 0, wall_sum = 0;
    for (long r = 0; r < rounds; ++r) {
      double c0 = ProcessCpuMs();
      auto w0 = Clock::now();
      double t = call();
      wall_sum += Ms(w0, Clock::now());
      cpu_sum += ProcessCpuMs() - c0;
      ms.push_back(t);
      run_ms.push_back(last_run_ms);
      std::string cd;
      double cr = CeilingRatio(&cd);  // outside the timed span
      if (cr >= 0 && (ceil_low < 0 || cr < ceil_low)) ceil_low = cr;
      std::printf("round %ld ms=%.3f run_ms=%.3f ceiling=%.4f\n", r, t, last_run_ms, cr);
    }
    int th_end = ProcThreads();
    std::map<std::string, long> ticks1 = TaskTicks();
    int busy = 0;
    for (const auto& kv : ticks1) { auto it = ticks0.find(kv.first); if (kv.second > (it == ticks0.end() ? 0 : it->second)) ++busy; }
    std::printf("thread_check model[0] requested=%d threads_load=%d threads_calls=%d threads_end=%d tasks_with_cpu=%d cpu_ms=%.1f wall_ms=%.1f cpu_over_wall=%.2f\n",
                o.cpu_threads, th_load, th_calls, th_end, busy, cpu_sum, wall_sum, wall_sum > 0 ? cpu_sum / wall_sum : 0.0);
    std::printf("summary model[0] %s runtime=%s accel=%s gpu_precision=%s form=%s n=%zu median=%.3f p10=%.3f p90=%.3f min=%.3f max=%.3f "
                "run_median=%.3f load_ms=%.1f first_call_ms=%.3f ceiling_min=%.4f nonfinite_values=%zu\n",
                o.model_path.c_str(), runtime.c_str(), o.accel.c_str(), o.gpu_precision.c_str(), form.empty() ? "-" : form.c_str(), ms.size(),
                Median(ms), Pct(ms, 10), Pct(ms, 90), *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end()),
                Median(run_ms), load_ms, first_ms, ceil_low, nonfinite_total);
  }
  // the summary is out before the teardown: a crash while the runtime tears down (react-native-fast-tflite #186 reports
  // SIGABRT in the PowerVR OpenCL watchdog thread) leaves the numbers and shows up as a missing done line
  std::printf("teardown_start wall_ms=%lld\n", WallMs());
  std::fflush(stdout);
  be.reset();
  std::printf("done wall_ms=%lld\n", WallMs());
  std::fflush(stdout);
  return 0;
}

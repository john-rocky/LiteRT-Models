// Helpers Swift cannot express directly: LiteRtLayout bitfield access, tensor
// type construction, and the GPU-options opaque-payload dance (C function
// pointers are awkward to thread through Swift).
#ifndef SAM3_HELPERS_H_
#define SAM3_HELPERS_H_

#include <stdbool.h>
#include <stdint.h>

#include "litert/c/litert_model_types.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/options/litert_gpu_options.h"

static inline uint32_t Sam3Rank(const LiteRtRankedTensorType* t) {
  return t->layout.rank;
}

static inline int32_t Sam3Dim(const LiteRtRankedTensorType* t, uint32_t i) {
  return t->layout.dimensions[i];
}

static inline int32_t Sam3ElemType(const LiteRtRankedTensorType* t) {
  return (int32_t)t->element_type;
}

static inline LiteRtRankedTensorType Sam3MakeType(int32_t element_type,
                                                  const int32_t* dims,
                                                  uint32_t rank) {
  LiteRtRankedTensorType type;
  type.element_type = (LiteRtElementType)element_type;
  type.layout.rank = rank;
  type.layout.has_strides = false;
  for (uint32_t i = 0; i < rank; ++i) {
    type.layout.dimensions[i] = dims[i];
  }
  return type;
}

// Attach GPU accelerator options to `options`. With `enforce_f32` the Metal
// delegate computes in float32 (exact); without it fp16 (fast). The gpu-options
// payload's ownership moves into the opaque options list, which `options`
// takes over on add — nothing to destroy here on success.
static inline LiteRtStatus Sam3AddGpuOptions(LiteRtOptions options,
                                             bool enforce_f32) {
  LrtGpuOptions* gpu = 0;
  LiteRtStatus s = LrtCreateGpuOptions(&gpu);
  if (s != kLiteRtStatusOk) return s;
  if (enforce_f32) {
    s = LrtSetGpuAcceleratorCompilationOptionsPrecision(
        gpu, kLiteRtDelegatePrecisionFp32);
    if (s != kLiteRtStatusOk) return s;
  }
  const char* identifier = 0;
  void* payload = 0;
  void (*deleter)(void*) = 0;
  s = LrtGetOpaqueGpuOptionsData(gpu, &identifier, &payload, &deleter);
  if (s != kLiteRtStatusOk) return s;
  LiteRtOpaqueOptions opaque = 0;
  s = LiteRtCreateOpaqueOptions(identifier, payload, deleter, &opaque);
  if (s != kLiteRtStatusOk) return s;
  return LiteRtAddOpaqueOptions(options, opaque);
}

#endif  // SAM3_HELPERS_H_

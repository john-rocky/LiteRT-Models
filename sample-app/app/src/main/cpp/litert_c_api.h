// Minimal LiteRT C API declarations for dlopen/dlsym usage.
// Based on LiteRT 2.1.3 public C headers (litert/c/*.h).
// We load libLiteRt.so at runtime — no link-time dependency needed.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <dlfcn.h>
#include <android/log.h>

#define LITERT_TENSOR_MAX_RANK 8

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------
typedef struct LiteRtEnvironmentT* LiteRtEnvironment;
typedef struct LiteRtModelT* LiteRtModel;
typedef struct LiteRtCompiledModelT* LiteRtCompiledModel;
typedef struct LiteRtTensorBufferT* LiteRtTensorBuffer;
typedef struct LiteRtTensorBufferRequirementsT* LiteRtTensorBufferRequirements;
typedef struct LiteRtOptionsT* LiteRtOptions;
typedef struct LiteRtEventT* LiteRtEvent;

// ---------------------------------------------------------------------------
// Basic types
// ---------------------------------------------------------------------------
typedef int LiteRtParamIndex;

// Status codes (subset)
typedef enum {
    kLiteRtStatusOk = 0,
    kLiteRtStatusErrorInvalidArgument = 1,
    kLiteRtStatusErrorMemoryAllocationFailure = 2,
    kLiteRtStatusErrorRuntimeFailure = 3,
    kLiteRtStatusErrorMissingInputTensor = 4,
    kLiteRtStatusErrorUnsupported = 5,
    kLiteRtStatusErrorNotFound = 6,
} LiteRtStatus;

// Hardware accelerator flags (bitmask)
typedef uint64_t LiteRtHwAcceleratorSet;
#define kLiteRtHwAcceleratorCpu   (1ULL << 0)
#define kLiteRtHwAcceleratorGpu   (1ULL << 1)
#define kLiteRtHwAcceleratorNpu   (1ULL << 2)

// Element types (must match TFLite values exactly)
typedef enum {
    kLiteRtElementTypeNone = 0,
    kLiteRtElementTypeFloat32 = 1,
    kLiteRtElementTypeInt32 = 2,
    kLiteRtElementTypeUInt8 = 3,
    kLiteRtElementTypeInt64 = 4,
    kLiteRtElementTypeBool = 6,
    kLiteRtElementTypeInt16 = 7,
    kLiteRtElementTypeInt8 = 9,
    kLiteRtElementTypeFloat16 = 10,
    kLiteRtElementTypeFloat64 = 11,
} LiteRtElementType;

// Layout
typedef struct {
    unsigned int rank : 7;
    bool has_strides : 1;
    int32_t dimensions[LITERT_TENSOR_MAX_RANK];
    uint32_t strides[LITERT_TENSOR_MAX_RANK];
} LiteRtLayout;

// RankedTensorType
typedef struct {
    LiteRtElementType element_type;
    LiteRtLayout layout;
} LiteRtRankedTensorType;

// Tensor buffer types
typedef enum {
    kLiteRtTensorBufferTypeHostMemory = 0,
    kLiteRtTensorBufferTypeAhwb = 1,
    kLiteRtTensorBufferTypeGlBuffer = 6,
    kLiteRtTensorBufferTypeGlTexture = 7,
} LiteRtTensorBufferType;

// Lock mode
typedef enum {
    kLiteRtTensorBufferLockModeRead = 0,
    kLiteRtTensorBufferLockModeWrite = 1,
    kLiteRtTensorBufferLockModeWriteReplace = 2,
} LiteRtTensorBufferLockMode;

// GL types (matching OpenGL ES)
typedef uint32_t LiteRtGLenum;
typedef uint32_t LiteRtGLuint;

// Deallocator callback (can be nullptr for no-op)
typedef void (*LiteRtGlBufferDeallocator)(LiteRtGLuint);

// Delegate precision
typedef enum {
    kLiteRtDelegatePrecisionDefault = 0,
    kLiteRtDelegatePrecisionFp16 = 1,
    kLiteRtDelegatePrecisionFp32 = 2,
} LiteRtDelegatePrecision;

// ---------------------------------------------------------------------------
// Environment options
// ---------------------------------------------------------------------------
typedef enum {
    kLiteRtEnvOptionTagNull = 255,
    kLiteRtEnvOptionTagOpenClDeviceId = 10,
    kLiteRtEnvOptionTagOpenClPlatformId = 11,
    kLiteRtEnvOptionTagOpenClContext = 12,
    kLiteRtEnvOptionTagOpenClCommandQueue = 13,
    kLiteRtEnvOptionTagEglDisplay = 14,
    kLiteRtEnvOptionTagEglContext = 15,
} LiteRtEnvOptionTag;

typedef struct {
    LiteRtEnvOptionTag tag;
    union {
        void* ptr;
        uint64_t u64;
    } value;
} LiteRtEnvOption;

// GPU options opaque handle
typedef struct LrtGpuOptionsT* LrtGpuOptions;

// ---------------------------------------------------------------------------
// Function pointer typedefs for dlsym loading
// ---------------------------------------------------------------------------

// Environment
typedef LiteRtStatus (*pfn_LiteRtCreateEnvironment)(
    int num_options, const LiteRtEnvOption* options,
    LiteRtEnvironment* environment);
typedef void (*pfn_LiteRtDestroyEnvironment)(LiteRtEnvironment environment);
typedef LiteRtStatus (*pfn_LiteRtGpuEnvironmentCreate)(
    LiteRtEnvironment environment, int num_options,
    const LiteRtEnvOption* options);

// Model
typedef LiteRtStatus (*pfn_LiteRtCreateModelFromFile)(
    const char* filename, LiteRtModel* model);
typedef LiteRtStatus (*pfn_LiteRtCreateModelFromBuffer)(
    const void* buffer_addr, size_t buffer_size, LiteRtModel* model);
typedef void (*pfn_LiteRtDestroyModel)(LiteRtModel model);

// Options
typedef LiteRtStatus (*pfn_LiteRtCreateOptions)(LiteRtOptions* options);
typedef void (*pfn_LiteRtDestroyOptions)(LiteRtOptions options);
typedef LiteRtStatus (*pfn_LiteRtSetOptionsHardwareAccelerators)(
    LiteRtOptions options, LiteRtHwAcceleratorSet hw_accelerators);

// GPU options
typedef LiteRtStatus (*pfn_LrtCreateGpuOptions)(LrtGpuOptions* options);
typedef void (*pfn_LrtDestroyGpuOptions)(LrtGpuOptions options);
typedef LiteRtStatus (*pfn_LrtSetGpuAcceleratorCompilationOptionsPrecision)(
    LrtGpuOptions gpu_options, LiteRtDelegatePrecision precision);
typedef LiteRtStatus (*pfn_LiteRtAddOpaqueOptions)(
    LiteRtOptions options, const char* identifier,
    const void* payload, void (*payload_deleter)(void*));

// CompiledModel
typedef LiteRtStatus (*pfn_LiteRtCreateCompiledModel)(
    LiteRtEnvironment environment, LiteRtModel model,
    LiteRtOptions compilation_options,
    LiteRtCompiledModel* compiled_model);
typedef void (*pfn_LiteRtDestroyCompiledModel)(
    LiteRtCompiledModel compiled_model);
typedef LiteRtStatus (*pfn_LiteRtRunCompiledModel)(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    size_t num_input_buffers, LiteRtTensorBuffer* input_buffers,
    size_t num_output_buffers, LiteRtTensorBuffer* output_buffers);

// Buffer requirements
typedef LiteRtStatus (*pfn_LiteRtGetCompiledModelInputBufferRequirements)(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex input_index,
    LiteRtTensorBufferRequirements* buffer_requirements);
typedef LiteRtStatus (*pfn_LiteRtGetCompiledModelOutputBufferRequirements)(
    LiteRtCompiledModel compiled_model, LiteRtParamIndex signature_index,
    LiteRtParamIndex output_index,
    LiteRtTensorBufferRequirements* buffer_requirements);
typedef LiteRtStatus (*pfn_LiteRtGetTensorBufferRequirementsBufferSize)(
    LiteRtTensorBufferRequirements requirements, size_t* buffer_size);

// TensorBuffer
typedef LiteRtStatus (*pfn_LiteRtCreateTensorBufferFromGlBuffer)(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtGLenum target, LiteRtGLuint id, size_t size_bytes, size_t offset,
    LiteRtGlBufferDeallocator deallocator, LiteRtTensorBuffer* buffer);
typedef LiteRtStatus (*pfn_LiteRtCreateManagedTensorBuffer)(
    LiteRtEnvironment env, LiteRtTensorBufferType buffer_type,
    const LiteRtRankedTensorType* tensor_type, size_t buffer_size,
    LiteRtTensorBuffer* buffer);
typedef void (*pfn_LiteRtDestroyTensorBuffer)(LiteRtTensorBuffer buffer);
typedef LiteRtStatus (*pfn_LiteRtLockTensorBuffer)(
    LiteRtTensorBuffer buffer, void** host_mem_addr,
    LiteRtTensorBufferLockMode lock_mode);
typedef LiteRtStatus (*pfn_LiteRtUnlockTensorBuffer)(
    LiteRtTensorBuffer buffer);

// Create managed buffer from requirements (lets compiled model decide the format)
typedef LiteRtStatus (*pfn_LiteRtCreateManagedTensorBufferFromRequirements)(
    LiteRtEnvironment env, const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements requirements,
    LiteRtTensorBuffer* buffer);

// ---------------------------------------------------------------------------
// API loader singleton
// ---------------------------------------------------------------------------
struct LiteRtApi {
    void* handle = nullptr;

    // Environment
    pfn_LiteRtCreateEnvironment CreateEnvironment = nullptr;
    pfn_LiteRtDestroyEnvironment DestroyEnvironment = nullptr;
    pfn_LiteRtGpuEnvironmentCreate GpuEnvironmentCreate = nullptr;

    // Model
    pfn_LiteRtCreateModelFromFile CreateModelFromFile = nullptr;
    pfn_LiteRtCreateModelFromBuffer CreateModelFromBuffer = nullptr;
    pfn_LiteRtDestroyModel DestroyModel = nullptr;

    // Options
    pfn_LiteRtCreateOptions CreateOptions = nullptr;
    pfn_LiteRtDestroyOptions DestroyOptions = nullptr;
    pfn_LiteRtSetOptionsHardwareAccelerators SetOptionsHwAccelerators = nullptr;
    pfn_LiteRtAddOpaqueOptions AddOpaqueOptions = nullptr;

    // GPU options
    pfn_LrtCreateGpuOptions CreateGpuOptions = nullptr;
    pfn_LrtDestroyGpuOptions DestroyGpuOptions = nullptr;
    pfn_LrtSetGpuAcceleratorCompilationOptionsPrecision SetGpuPrecision = nullptr;

    // CompiledModel
    pfn_LiteRtCreateCompiledModel CreateCompiledModel = nullptr;
    pfn_LiteRtDestroyCompiledModel DestroyCompiledModel = nullptr;
    pfn_LiteRtRunCompiledModel RunCompiledModel = nullptr;

    // Buffer requirements
    pfn_LiteRtGetCompiledModelInputBufferRequirements GetInputBufferRequirements = nullptr;
    pfn_LiteRtGetCompiledModelOutputBufferRequirements GetOutputBufferRequirements = nullptr;
    pfn_LiteRtGetTensorBufferRequirementsBufferSize GetBufferRequirementsSize = nullptr;

    // TensorBuffer
    pfn_LiteRtCreateTensorBufferFromGlBuffer CreateTensorBufferFromGlBuffer = nullptr;
    pfn_LiteRtCreateManagedTensorBuffer CreateManagedTensorBuffer = nullptr;
    pfn_LiteRtCreateManagedTensorBufferFromRequirements CreateManagedTensorBufferFromRequirements = nullptr;
    pfn_LiteRtDestroyTensorBuffer DestroyTensorBuffer = nullptr;
    pfn_LiteRtLockTensorBuffer LockTensorBuffer = nullptr;
    pfn_LiteRtUnlockTensorBuffer UnlockTensorBuffer = nullptr;

    bool load() {
        handle = dlopen("libLiteRt.so", RTLD_NOW);
        if (!handle) {
            __android_log_print(ANDROID_LOG_ERROR, "DepthPipeline",
                "Failed to load libLiteRt.so: %s", dlerror());
            return false;
        }

        #define LOAD(member, sym) do { \
            member = (pfn_##sym)dlsym(handle, #sym); \
            if (!member) { \
                __android_log_print(ANDROID_LOG_WARN, "DepthPipeline", \
                    "dlsym failed: %s", #sym); \
            } \
        } while(0)

        LOAD(CreateEnvironment, LiteRtCreateEnvironment);
        LOAD(DestroyEnvironment, LiteRtDestroyEnvironment);
        LOAD(GpuEnvironmentCreate, LiteRtGpuEnvironmentCreate);
        LOAD(CreateModelFromFile, LiteRtCreateModelFromFile);
        LOAD(CreateModelFromBuffer, LiteRtCreateModelFromBuffer);
        LOAD(DestroyModel, LiteRtDestroyModel);
        LOAD(CreateOptions, LiteRtCreateOptions);
        LOAD(DestroyOptions, LiteRtDestroyOptions);
        LOAD(SetOptionsHwAccelerators, LiteRtSetOptionsHardwareAccelerators);
        LOAD(AddOpaqueOptions, LiteRtAddOpaqueOptions);
        LOAD(CreateGpuOptions, LrtCreateGpuOptions);
        LOAD(DestroyGpuOptions, LrtDestroyGpuOptions);
        LOAD(SetGpuPrecision, LrtSetGpuAcceleratorCompilationOptionsPrecision);
        LOAD(CreateCompiledModel, LiteRtCreateCompiledModel);
        LOAD(DestroyCompiledModel, LiteRtDestroyCompiledModel);
        LOAD(RunCompiledModel, LiteRtRunCompiledModel);
        LOAD(GetInputBufferRequirements, LiteRtGetCompiledModelInputBufferRequirements);
        LOAD(GetOutputBufferRequirements, LiteRtGetCompiledModelOutputBufferRequirements);
        LOAD(GetBufferRequirementsSize, LiteRtGetTensorBufferRequirementsBufferSize);
        LOAD(CreateTensorBufferFromGlBuffer, LiteRtCreateTensorBufferFromGlBuffer);
        LOAD(CreateManagedTensorBuffer, LiteRtCreateManagedTensorBuffer);
        LOAD(CreateManagedTensorBufferFromRequirements, LiteRtCreateManagedTensorBufferFromRequirements);
        LOAD(DestroyTensorBuffer, LiteRtDestroyTensorBuffer);
        LOAD(LockTensorBuffer, LiteRtLockTensorBuffer);
        LOAD(UnlockTensorBuffer, LiteRtUnlockTensorBuffer);

        #undef LOAD

        // Check critical functions
        if (!CreateEnvironment || !CreateModelFromBuffer ||
            !CreateCompiledModel || !RunCompiledModel) {
            __android_log_print(ANDROID_LOG_ERROR, "DepthPipeline",
                "Missing critical LiteRT functions");
            return false;
        }

        return true;
    }

    void unload() {
        if (handle) {
            dlclose(handle);
            handle = nullptr;
        }
    }
};

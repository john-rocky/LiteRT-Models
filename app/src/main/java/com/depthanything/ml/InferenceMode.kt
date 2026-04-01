package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    NHWC_GPU_FP32(
        "GPU FP32",
        "NHWC model, GPU delegate FP32"
    ),
    NHWC_GPU_FP16(
        "GPU FP16",
        "NHWC model, GPU delegate FP16 (may NaN)"
    ),
    FP16W_GPU_FP32(
        "GPU FP16w",
        "FP16 weight model, GPU FP32 compute"
    ),
    CLAMPED_GPU_FP16(
        "GPU clamp+FP16",
        "Clamped model, GPU FP16 compute"
    ),
    CLAMPED_FP16W_GPU_FP16(
        "GPU clamp+FP16w",
        "Clamped FP16 weight, GPU FP16 compute"
    ),
    ONNX_RUNTIME(
        "ONNX Runtime",
        "ONNX Runtime CPU, 4 threads"
    );
}

package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    GPU_FP32(
        "GPU FP32",
        "No clamp (reference, will break)"
    ),
    GPU_C60K_FP32W(
        "clamp60k FP32w",
        "Clamp ±60k, FP32 weights (99MB)"
    ),
    GPU_C60K_FP16W(
        "clamp60k FP16w",
        "Clamp ±60k, FP16 weights (49MB)"
    ),
    ONNX_RUNTIME(
        "ONNX (ground truth)",
        "ONNX FP32 CPU - reference quality"
    );
}

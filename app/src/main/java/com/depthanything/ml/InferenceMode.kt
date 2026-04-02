package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    CLAMPED_FP16W(
        "Clamped FP16w (7ms)",
        "518x518, onnx2tf clamped, FP16 weight, ML Drift"
    ),
    CLAMPED_FP32W(
        "Clamped FP32w",
        "518x518, onnx2tf clamped, FP32 weight, ML Drift"
    ),
    NHWC_518(
        "NHWC 518 no-clamp",
        "518x518, onnx2tf, no clamp, ML Drift"
    ),
    ONNX_CPU(
        "ONNX CPU (truth)",
        "ONNX FP32 CPU = PyTorch"
    );
}

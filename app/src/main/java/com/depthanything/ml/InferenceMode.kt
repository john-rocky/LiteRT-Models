package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    GPU_FP32_TRUTH(
        "GPU FP32 (truth)",
        "Old delegate FP32 - ground truth"
    ),
    DRIFT_ATTN(
        "Drift Attn (64)",
        "ML Drift + Attention clamp"
    ),
    DRIFT_ALL(
        "Drift All (112)",
        "ML Drift + All ops clamp"
    ),
    ONNX_RUNTIME(
        "ONNX (truth)",
        "ONNX FP32 CPU"
    );
}

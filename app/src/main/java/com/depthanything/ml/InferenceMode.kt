package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    CLAMP_ATTN(
        "Attention (64)",
        "Softmax + LN + Attention MatMul only"
    ),
    CLAMP_ALL(
        "All ops (112)",
        "Softmax + LN + ALL MatMul"
    ),
    ONNX_RUNTIME(
        "ONNX (truth)",
        "ONNX FP32 CPU - reference"
    );
}

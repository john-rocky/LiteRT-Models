package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    // Native Keras models (corr≈1.0, ML Drift GPU)
    KERAS_NATIVE(
        "Keras 518 GPU",
        "518x518, Native Keras, ML Drift GPU FP32"
    ),
    KERAS_392(
        "Keras 392x518 GPU",
        "392x518, Native Keras, ML Drift GPU FP32"
    ),

    // CPU reference
    ONNX_CPU(
        "ONNX CPU (truth)",
        "ONNX FP32 CPU = PyTorch"
    ),

    // --- Legacy modes (for benchmarking / investigation) ---
//    CLAMPED_FP16W(
//        "Clamped FP16w (7ms)",
//        "518x518, onnx2tf clamped, FP16 weight, ML Drift"
//    ),
//    CLAMPED_FP32W(
//        "Clamped FP32w",
//        "518x518, onnx2tf clamped, FP32 weight, ML Drift"
//    ),
//    NHWC_518(
//        "NHWC 518 no-clamp",
//        "518x518, onnx2tf, no clamp, ML Drift"
//    ),
//    DIRECT_NHWC(
//        "Direct NHWC (corr=1.0)",
//        "litert-torch NHWC, corr=1.0"
//    ),
//    V129(
//        "v129 (onnx2tf 1.29)",
//        "onnx2tf 1.29.24, ML Drift GPU"
//    ),
//    V129_CLAMPED(
//        "v129 clamped",
//        "onnx2tf 1.29.24 clamped, ML Drift GPU"
//    ),
//    ONNX_XNNPACK(
//        "ONNX XNNPACK",
//        "ONNX FP32 XNNPACK CPU accelerated"
//    ),
    ;
}

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
    DIRECT_NHWC(
        "Direct NHWC (corr=1.0)",
        "litert-torch NHWC, corr=1.0"
    ),

    // Native Keras model (corr=0.999998, no conversion artifacts)
    KERAS_NATIVE(
        "Keras Native (corr=1.0)",
        "Native TF/Keras NHWC, ML Drift GPU"
    ),
    KERAS_NATIVE_FP16W(
        "Keras Native FP16w",
        "Native TF/Keras NHWC, FP16 weights, ML Drift GPU"
    ),

    // onnx2tf 1.29 variant (same quality as 1.28, kept for comparison)
    V129(
        "v129 (onnx2tf 1.29)",
        "onnx2tf 1.29.24, ML Drift GPU"
    ),
    V129_CLAMPED(
        "v129 clamped",
        "onnx2tf 1.29.24 clamped, ML Drift GPU"
    ),

    // ONNX Runtime modes
    ONNX_CPU(
        "ONNX CPU (truth)",
        "ONNX FP32 CPU = PyTorch"
    ),
    ONNX_XNNPACK(
        "ONNX XNNPACK",
        "ONNX FP32 XNNPACK CPU accelerated"
    );
}

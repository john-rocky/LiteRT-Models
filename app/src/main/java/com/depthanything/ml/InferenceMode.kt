package com.depthanything.ml

enum class InferenceMode(val label: String, val description: String) {
    NHWC_GPU(
        "NHWC GPU",
        "Our NHWC model + GPU delegate"
    ),
    QUALCOMM_GPU(
        "Qualcomm GPU",
        "Qualcomm NHWC model + GPU (norm baked in)"
    ),
    ONNX_RUNTIME(
        "ONNX Runtime",
        "ONNX Runtime CPU, 4 threads"
    );

    // Commented out - not needed for current investigation
    // NHWC_GPU_FP16("NHWC GPU FP16", "FP16 - causes NaN overflow")
    // NHWC_XNNPACK("NHWC XNNPACK", "CPU XNNPACK - ~2700ms")
    // TFLITE_GPU_FP32("NCHW GPU FP32", "NCHW on GPU - doesn't accelerate")
    // TFLITE_XNNPACK_FP16("NCHW XNNPACK", "CPU XNNPACK - ~2700ms")
    // TFLITE_CPU_FP32("TFLite CPU raw", "No delegate - ~15000ms")
}

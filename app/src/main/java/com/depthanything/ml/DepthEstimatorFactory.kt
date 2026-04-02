package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_NHWC = "depth_anything_v2_nhwc.tflite"
    private const val MODEL_C60K_FP32 = "depth_anything_v2_c60k_fp32.tflite"
    private const val MODEL_C60K_FP16W = "depth_anything_v2_c60k_fp16w.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.GPU_FP32 -> TFLiteDepthEstimator(context, mode, MODEL_NHWC)
            InferenceMode.GPU_C60K_FP32W -> TFLiteDepthEstimator(context, mode, MODEL_C60K_FP32)
            InferenceMode.GPU_C60K_FP16W -> TFLiteDepthEstimator(context, mode, MODEL_C60K_FP16W)
            InferenceMode.ONNX_RUNTIME -> OnnxDepthEstimator(context, MODEL_ONNX, optimized = false)
        }
    }

    fun availableModes(context: Context): List<InferenceMode> {
        val assetFiles = context.assets.list("")?.toSet() ?: emptySet()
        return InferenceMode.entries.filter { mode ->
            when (mode) {
                InferenceMode.GPU_FP32 -> MODEL_NHWC in assetFiles
                InferenceMode.GPU_C60K_FP32W -> MODEL_C60K_FP32 in assetFiles
                InferenceMode.GPU_C60K_FP16W -> MODEL_C60K_FP16W in assetFiles
                InferenceMode.ONNX_RUNTIME -> MODEL_ONNX in assetFiles
            }
        }
    }
}

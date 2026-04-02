package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_CLAMPED_FP16W = "depth_anything_v2_nhwc_clamped_fp16w.tflite"
    private const val MODEL_CLAMPED_FP32W = "depth_anything_v2_nhwc_clamped.tflite"
    private const val MODEL_NHWC_518 = "depth_anything_v2_nhwc_518.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.CLAMPED_FP16W -> TFLiteDepthEstimator(context, mode, MODEL_CLAMPED_FP16W)
            InferenceMode.CLAMPED_FP32W -> TFLiteDepthEstimator(context, mode, MODEL_CLAMPED_FP32W)
            InferenceMode.NHWC_518 -> TFLiteDepthEstimator(context, mode, MODEL_NHWC_518)
            InferenceMode.ONNX_CPU -> OnnxDepthEstimator(context, MODEL_ONNX, optimized = false)
        }
    }

    fun availableModes(context: Context): List<InferenceMode> {
        val assetFiles = context.assets.list("")?.toSet() ?: emptySet()
        return InferenceMode.entries.filter { mode ->
            when (mode) {
                InferenceMode.CLAMPED_FP16W -> MODEL_CLAMPED_FP16W in assetFiles
                InferenceMode.CLAMPED_FP32W -> MODEL_CLAMPED_FP32W in assetFiles
                InferenceMode.NHWC_518 -> MODEL_NHWC_518 in assetFiles
                InferenceMode.ONNX_CPU -> MODEL_ONNX in assetFiles
            }
        }
    }
}

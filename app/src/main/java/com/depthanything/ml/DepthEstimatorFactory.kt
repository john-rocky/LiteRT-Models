package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_ATTN = "depth_anything_v2_attn.tflite"
    private const val MODEL_ALL = "depth_anything_v2_c60k_fp32.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.CLAMP_ATTN -> TFLiteDepthEstimator(context, mode, MODEL_ATTN)
            InferenceMode.CLAMP_ALL -> TFLiteDepthEstimator(context, mode, MODEL_ALL)
            InferenceMode.ONNX_RUNTIME -> OnnxDepthEstimator(context, MODEL_ONNX, optimized = false)
        }
    }

    fun availableModes(context: Context): List<InferenceMode> {
        val assetFiles = context.assets.list("")?.toSet() ?: emptySet()
        return InferenceMode.entries.filter { mode ->
            when (mode) {
                InferenceMode.CLAMP_ATTN -> MODEL_ATTN in assetFiles
                InferenceMode.CLAMP_ALL -> MODEL_ALL in assetFiles
                InferenceMode.ONNX_RUNTIME -> MODEL_ONNX in assetFiles
            }
        }
    }
}

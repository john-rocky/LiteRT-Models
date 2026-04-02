package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_CLAMPED_FP16W = "depth_anything_v2_nhwc_clamped_fp16w.tflite"
    private const val MODEL_CLAMPED_FP32W = "depth_anything_v2_nhwc_clamped.tflite"
    private const val MODEL_NHWC_518 = "depth_anything_v2_nhwc_518.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"
    private const val MODEL_KERAS = "depth_anything_v2_keras.tflite"
    private const val MODEL_KERAS_FP16W = "depth_anything_v2_keras_fp16w.tflite"
    private const val MODEL_V129 = "depth_anything_v2_v129.tflite"
    private const val MODEL_V129_CLAMPED = "depth_anything_v2_v129_clamped.tflite"

    private fun modelFile(mode: InferenceMode): String = when (mode) {
        InferenceMode.CLAMPED_FP16W -> MODEL_CLAMPED_FP16W
        InferenceMode.CLAMPED_FP32W -> MODEL_CLAMPED_FP32W
        InferenceMode.NHWC_518 -> MODEL_NHWC_518
        InferenceMode.DIRECT_NHWC -> "depth_anything_v2_nhwc_direct.tflite"
        InferenceMode.KERAS_NATIVE -> MODEL_KERAS
        InferenceMode.KERAS_NATIVE_FP16W -> MODEL_KERAS_FP16W
        InferenceMode.V129 -> MODEL_V129
        InferenceMode.V129_CLAMPED -> MODEL_V129_CLAMPED
        InferenceMode.ONNX_CPU, InferenceMode.ONNX_XNNPACK -> MODEL_ONNX
    }

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.ONNX_CPU -> OnnxDepthEstimator(context, MODEL_ONNX, useXnnpack = false)
            InferenceMode.ONNX_XNNPACK -> OnnxDepthEstimator(context, MODEL_ONNX, useXnnpack = true)
            else -> TFLiteDepthEstimator(context, mode, modelFile(mode))
        }
    }

    fun availableModes(context: Context): List<InferenceMode> {
        val assetFiles = context.assets.list("")?.toSet() ?: emptySet()
        return InferenceMode.entries.filter { mode ->
            modelFile(mode) in assetFiles
        }
    }
}

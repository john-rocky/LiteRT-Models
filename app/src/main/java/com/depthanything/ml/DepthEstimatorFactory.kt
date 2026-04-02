package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_KERAS = "depth_anything_v2_keras.tflite"
    private const val MODEL_KERAS_392 = "depth_anything_v2_keras_392x518.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"

    private fun modelFile(mode: InferenceMode): String = when (mode) {
        InferenceMode.KERAS_NATIVE -> MODEL_KERAS
        InferenceMode.KERAS_392 -> MODEL_KERAS_392
        InferenceMode.ONNX_CPU -> MODEL_ONNX
    }

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.ONNX_CPU -> OnnxDepthEstimator(context, MODEL_ONNX)
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

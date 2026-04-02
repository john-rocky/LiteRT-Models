package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_KERAS = "depth_anything_v2_keras.tflite"
    private const val MODEL_KERAS_392 = "depth_anything_v2_keras_392x518.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"

    private fun modelFile(mode: InferenceMode): String = when (mode) {
        InferenceMode.KERAS_NATIVE -> MODEL_KERAS
        InferenceMode.KERAS_392 -> MODEL_KERAS_392
        InferenceMode.INTERPRETER_GPU -> MODEL_KERAS_392
        InferenceMode.ONNX_CPU -> MODEL_ONNX
    }

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.ONNX_CPU -> OnnxDepthEstimator(context, MODEL_ONNX)
            // All GPU modes now use InterpreterDepthEstimator for fast ByteBuffer readback.
            // CompiledModel (TFLiteDepthEstimator) requires litert:2.1.3 which lacks
            // addDelegate() — switch deps in build.gradle.kts to re-enable.
            else -> InterpreterDepthEstimator(context, mode, modelFile(mode))
        }
    }

    fun availableModes(context: Context): List<InferenceMode> {
        val assetFiles = context.assets.list("")?.toSet() ?: emptySet()
        return InferenceMode.entries.filter { mode ->
            modelFile(mode) in assetFiles
        }
    }
}

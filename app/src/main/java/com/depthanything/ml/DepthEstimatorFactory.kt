package com.depthanything.ml

import android.content.Context

object DepthEstimatorFactory {

    private const val MODEL_NHWC = "depth_anything_v2_nhwc.tflite"
    private const val MODEL_FP16W = "depth_anything_v2_nhwc_fp16w.tflite"
    private const val MODEL_CLAMPED = "depth_anything_v2_nhwc_clamped.tflite"
    private const val MODEL_CLAMPED_FP16W = "depth_anything_v2_nhwc_clamped_fp16w.tflite"
    private const val MODEL_QUALCOMM = "depth_anything_v2_qualcomm.tflite"
    private const val MODEL_ONNX = "depth_anything_v2.onnx"

    fun create(context: Context, mode: InferenceMode): DepthEstimator {
        return when (mode) {
            InferenceMode.NHWC_GPU_FP32 -> TFLiteDepthEstimator(context, mode, MODEL_NHWC)
            InferenceMode.NHWC_GPU_FP16 -> TFLiteDepthEstimator(context, mode, MODEL_NHWC)
            InferenceMode.FP16W_GPU_FP32 -> TFLiteDepthEstimator(context, mode, MODEL_FP16W)
            InferenceMode.CLAMPED_GPU_FP16 -> TFLiteDepthEstimator(context, mode, MODEL_CLAMPED)
            InferenceMode.CLAMPED_FP16W_GPU_FP16 -> TFLiteDepthEstimator(context, mode, MODEL_CLAMPED_FP16W)
            InferenceMode.QUALCOMM_GPU -> TFLiteDepthEstimator(context, mode, MODEL_QUALCOMM)
            InferenceMode.ONNX_RUNTIME -> OnnxDepthEstimator(context, MODEL_ONNX, optimized = false)
        }
    }

    fun availableModes(context: Context): List<InferenceMode> {
        val assetFiles = context.assets.list("")?.toSet() ?: emptySet()
        return InferenceMode.entries.filter { mode ->
            when (mode) {
                InferenceMode.NHWC_GPU_FP32 -> MODEL_NHWC in assetFiles
                InferenceMode.NHWC_GPU_FP16 -> MODEL_NHWC in assetFiles
                InferenceMode.FP16W_GPU_FP32 -> MODEL_FP16W in assetFiles
                InferenceMode.CLAMPED_GPU_FP16 -> MODEL_CLAMPED in assetFiles
                InferenceMode.CLAMPED_FP16W_GPU_FP16 -> MODEL_CLAMPED_FP16W in assetFiles
                InferenceMode.QUALCOMM_GPU -> MODEL_QUALCOMM in assetFiles
                InferenceMode.ONNX_RUNTIME -> MODEL_ONNX in assetFiles
            }
        }
    }
}

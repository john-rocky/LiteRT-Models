package com.depthanything.ml

import android.graphics.Bitmap

data class DepthResult(
    val depthMap: Bitmap,
    val inferenceTimeMs: Long,
    val mode: InferenceMode
)

interface DepthEstimator {
    val mode: InferenceMode
    fun predict(bitmap: Bitmap): DepthResult
    fun close()
}

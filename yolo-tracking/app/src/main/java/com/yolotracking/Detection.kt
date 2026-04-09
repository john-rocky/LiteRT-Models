package com.yolotracking

data class Detection(
    val classId: Int,
    val className: String,
    val score: Float,
    val xMin: Float,
    val yMin: Float,
    val xMax: Float,
    val yMax: Float,
    val feature: FloatArray? = null,   // Re-ID embedding (512-dim)
    val trackId: Int = -1,             // assigned by tracker (-1 = untracked)
) {
    val cx get() = (xMin + xMax) / 2f
    val cy get() = (yMin + yMax) / 2f
    val w get() = xMax - xMin
    val h get() = yMax - yMin
}

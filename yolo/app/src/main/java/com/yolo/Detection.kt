package com.yolo

data class Detection(
    val classId: Int,
    val className: String,
    val score: Float,
    val xMin: Float,
    val yMin: Float,
    val xMax: Float,
    val yMax: Float,
)

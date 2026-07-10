package com.ssdlite

/** A single detection in original-bitmap pixel coordinates. */
data class Detection(
    val classId: Int,      // 1..90 (torchvision COCO label; 0 = background, never emitted)
    val className: String,
    val score: Float,
    val xMin: Float,
    val yMin: Float,
    val xMax: Float,
    val yMax: Float,
)

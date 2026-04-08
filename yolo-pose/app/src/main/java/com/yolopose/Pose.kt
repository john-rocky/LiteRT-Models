package com.yolopose

/**
 * One person detection with bounding box and 17 COCO keypoints.
 * Coordinates are in original image pixels.
 */
data class Pose(
    val score: Float,
    val xMin: Float,
    val yMin: Float,
    val xMax: Float,
    val yMax: Float,
    /** 17 keypoints, each (x, y, conf). x/y in original image pixels, conf in [0,1]. */
    val keypoints: FloatArray,
) {
    fun keypointX(i: Int): Float = keypoints[i * 3]
    fun keypointY(i: Int): Float = keypoints[i * 3 + 1]
    fun keypointConf(i: Int): Float = keypoints[i * 3 + 2]
}

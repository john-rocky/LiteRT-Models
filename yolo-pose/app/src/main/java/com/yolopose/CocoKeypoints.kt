package com.yolopose

/**
 * COCO 17-keypoint definition used by Ultralytics YOLO pose models.
 */
object CocoKeypoints {

    const val NUM_KEYPOINTS = 17

    val NAMES = arrayOf(
        "nose",            // 0
        "left_eye",        // 1
        "right_eye",       // 2
        "left_ear",        // 3
        "right_ear",       // 4
        "left_shoulder",   // 5
        "right_shoulder",  // 6
        "left_elbow",      // 7
        "right_elbow",     // 8
        "left_wrist",      // 9
        "right_wrist",     // 10
        "left_hip",        // 11
        "right_hip",       // 12
        "left_knee",       // 13
        "right_knee",      // 14
        "left_ankle",      // 15
        "right_ankle",     // 16
    )

    /**
     * Skeleton edges as (kp_a, kp_b, color) triples.
     * Colors loosely group limb / torso / face for visual clarity.
     */
    private val LIMB_L = 0xFF00CCFFu.toInt()  // light blue — left limbs
    private val LIMB_R = 0xFFFFB300u.toInt()  // amber — right limbs
    private val TORSO  = 0xFFE040FBu.toInt()  // magenta — torso/hips
    private val FACE   = 0xFF69F0AEu.toInt()  // green — face

    val EDGES: Array<IntArray> = arrayOf(
        // Face
        intArrayOf(0, 1, FACE),
        intArrayOf(0, 2, FACE),
        intArrayOf(1, 3, FACE),
        intArrayOf(2, 4, FACE),
        // Left arm
        intArrayOf(5, 7, LIMB_L),
        intArrayOf(7, 9, LIMB_L),
        // Right arm
        intArrayOf(6, 8, LIMB_R),
        intArrayOf(8, 10, LIMB_R),
        // Shoulders / torso
        intArrayOf(5, 6, TORSO),
        intArrayOf(5, 11, TORSO),
        intArrayOf(6, 12, TORSO),
        intArrayOf(11, 12, TORSO),
        // Left leg
        intArrayOf(11, 13, LIMB_L),
        intArrayOf(13, 15, LIMB_L),
        // Right leg
        intArrayOf(12, 14, LIMB_R),
        intArrayOf(14, 16, LIMB_R),
    )

    /** Color for keypoint dots (left/right asymmetric, face white). */
    val POINT_COLORS: IntArray = IntArray(NUM_KEYPOINTS) { i ->
        when (i) {
            0 -> 0xFFFFFFFFu.toInt()
            1, 3 -> LIMB_L
            2, 4 -> LIMB_R
            5, 7, 9, 11, 13, 15 -> LIMB_L
            6, 8, 10, 12, 14, 16 -> LIMB_R
            else -> 0xFFFFFFFFu.toInt()
        }
    }
}

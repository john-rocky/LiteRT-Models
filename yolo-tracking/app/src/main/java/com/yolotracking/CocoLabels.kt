package com.yolotracking

object CocoLabels {
    private val NAMES = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    )

    fun name(classId: Int): String = NAMES.getOrElse(classId) { "class$classId" }

    // 20 distinct colors for track IDs
    private val COLORS = intArrayOf(
        0xFFFF6B6B.toInt(), 0xFF4ECDC4.toInt(), 0xFF45B7D1.toInt(), 0xFFFFA07A.toInt(),
        0xFF98D8C8.toInt(), 0xFFF7DC6F.toInt(), 0xFFBB8FCE.toInt(), 0xFF85C1E9.toInt(),
        0xFFF8C471.toInt(), 0xFF82E0AA.toInt(), 0xFFD7BDE2.toInt(), 0xFFA3E4D7.toInt(),
        0xFFE59866.toInt(), 0xFFF1948A.toInt(), 0xFF73C6B6.toInt(), 0xFFAED6F1.toInt(),
        0xFFFAD7A0.toInt(), 0xFFA9DFBF.toInt(), 0xFFD2B4DE.toInt(), 0xFFA9CCE3.toInt(),
    )

    fun color(classId: Int): Int = COLORS[classId % COLORS.size]

    fun trackColor(trackId: Int): Int = COLORS[trackId % COLORS.size]
}

package com.yolox

object CocoLabels {
    // YOLOX uses the 80 contiguous COCO classes (index 0-79).
    private val NAMES = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    )

    fun name(classId: Int): String = NAMES.getOrElse(classId) { "obj$classId" }

    /** Deterministic per-class color (ARGB), evenly spread around the hue wheel. */
    fun color(classId: Int): Int {
        val hue = (classId * 47 % 360).toFloat()
        return android.graphics.Color.HSVToColor(floatArrayOf(hue, 0.85f, 1.0f))
    }
}

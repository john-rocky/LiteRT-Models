package com.ssdlite

object CocoLabels {
    // torchvision COCO label map (91 entries, index 0 = background). The "N/A"
    // entries are unused COCO category-id gaps — they keep label ids aligned with
    // the model's 91-way classifier so labelId directly indexes this array.
    private val NAMES = arrayOf(
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
        "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
        "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    )

    fun name(classId: Int): String = NAMES.getOrElse(classId) { "obj$classId" }

    /** Deterministic per-class color (ARGB), evenly spread around the hue wheel. */
    fun color(classId: Int): Int {
        val hue = (classId * 47 % 360).toFloat()
        return android.graphics.Color.HSVToColor(floatArrayOf(hue, 0.85f, 1.0f))
    }
}

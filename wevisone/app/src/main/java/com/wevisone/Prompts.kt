package com.wevisone

/**
 * The instructions the released checkpoint was trained with (`wevisone_vl.prompts.build_prompt` of
 * github.com/Tencent/WeVisOne-VL at 40f735b8) and the chat turn they go into.
 */
enum class Task(val label: String, val hint: String, val needsQuery: Boolean, val multiLabel: Boolean) {
    CUT_OUT("Cut out", "What to cut out, e.g. the man on the left", true, false),
    DETECT("Detect", "Labels, one per line or comma-separated, e.g. person, hat", true, true),
    FIND("Find one", "A description, e.g. the yellow rectangle", true, false),
    POINT("Point", "Labels, one per line or comma-separated, e.g. airplane, windsock", true, true),
    COUNT_BOXES("Count (boxes)", "What to count, e.g. blue circles", true, false),
    COUNT_POINTS("Count (points)", "What to count, e.g. people", true, false),
    READ_TEXT("Read text", "No words needed: every word in the photo is read", false, false),
    FIND_TEXT("Find text", "The text to find, e.g. EXIT 12", true, false);

    /** The package's task name for this choice. */
    val packageTask: String
        get() = when (this) {
            CUT_OUT -> "matting"
            DETECT -> "detection"
            FIND -> "referring"
            POINT -> "point"
            COUNT_BOXES -> "count_box"
            COUNT_POINTS -> "count_point"
            READ_TEXT -> "ocr"
            FIND_TEXT -> "ocr_grounding"
        }

    companion object {
        fun fromPackageTask(name: String): Task = when (name.trim().lowercase()) {
            "matting", "segmentation" -> CUT_OUT
            "detection" -> DETECT
            "referring" -> FIND
            "point" -> POINT
            "count_box" -> COUNT_BOXES
            "count_point" -> COUNT_POINTS
            "ocr" -> READ_TEXT
            "ocr_grounding" -> FIND_TEXT
            else -> throw IllegalArgumentException("unknown task $name")
        }
    }
}

object Prompts {
    const val SYSTEM = "You are a helpful assistant."

    /** Everything before the 256 image positions: the fixed system turn, then the user turn with the image first. */
    const val BEFORE_IMAGE = "<|im_start|>system\n$SYSTEM<|im_end|>\n<|im_start|>user\n<|vision_start|>"

    /** Everything after the image: the instruction, then the assistant turn the model completes. */
    fun afterImage(instruction: String): String = "<|vision_end|>$instruction<|im_end|>\n<|im_start|>assistant\n"

    /**
     * The package's build_prompt(task, query) for [task]. Multi-label tasks take one label per line (the package's
     * own rule) or per comma (this app's input rule); the other tasks take the whole text as one phrase.
     */
    fun instruction(task: Task, query: String): String {
        val text = query.trim()
        return when (task) {
            Task.DETECT ->
                "Locate all the instances that match the following description: " + labels(query).joinToString(". ") + "."
            Task.FIND -> "Locate the instances that match the following description: " + phrase(text)
            Task.POINT -> "point to: " + labels(query).joinToString(". ")
            Task.COUNT_BOXES -> "How many ${text.ifEmpty { "objects" }} in the image? Please count them with boxes."
            Task.COUNT_POINTS -> "How many ${text.ifEmpty { "objects" }} in the image?"
            Task.CUT_OUT ->
                "Locate the instances that match the following description and provide the alpha mattes: " + phrase(text)
            Task.READ_TEXT -> "detect all the word in the box format"
            Task.FIND_TEXT -> "Locate the following text in the image: $text"
        }
    }

    /** `_labels`: the non-empty stripped lines (and here also comma-separated parts). */
    fun labels(query: String): List<String> =
        query.split('\n', ',').map { it.trim() }.filter { it.isNotEmpty() }

    /** `_phrase`: the stripped text with a final "." unless it already ends in . ? or !. */
    fun phrase(text: String): String =
        if (text.isEmpty() || text.last() in ".?!") text else "$text."
}

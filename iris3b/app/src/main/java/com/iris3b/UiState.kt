package com.iris3b

import android.graphics.Bitmap
import androidx.annotation.StringRes
import androidx.compose.runtime.Immutable

/**
 * The example prompts under the editor. The verify entry makes these three from the reference noise
 * and scores them against the PyTorch pipeline's images, so their text must stay byte for byte the
 * prompts of that reference run.
 */
enum class Example(val key: String, @param:StringRes val label: Int, val text: String) {
    APPLE("apple", R.string.example_apple, "a red apple on a wooden table, studio lighting"),
    CUBES(
        "punct_num",
        R.string.example_cubes,
        "three blue glass cubes, two red spheres & one green cone on a white table; 45° top light, 50mm f/1.8 (sharp focus!)",
    ),
    // 　 = the ideographic (full-width) space of the reference prompt.
    LIGHTHOUSE("japanese", R.string.example_lighthouse, "夕暮れの海辺に立つ白い灯台、　柔らかな光と長い影"),
}

/** Whether the model files are on the phone and open. */
sealed interface ModelState {
    data object Checking : ModelState

    /** [files] = the missing names (only the config when the config itself is missing). */
    data class Missing(val files: List<String>) : ModelState

    data class Failed(val error: String) : ModelState

    data object Ready : ModelState
}

/** One snapshot of the screen. */
@Immutable
data class UiState(
    val model: ModelState = ModelState.Checking,
    val prompt: String = "",
    /** Caption tokens of [prompt] (null until counted). */
    val promptTokens: Int? = null,
    val maxPromptTokens: Int = 295,
    val seedText: String = "0",
    val running: Boolean = false,
    val stage: String = "",
    val progress: Float = 0f,
    val elapsedSeconds: Int = 0,
    val image: Bitmap? = null,
    val imageInfo: String? = null,
    val message: String? = null,
    val messageIsError: Boolean = false,
) {
    val ready: Boolean
        get() = model == ModelState.Ready

    val canGenerate: Boolean
        get() = ready && !running && prompt.isNotBlank()
}

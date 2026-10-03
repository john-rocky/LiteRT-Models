package com.kev

import java.io.File

/**
 * The external files `scripts/install_to_device.sh` copies into the app's `files/`, named as in the
 * model repository. Change a name here and in the install script together.
 */
object KevFiles {
  /** The byte-level BPE tokenizer published with the Kev checkpoint. */
  const val TOKENIZER = "tokenizer.json"

  /** The pointer head: q / k weight and bias, float32. */
  const val HEAD = "kev_0.8b_pointer_head.safetensors"

  /** The graph windows (sequence lengths) the model repository publishes, in ascending order. */
  val WINDOWS = listOf(128, 256, 512, 1024, 2048)

  /** The windows the install script copies when `WINDOWS` is not set. */
  val DEFAULT_INSTALL = listOf(256, 512)

  /** The row-prefill graph of [window] tokens (fp16 fully connected weights, int8 embedding). */
  fun graph(window: Int): String = "kev-0.8b_rowprefill_L${window}_fp16fc_i8emb.tflite"

  /** The [WINDOWS] whose graph is in [directory], in ascending order. */
  fun installedWindows(directory: File): List<Int> = WINDOWS.filter {
    File(directory, graph(it)).isFile
  }

  /**
   * Files a launch needs that are not in [directory]: the tokenizer, the head, and the graphs of
   * [DEFAULT_INSTALL] when no graph is installed at all.
   */
  fun missing(directory: File): List<String> =
    listOf(TOKENIZER, HEAD).filterNot { File(directory, it).isFile } +
      if (installedWindows(directory).isEmpty()) DEFAULT_INSTALL.map { graph(it) } else emptyList()

  /** Files a launch on the one [window] needs that are not in [directory]. */
  fun missing(directory: File, window: Int): List<String> =
    listOf(TOKENIZER, HEAD, graph(window)).filterNot { File(directory, it).isFile }
}

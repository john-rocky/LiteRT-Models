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

  /** The graph window loaded at startup unless a launch asks for another. */
  const val DEFAULT_WINDOW = 512

  /** The row-prefill graph of [window] tokens (fp16 fully connected weights, int8 embedding). */
  fun graph(window: Int): String = "kev-0.8b_rowprefill_L${window}_fp16fc_i8emb.tflite"

  /** Files a launch at [window] needs: tokenizer, head and that window's graph. */
  fun required(window: Int): List<String> = listOf(TOKENIZER, HEAD, graph(window))

  /** The [required] files missing from [directory]. */
  fun missing(directory: File, window: Int): List<String> = required(window).filterNot { File(directory, it).isFile }
}

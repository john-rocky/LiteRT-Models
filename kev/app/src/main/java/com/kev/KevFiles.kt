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
  val WINDOWS = listOf(64, 128, 256, 512, 1024, 2048)

  /** The windows the install script copies when `WINDOWS` is not set. */
  val DEFAULT_INSTALL = listOf(128, 256)

  /** The shared-state pairs the install script copies when `PAIR` is not set. */
  val DEFAULT_PAIRS = listOf(KevPairShape(128, 64))

  /** The shared-state pairs the app plans with: the shapes measured on the Galaxy S26. */
  val PAIRS = listOf(KevPairShape(128, 64), KevPairShape(256, 64))

  /**
   * File sizes of the graphs the model repository published before the fp16-safe kernel rewrite. A
   * file of one of these sizes runs at FP32 by default ([KevPrecision.defaultFor]).
   */
  // Their kernel gives non-finite values on some rows at FP16_WITH_FP32_ACCUM.
  val PRE_REWRITE_BYTES: Map<KevGraphKey, Long> =
    mapOf(
      KevGraphKey.Window(512) to 1_264_068_368L,
      KevGraphKey.Window(1024) to 1_269_023_216L,
      KevGraphKey.Window(2048) to 1_285_227_888L,
    )

  /** The row-prefill graph of [window] tokens (fp16 fully connected weights, int8 embedding). */
  fun graph(window: Int): String = "kev-0.8b_rowprefill_L${window}_fp16fc_i8emb.tflite"

  /**
   * The shared-state pair of [stateLength] state tokens and [questionLength] question tokens: one
   * file with the signatures `state_prefill_<Ls>` and `question_step_<Ls>_<Lq>`.
   */
  fun pair(stateLength: Int, questionLength: Int): String =
    "kev-0.8b_sharedstate_Ls${stateLength}_Lq${questionLength}_fp16fc_i8emb.tflite"

  fun pair(shape: KevPairShape): String = pair(shape.stateLength, shape.questionLength)

  /** The [WINDOWS] whose graph is in [directory], in ascending order. */
  fun installedWindows(directory: File): List<Int> = WINDOWS.filter {
    File(directory, graph(it)).isFile
  }

  /** The [PAIRS] whose file is in [directory]. */
  fun installedPairs(directory: File): List<KevPairShape> = PAIRS.filter {
    File(directory, pair(it)).isFile
  }

  /**
   * Files a launch needs that are not in [directory]: the tokenizer, the head, and the graphs of
   * [DEFAULT_INSTALL] and [DEFAULT_PAIRS] when no graph (window or pair) is installed at all.
   */
  fun missing(directory: File): List<String> =
    listOf(TOKENIZER, HEAD).filterNot { File(directory, it).isFile } +
      if (installedWindows(directory).isEmpty() && installedPairs(directory).isEmpty()) {
        DEFAULT_INSTALL.map { graph(it) } + DEFAULT_PAIRS.map { pair(it) }
      } else {
        emptyList()
      }

  /** Files a launch on the one [graph] needs that are not in [directory]. */
  fun missing(directory: File, graph: KevGraphKey): List<String> =
    listOf(TOKENIZER, HEAD, graph.file).filterNot { File(directory, it).isFile }
}

/**
 * The two windows of a shared-state pair: [stateLength] (Ls) tokens for `[state] + state tokens`
 * and [questionLength] (Lq) tokens for one question's branch.
 */
data class KevPairShape(val stateLength: Int, val questionLength: Int)

/** One graph the app compiles: a row-prefill window or a shared-state pair. */
sealed interface KevGraphKey {
  /** The graph's file in `files/`. */
  val file: String

  /** The graph's name in logs and reports: "L256" or "S128+Q64". */
  val label: String

  data class Window(val window: Int) : KevGraphKey {
    override val file: String
      get() = KevFiles.graph(window)

    override val label: String
      get() = "L$window"
  }

  data class Pair(val shape: KevPairShape) : KevGraphKey {
    override val file: String
      get() = KevFiles.pair(shape)

    override val label: String
      get() = "S${shape.stateLength}+Q${shape.questionLength}"
  }
}

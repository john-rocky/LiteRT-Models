package com.gliner25

import androidx.annotation.StringRes

/**
 * The two published GLiNER2.5 LiteRT packages this sample runs. Both keep the five-label host
 * contract, the `serving_default` signature with inputs `args_0`..`args_4` and one packed output,
 * N/T windows 128/48, 256/192 and 512/384, and the 16-tensor sparse decoder. They differ in encoder
 * width, vocabulary, embedding-table storage, packed length, installed files and word splitters.
 *
 * [SMALL] keeps the files and host path validated by this sample's Small gates. [MULTI] follows
 * the GLiNER2.5-Multi-LiteRT host contract: the caller picks the whitespace or char splitter, the
 * float16 table's selected rows are upcast to float32, and gliner2 2.0.0's terminal period applies.
 */
enum class GlinerModel(
  /** Launch-extra and report identifier. */
  val id: String,
  @param:StringRes val title: Int,
  @param:StringRes val shortTitle: Int,
  /** Subdirectory of filesDir with this model's installed files; empty means filesDir itself. */
  val directory: String,
  private val graphPrefix: String,
  val hiddenSize: Int,
  val vocabularySize: Int,
  val tableFile: String,
  /** True for a little-endian float16 table, false for float32. */
  val halfPrecisionTable: Boolean,
  private val packedPerTextSlot: Int,
  private val packedBase: Int,
  /** Bundled checkpoint configuration, or null when `config.json` is installed with the files. */
  val configAsset: String?,
  val splitters: List<GlinerInputs.Splitter>,
  val defaultSplitter: GlinerInputs.Splitter,
  /**
   * gliner2 2.0.0 `processor.py:_collate_batch` appends `.` unless the text ends in `.`, `!` or
   * `?`. Multi follows it. Small keeps the host port its gates validated; every Small fixture
   * ends in `.`, where both behaviors agree.
   */
  val appendsTerminalPeriod: Boolean,
  /** Windows whose graphs must be installed before loading; others are required on first use. */
  val startupWindows: List<Int>,
  /** Installer invocation named in missing-file messages. */
  val installCommand: String,
) {
  SMALL(
    "small",
    R.string.model_small_title,
    R.string.model_small,
    "",
    "gliner25_small",
    384,
    128011,
    "word_embeddings_fp32.bin",
    false,
    1108,
    4574,
    "gliner_config.json",
    listOf(GlinerInputs.Splitter.WHITESPACE),
    GlinerInputs.Splitter.WHITESPACE,
    false,
    listOf(128, 256, 512),
    "scripts/install_to_device.sh",
  ),
  MULTI(
    "multi",
    R.string.model_multi_title,
    R.string.model_multi,
    "multi",
    "gliner25_multi",
    768,
    250112,
    "word_embeddings_fp16.bin",
    true,
    1492,
    6494,
    null,
    listOf(GlinerInputs.Splitter.WHITESPACE, GlinerInputs.Splitter.CHAR),
    GlinerInputs.Splitter.CHAR,
    true,
    listOf(128),
    "scripts/install_to_device.sh --multi",
  );

  /** Headerless row-major table size: rows × hidden × 2 or 4 bytes. */
  val tableBytes: Long
    get() =
      vocabularySize.toLong() *
        hiddenSize *
        if (halfPrecisionTable) {
          2
        } else {
          Float.SIZE_BYTES
        }

  /** Published `[1,1,1,P]` packed output length for text capacity T. */
  fun packedFloatCount(textCapacity: Int): Int = packedPerTextSlot * textCapacity + packedBase

  fun graphFile(window: Int): String = "${graphPrefix}_s${window}_wfp16.tflite"

  /** Files checked before any resource opens, in the order missing files are reported. */
  val requiredFiles: List<String>
    get() =
      startupWindows.map(::graphFile) +
        listOf(tableFile, "tokenizer.json", "sparse_decoder_fp32.safetensors") +
        listOfNotNull("config.json".takeIf { configAsset == null }) +
        GlinerInputs.WINDOWS.map { "graph_contract_s${it.sequenceLength}.json" }

  /** Prefilled sentence for [splitter]: Multi pairs English with whitespace, Japanese with char. */
  @StringRes
  fun exampleText(splitter: GlinerInputs.Splitter): Int =
    when {
      this == SMALL -> R.string.example_text
      splitter == GlinerInputs.Splitter.CHAR -> R.string.example_text_multi_ja
      else -> R.string.example_text_multi_en
    }

  companion object {
    fun fromId(id: String): GlinerModel =
      entries.firstOrNull { it.id == id.lowercase(java.util.Locale.ROOT) }
        ?: throw IllegalArgumentException("Unknown model: $id (use small or multi)")
  }
}

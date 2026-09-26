package com.gliformer

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.regex.Pattern

/** Pinned GLiFormer 0.1.2 plain-text NER prompt and the shipped HOST_CONTRACT.md routing. */
class GliformerInputs(private val tokenizer: GliformerTokenizer) {
  data class Window(val sequenceLength: Int, val textCapacity: Int) {
    init {
      require(
        sequenceLength == 128 && textCapacity == 48 ||
          sequenceLength == 256 && textCapacity == 256 ||
          sequenceLength == 512 && textCapacity == 512
      ) {
        "Unsupported GLiFormer graph window"
      }
    }

    val packedFloatCount: Int
      get() = textCapacity * 5 * 3
  }

  /** Half-open offsets in Python Unicode code points, never Kotlin UTF-16 code units. */
  data class Word(val text: String, val start: Int, val end: Int)

  /** All graph arrays are flattened C-order; embeddings are looked up separately, including PAD. */
  data class Prepared(
    val text: String,
    val labels: List<String>,
    val window: Window,
    val inputIds: IntArray,
    val attentionMask: FloatArray,
    val textRouting: FloatArray,
    val parentRouting: FloatArray,
    val labelRouting: FloatArray,
    val textMask: FloatArray,
    val firstSubtokenPositions: IntArray,
    val parentPositions: IntArray,
    val entityPositions: IntArray,
    val words: List<Word>,
    val encodedLength: Int,
  ) {
    fun substring(start: Int, end: Int): String = codePointSubstring(text, start, end)
  }

  /**
   * The processor passes each entry as one pre-split word: [SEQ], [SCHEMA], five
   * "[ENTITY] label" entries, [SEP], [SEP], then each original-case text word.
   * [SEQ] is NOT an added token in this package. It encodes to three ordinary IDs.
   * The tokenizer's single-sequence postprocessor adds outer CLS and SEP.
   * Selects the smallest window satisfying BOTH limits. No truncation occurs.
   */
  fun prepare(text: String, labels: List<String> = LABELS, window: Window? = null): Prepared {
    require(
      labels.size == 5 &&
        labels.distinct().size == 5 &&
        labels.all { NON_WHITESPACE_PATTERN.matcher(it).find() }
    ) {
      "The static graph requires exactly five distinct nonempty label strings"
    }
    val words = splitWords(text)
    require(words.isNotEmpty()) { "Expected one nonempty text" }
    require(words.size <= 512) {
      "Input has ${words.size} text-word slots; package limits are 512 encoded tokens including " +
        "the label prompt and 512 text-word slots"
    }
    val ids = ArrayList<Int>()
    fun add(word: String) {
      ids.addAll(tokenizer.encodeWord(word).asList())
    }
    ids.add(tokenizer.clsId)
    add("[SEQ]")
    add("[SCHEMA]")
    labels.forEach { add("[ENTITY] $it") }
    add("[SEP]") // End of this NER task's prompt.
    add("[SEP]") // End of the orchestrator's combined prompt.
    val positions = IntArray(words.size)
    words.forEachIndexed { index, word ->
      positions[index] = ids.size
      val encoded = tokenizer.encodeWord(word.text)
      require(encoded.isNotEmpty()) { "Text word $index has no subtokens" }
      ids.addAll(encoded.asList())
    }
    ids.add(tokenizer.sepId)
    val parents = ids.indices.filter { ids[it] == tokenizer.schemaId }.toIntArray()
    val entities = ids.indices.filter { ids[it] == tokenizer.entityId }.toIntArray()
    require(parents.size == 1 && entities.size == labels.size) {
      "Expected one schema marker and five entity markers"
    }
    val selected =
      window
        ?: WINDOWS.firstOrNull {
          ids.size <= it.sequenceLength && words.size <= it.textCapacity
        }
        ?: throw IllegalArgumentException(
          "Input has ${ids.size} encoded tokens including the label prompt and ${words.size} " +
            "text-word slots; package limits are 512 encoded tokens and 512 text-word slots. " +
            "Split the text before calling prepare."
        )
    require(ids.size <= selected.sequenceLength && words.size <= selected.textCapacity) {
      "Input has ${ids.size} encoded tokens and ${words.size} text-word slots; " +
        "s${selected.sequenceLength} limits are ${selected.sequenceLength} encoded tokens and " +
        "${selected.textCapacity} text-word slots"
    }
    val n = selected.sequenceLength
    val t = selected.textCapacity
    val paddedIds = IntArray(n) { tokenizer.padId }
    ids.forEachIndexed { index, id -> paddedIds[index] = id }
    val attention = FloatArray(n) { if (it < ids.size) 1f else 0f }
    val textRoute = FloatArray(t * n)
    positions.forEachIndexed { index, position -> textRoute[index * n + position] = 1f }
    val parentRoute = FloatArray(n)
    parentRoute[parents.single()] = 1f
    val labelRoute = FloatArray(labels.size * n)
    entities.forEachIndexed { index, position -> labelRoute[index * n + position] = 1f }
    val textMask = FloatArray(t) { if (it < words.size) 1f else 0f }
    return Prepared(
      text,
      labels.toList(),
      selected,
      paddedIds,
      attention,
      textRoute,
      parentRoute,
      labelRoute,
      textMask,
      positions,
      parents,
      entities,
      words,
      ids.size
    )
  }

  /**
   * Headerless little-endian [128008,1024] fp16/fp32 table, mapped read-only outside Java heap.
   * Only requested rows are copied/upcast; padding ID 0 receives its real embedding too.
   * Closing releases the file descriptor; the VM manages the mapped buffer's lifetime.
   */
  class EmbeddingTable(file: File, val storage: Storage = Storage.FP32) : Closeable {
    enum class Storage(val bytesPerValue: Int) {
      FP16(2),
      FP32(4)
    }

    private val channel: FileChannel
    private val values: ByteBuffer

    init {
      val expected = VOCABULARY_SIZE.toLong() * HIDDEN_SIZE * storage.bytesPerValue
      require(file.length() == expected) {
        "Embedding table must be [128008,1024] ${storage.name.lowercase()}, $expected bytes"
      }
      channel = RandomAccessFile(file, "r").channel
      try {
        values =
          channel.map(FileChannel.MapMode.READ_ONLY, 0, expected).order(ByteOrder.LITTLE_ENDIAN)
      } catch (failure: Throwable) {
        channel.close()
        throw failure
      }
    }

    fun lookup(ids: IntArray): FloatArray {
      require(channel.isOpen) { "Embedding table is closed" }
      require(ids.all { it in 0 until VOCABULARY_SIZE }) { "Tokenizer ID outside embedding table" }
      val output = FloatArray(ids.size * HIDDEN_SIZE)
      ids.forEachIndexed { row, id ->
        val source = id * HIDDEN_SIZE * storage.bytesPerValue
        val target = row * HIDDEN_SIZE
        for (column in 0 until HIDDEN_SIZE) {
          output[target + column] =
            when (storage) {
              Storage.FP32 -> values.getFloat(source + column * 4)
              Storage.FP16 -> halfToFloat(values.getShort(source + column * 2).toInt() and 0xffff)
            }
        }
      }
      return output
    }

    override fun close() = channel.close()

    companion object {
      /** Exact IEEE 754 binary16 → binary32 conversion, including subnormals and signed zero. */
      internal fun halfToFloat(bits: Int): Float {
        val sign = (bits and 0x8000) shl 16
        val exponent = (bits ushr 10) and 0x1f
        var fraction = bits and 0x3ff
        if (exponent == 0) {
          if (fraction == 0) return Float.fromBits(sign)
          var unbiased = -14
          while (fraction and 0x400 == 0) {
            fraction = fraction shl 1;
            unbiased--
          }
          return Float.fromBits(sign or ((unbiased + 127) shl 23) or ((fraction and 0x3ff) shl 13))
        }
        if (exponent == 31) return Float.fromBits(sign or 0x7f800000 or (fraction shl 13))
        return Float.fromBits(sign or ((exponent + 112) shl 23) or (fraction shl 13))
      }
    }
  }

  companion object {
    val LABELS: List<String> = listOf("person", "organization", "location", "product", "date")
    val WINDOWS: List<Window> = listOf(Window(128, 48), Window(256, 256), Window(512, 512))
    const val HIDDEN_SIZE = 1024
    const val VOCABULARY_SIZE = 128008

    /**
     * gliner 0.2.29 WhitespaceTokenSplitter: Python's `\w+(?:[-_]\w+)*|\S`.
     * Python \w is letters/numbers/underscore, not Java's Unicode combining-mark class.
     * Explicit Python whitespace also includes U+001C..U+001F. Android has no
     * UNICODE_CHARACTER_CLASS flag. Convert matcher UTF-16 indices using codePointCount.
     */
    fun splitWords(text: String): List<Word> {
      val words = ArrayList<Word>()
      val matcher = WORD_PATTERN.matcher(text)
      while (matcher.find()) {
        words.add(
          Word(
            matcher.group(),
            text.codePointCount(0, matcher.start()),
            text.codePointCount(0, matcher.end())
          )
        )
      }
      return words
    }

    /** Translate a Python code-point interval before using Kotlin's UTF-16 substring API. */
    fun codePointSubstring(text: String, start: Int, end: Int): String {
      require(start >= 0 && end >= start && end <= text.codePointCount(0, text.length)) {
        "Invalid Unicode code-point interval [$start,$end)"
      }
      return text.substring(text.offsetByCodePoints(0, start), text.offsetByCodePoints(0, end))
    }

    private const val SPACE =
      "\\x{09}-\\x{0d}\\x{1c}-\\x{20}\\x{85}\\x{a0}\\x{1680}" +
        "\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"
    private const val WORD = "\\p{L}\\p{N}_"
    private val WORD_PATTERN = Pattern.compile("[$WORD]+(?:[-_][$WORD]+)*|[^$SPACE]")
    // Python str.strip() includes U+0085 and U+001C..U+001F, unlike Kotlin isBlank().
    private val NON_WHITESPACE_PATTERN = Pattern.compile("[^$SPACE]")
  }
}

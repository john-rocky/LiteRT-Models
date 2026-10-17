package com.gliner25

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.regex.Pattern

/**
 * Ports the fixed NER path of gliner2 2.0.0
 * `processor.py:SchemaTransformer._format_input_with_mapping` and the published runtime's
 * `shaped_graph.py:fixed_inputs`.
 *
 * Windows use encoded capacity N and text-word capacity T: 128/48, 256/192 and 512/384. Both limits
 * must fit; truncation would invalidate routing and character offsets. The five graph inputs are
 * float32 embeddings `[1,N,hidden]`, attention `[1,N]`, text routing `[1,T,N]`, query routing
 * `[1,5,N]` and text mask `[1,T]`, flattened in row-major order. [appendsTerminalPeriod] applies
 * `processor.py:_collate_batch`'s `.` for text that does not end in `.`, `!` or `?`.
 */
class GlinerInputs(
  private val tokenizer: GlinerTokenizer,
  private val appendsTerminalPeriod: Boolean = false,
) {
  /**
   * gliner2 2.0.0 `processing/word_splitter.py` splitters. The caller chooses per request from the
   * text's known language; there is no detection. Both lowercase only token values and report
   * Unicode code-point offsets into the processed text.
   */
  enum class Splitter(val id: String) {
    /** `WhitespaceTokenSplitter`, for space-delimited text such as English. */
    WHITESPACE("whitespace"),
    /** `CharLevelSplitter`, for Japanese and other text without word spaces. */
    CHAR("char");

    fun split(text: String): List<Word> =
      when (this) {
        WHITESPACE -> splitWords(text)
        CHAR -> splitCharacters(text)
      }

    companion object {
      fun fromId(id: String): Splitter =
        entries.firstOrNull { it.id == id.lowercase(Locale.ROOT) }
          ?: throw IllegalArgumentException("Unknown splitter: $id (use whitespace or char)")
    }
  }

  /**
   * The published `shaped_graph.py:Shape` capacities, N/T = 128/48, 256/192 or 512/384. The
   * GLiNER2.5 Small packed graph output has `1108*T+4574` floats ([GlinerModel.packedFloatCount]
   * gives each model's count); capacities are not interchangeable.
   */
  data class Window(val sequenceLength: Int, val textCapacity: Int) {
    init {
      require(
        sequenceLength == 128 && textCapacity == 48 ||
          sequenceLength == 256 && textCapacity == 192 ||
          sequenceLength == 512 && textCapacity == 384
      ) {
        "Unsupported graph window"
      }
    }

    val packedFloatCount: Int
      get() = 1108 * textCapacity + 4574
  }

  /**
   * A gliner2 2.0.0 `processing/word_splitter.py:WhitespaceTokenSplitter` token. Lowercasing
   * changes only [text]; half-open [start]/[end] offsets count Unicode code points in the original
   * string, so Java UTF-16 indices must not be substituted.
   */
  data class Word(val text: String, val start: Int, val end: Int)

  /**
   * Host metadata from `SchemaTransformer._format_input_with_mapping` plus fixed graph padding. IDs
   * and attention have N elements, text routing T*N, query routing 5*N and text mask T. Only the
   * first subword of each word or structural query marker receives a routing one. Keeping the
   * processed [text] and [words] alongside the arrays lets decoding recover exact source spans;
   * [text] differs from the caller's only by an appended terminal period.
   */
  data class Prepared(
    val text: String,
    val window: Window,
    val inputIds: IntArray,
    val attentionMask: FloatArray,
    val textRouting: FloatArray,
    val queryRouting: FloatArray,
    val textMask: FloatArray,
    val textWordPositions: IntArray,
    val queryMarkerPositions: IntArray,
    val words: List<Word>,
    val encodedLength: Int,
  ) {
    /** Converts gliner2's Unicode code-point interval to a source substring, preserving UTF-16. */
    fun substring(start: Int, end: Int): String = codePointSubstring(text, start, end)
  }

  /**
   * Ports gliner2 2.0.0 `SchemaTransformer._format_input_with_mapping` and `fixed_inputs` for the
   * ordered person, organization, location, product and date schema.
   *
   * Selects the smallest fitting N/T window unless [window] is supplied. Schema tokens count toward
   * N; padding uses the tokenizer's PAD ID and zero routing/masks. Oversized input is rejected
   * because dropping tokens would change both predictions and source offsets. [splitter] defines
   * the text slots, including the appended period's slot when [appendsTerminalPeriod] is set.
   */
  fun prepare(
    text: String,
    window: Window? = null,
    splitter: Splitter = Splitter.WHITESPACE,
  ): Prepared {
    val processed =
      if (appendsTerminalPeriod) {
        withTerminalPeriod(text)
      } else {
        text
      }
    val words = splitter.split(processed)
    val ids = ArrayList<Int>()
    val queries = ArrayList<Int>(LABELS.size)
    fun add(token: String) {
      ids.addAll(tokenizer.encodeWord(token).asList())
    }
    add("(")
    add("[P]")
    add("entities")
    add("(")
    for (label in LABELS) {
      queries.add(ids.size)
      add("[E]")
      add(label)
    }
    add(")")
    add(")")
    add("[SEP_TEXT]")
    val textPositions = IntArray(words.size)
    for ((index, word) in words.withIndex()) {
      textPositions[index] = ids.size
      add(word.text)
    }
    val selected =
      window
        ?: WINDOWS.firstOrNull {
          ids.size <= it.sequenceLength && words.size <= it.textCapacity
        }
        ?: throw IllegalArgumentException(
          "Input exceeds published windows: ${ids.size} encoded tokens, ${words.size} text words"
        )
    require(ids.size <= selected.sequenceLength && words.size <= selected.textCapacity) {
      "Input does not fit s${
selected.sequenceLength
}: ${ids.size} encoded tokens, ${words.size} text words"
    }
    val n = selected.sequenceLength
    val t = selected.textCapacity
    val paddedIds = IntArray(n) { tokenizer.padId }
    ids.forEachIndexed { index, id -> paddedIds[index] = id }
    val attention =
      FloatArray(n) {
        if (it < ids.size) {
          1f
        } else {
          0f
        }
      }
    val textRouting = FloatArray(t * n)
    textPositions.forEachIndexed { index, position -> textRouting[index * n + position] = 1f }
    val queryRouting = FloatArray(LABELS.size * n)
    queries.forEachIndexed { index, position -> queryRouting[index * n + position] = 1f }
    val textMask =
      FloatArray(t) {
        if (it < words.size) {
          1f
        } else {
          0f
        }
      }
    return Prepared(
      processed,
      selected,
      paddedIds,
      attention,
      textRouting,
      queryRouting,
      textMask,
      textPositions,
      queries.toIntArray(),
      words,
      ids.size,
    )
  }

  /**
   * Reproduces the published `fixed_inputs` word-embedding lookup used with gliner2 2.0.0. The
   * headerless little-endian table (Small: `[128011,384]` float32, 196624896 bytes; Multi:
   * `[250112,768]` float16, 384172032 bytes) is memory-mapped because it need not be copied to
   * the Java heap. Float16 rows are upcast to float32 only when selected. Padding IDs are looked
   * up too.
   */
  class EmbeddingTable(
    file: File,
    private val vocabularySize: Int = VOCABULARY_SIZE,
    private val hiddenSize: Int = HIDDEN_SIZE,
    halfPrecision: Boolean = false,
  ) : Closeable {
    private val channel = RandomAccessFile(file, "r").channel
    private val mapped =
      channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.LITTLE_ENDIAN)
    private val floats: FloatBuffer? =
      if (halfPrecision) {
        null
      } else {
        mapped.asFloatBuffer()
      }
    private val halves: ShortBuffer? =
      if (halfPrecision) {
        mapped.asShortBuffer()
      } else {
        null
      }

    init {
      val storage =
        if (halfPrecision) {
          "fp16"
        } else {
          "fp32"
        }
      val bytesPerValue =
        if (halfPrecision) {
          2
        } else {
          Float.SIZE_BYTES
        }
      require(channel.size() == vocabularySize.toLong() * hiddenSize * bytesPerValue) {
        "Embedding table is not the published [$vocabularySize,$hiddenSize] $storage table"
      }
    }

    /**
     * Returns row-major `[1,N,hidden]` float32 embeddings for N padded IDs, matching
     * `fixed_inputs`. Absolute buffer reads keep table position independent of the caller's
     * sequence length.
     */
    fun lookup(ids: IntArray): FloatArray {
      val out = FloatArray(ids.size * hiddenSize)
      for ((row, id) in ids.withIndex()) {
        require(id in 0 until vocabularySize) { "Tokenizer ID outside embedding table: $id" }
        val source = id * hiddenSize
        val target = row * hiddenSize
        if (floats != null) {
          for (column in 0 until hiddenSize) {
            out[target + column] = floats.get(source + column)
          }
        } else {
          val values = requireNotNull(halves)
          for (column in 0 until hiddenSize) {
            out[target + column] = halfToFloat(values.get(source + column).toInt())
          }
        }
      }
      return out
    }

    /** Releases the file channel when the host runtime no longer needs embedding lookups. */
    override fun close() = channel.close()
  }

  companion object {
    val LABELS: List<String> = listOf("person", "organization", "location", "product", "date")
    val WINDOWS: List<Window> = listOf(Window(128, 48), Window(256, 192), Window(512, 384))
    /** GLiNER2.5 Small table defaults; [GlinerModel] supplies each model's values. */
    const val HIDDEN_SIZE = 384
    const val VOCABULARY_SIZE = 128011

    /** gliner2 2.0.0 `_collate_batch`: append `.` unless the text ends in `.`, `!` or `?`. */
    fun withTerminalPeriod(text: String): String =
      if (text.endsWith('.') || text.endsWith('!') || text.endsWith('?')) {
        text
      } else {
        "$text."
      }

    /**
     * Ports gliner2 2.0.0 `processing/word_splitter.py:CharLevelSplitter.__call__`, the `char`
     * splitter of the Multi host contract: each ASCII `[A-Za-z0-9@._\-+]+` run is one slot and
     * every other non-space code point is its own slot. Python's Unicode `\S` is spelled out as
     * for the whitespace port. Offsets count code points; only token values are lowercased.
     */
    fun splitCharacters(text: String): List<Word> {
      val result = ArrayList<Word>()
      val matcher = CHARACTER_PATTERN.matcher(text)
      while (matcher.find()) {
        result.add(
          Word(
            matcher.group().lowercase(Locale.ROOT),
            text.codePointCount(0, matcher.start()),
            text.codePointCount(0, matcher.end()),
          )
        )
      }
      return result
    }

    /**
     * Exact IEEE 754 binary16 to binary32 conversion of one table value. Every float16 value,
     * subnormals included, is a float32 value, so the upcast adds no rounding. Pure Kotlin keeps
     * it identical on Android and in JVM tests.
     */
    fun halfToFloat(bits: Int): Float {
      val sign = (bits and 0x8000) shl 16
      val exponent = (bits ushr 10) and 0x1f
      var mantissa = bits and 0x3ff
      val single =
        when {
          exponent == 0x1f -> sign or 0x7f800000 or (mantissa shl 13)
          exponent != 0 -> sign or ((exponent + 112) shl 23) or (mantissa shl 13)
          mantissa == 0 -> sign
          else -> {
            // A subnormal half becomes a normal float: shift until the implicit bit appears.
            var biased = 113
            while (mantissa and 0x400 == 0) {
              mantissa = mantissa shl 1
              biased--
            }
            sign or (biased shl 23) or ((mantissa and 0x3ff) shl 13)
          }
        }
      return Float.fromBits(single)
    }

    /**
     * Ports gliner2 2.0.0 `processing/word_splitter.py:WhitespaceTokenSplitter.__call__`, the
     * default splitter selected by `processor.py:SchemaTransformer._tokenize_text`. Python's
     * Unicode word/space classes are explicit because Java's `\w` includes combining marks and its
     * default `\s` is ASCII-only. Original code-point offsets survive lowercasing, including
     * characters whose lowercase mapping changes length.
     */
    fun splitWords(text: String): List<Word> {
      val result = ArrayList<Word>()
      val matcher = WORD_PATTERN.matcher(text)
      while (matcher.find()) {
        // Lower only the token value. Lowercasing the source first can
        // change its length (for example U+0130) and corrupt offsets.
        result.add(
          Word(
            matcher.group().lowercase(Locale.ROOT),
            text.codePointCount(0, matcher.start()),
            text.codePointCount(0, matcher.end()),
          )
        )
      }
      return result
    }

    /**
     * Bridges the Unicode code-point offsets produced by gliner2's `WhitespaceTokenSplitter` to
     * Kotlin's UTF-16 substring API, so supplementary characters do not shift spans.
     */
    fun codePointSubstring(text: String, start: Int, end: Int): String {
      require(start >= 0 && end >= start)
      return text.substring(text.offsetByCodePoints(0, start), text.offsetByCodePoints(0, end))
    }

    private const val SPACE =
      "\\x{09}-\\x{0d}\\x{1c}-\\x{20}\\x{85}\\x{a0}\\x{1680}" +
        "\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"
    private const val WORD = "\\p{L}\\p{N}_"
    private val WORD_PATTERN =
      Pattern.compile(
        "(?:https?://[^$SPACE]+|www\\.[^$SPACE]+)" +
          "|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}" +
          "|@[a-z0-9_]+" +
          "|[$WORD]+(?:[-_][$WORD]+)*" +
          "|[^$SPACE]",
        Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
      )
    // Case-sensitive like the Python pattern; the ASCII class already lists both cases.
    private val CHARACTER_PATTERN = Pattern.compile("[A-Za-z0-9@._\\-+]+|[^$SPACE]")
  }
}

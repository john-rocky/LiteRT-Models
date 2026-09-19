package com.gliner25

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.regex.Pattern

/**
 * Ports the fixed English NER path of gliner2 2.0.0
 * `processor.py:SchemaTransformer._format_input_with_mapping` and the published runtime's
 * `shaped_graph.py:fixed_inputs`.
 *
 * Windows use encoded capacity N and text-word capacity T: 128/48, 256/192 and 512/384. Both limits
 * must fit; truncation would invalidate routing and character offsets. The five graph inputs are
 * float32 embeddings `[1,N,384]`, attention `[1,N]`, text routing `[1,T,N]`, query routing
 * `[1,5,N]` and text mask `[1,T]`, flattened in row-major order.
 */
class GlinerInputs(private val tokenizer: GlinerTokenizer) {
  /**
   * The published `shaped_graph.py:Shape` capacities, N/T = 128/48, 256/192 or 512/384. The packed
   * graph output has `1108*T+4574` floats; capacities are not interchangeable.
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
   * original [words] alongside the arrays lets decoding recover exact source spans.
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
   * because dropping tokens would change both predictions and source offsets.
   */
  fun prepare(text: String, window: Window? = null): Prepared {
    val words = splitWords(text)
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
      text,
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
   * headerless `[128011,384]` little-endian float32 table is memory-mapped because its 196624896
   * bytes need not be copied to the Java heap. Padding IDs are looked up too.
   */
  class EmbeddingTable(file: File) : Closeable {
    private val channel = RandomAccessFile(file, "r").channel
    private val values =
      channel
        .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
        .order(ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()

    init {
      require(channel.size() == VOCABULARY_SIZE.toLong() * HIDDEN_SIZE * Float.SIZE_BYTES) {
        "Embedding table is not the published [128011,384] fp32 table"
      }
    }

    /**
     * Returns row-major `[1,N,384]` embeddings for N padded IDs, matching `fixed_inputs`. Absolute
     * buffer reads keep table position independent of the caller's sequence length.
     */
    fun lookup(ids: IntArray): FloatArray {
      val out = FloatArray(ids.size * HIDDEN_SIZE)
      for ((row, id) in ids.withIndex()) {
        require(id in 0 until VOCABULARY_SIZE) { "Tokenizer ID outside embedding table: $id" }
        val source = id * HIDDEN_SIZE
        val target = row * HIDDEN_SIZE
        for (column in 0 until HIDDEN_SIZE) {
          out[target + column] = values.get(source + column)
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
    const val HIDDEN_SIZE = 384
    const val VOCABULARY_SIZE = 128011

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
  }
}

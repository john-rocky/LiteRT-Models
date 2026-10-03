package com.kev

import java.io.File
import java.text.Normalizer
import java.util.PriorityQueue
import java.util.regex.Pattern

/**
 * Kotlin port of the tokenizer Kev runs, read from the `tokenizer.json` published with the Kev
 * checkpoint: transformers 5.17's `Qwen2Tokenizer` pipeline over the Qwen3.5-0.8B-Base vocabulary,
 * called with `add_special_tokens=False`. `encode` follows the `tokenizers` library's order:
 * 1. split the raw text on the added tokens (all `normalized: false`, so matched before NFC),
 *    leftmost-longest;
 * 2. NFC-normalize each remaining segment;
 * 3. split it with the pre-tokenizer regex, keeping matches and the text between them (`Isolated`);
 * 4. map each piece's UTF-8 bytes to GPT-2's byte characters and apply the BPE merges by rank.
 *
 * No prefix space is added (`add_prefix_space: false`) and nothing is wrapped around the ids (the
 * post-processor is ByteLevel). The file's regex is compiled after two rewrites that keep its
 * meaning under java.util.regex and Android's ICU: `\s` / `\S` become onig's Unicode White_Space
 * class (Java's `\s` is ASCII-only and Android rejects `UNICODE_CHARACTER_CLASS`), and the
 * case-insensitive contractions spell out onig's Unicode case folding (`'ſ` matches `'s`) instead
 * of relying on regex flags: the JVM's `(?i)` folds ASCII only, and Unicode folding is enabled
 * differently by java.util.regex and ICU.
 */
class KevTokenizer(tokenizerJson: File) {
  private class TrieNode {
    val children = HashMap<Char, TrieNode>()
    var tokenId = NO_TOKEN
  }

  private class Candidate(
    val rank: Int,
    val left: Int,
    val right: Int,
    val leftId: Int,
    val rightId: Int,
    val mergedId: Int,
  )

  private val addedIds = LinkedHashMap<String, Int>()
  private val addedTokens = TrieNode()
  private val merges: MergeTable
  // Vocabulary ID of each byte's GPT-2 character.
  private val byteIds = IntArray(BYTE_VALUES)
  private val splitPattern: Pattern

  /** The pre-tokenizer regex exactly as tokenizer.json spells it. */
  val pretokenizerRegex: String

  /** Number of token IDs: the BPE vocabulary plus the added tokens. */
  val vocabularySize: Int

  /** Entries of the BPE vocabulary. */
  val bpeVocabularySize: Int

  /** Added tokens, in file order, with their IDs. */
  val addedTokenIds: Map<String, Int>
    get() = addedIds

  init {
    val loaded = Loader(KevJsonReader(tokenizerJson.readBytes())).load()
    pretokenizerRegex = loaded.regex
    require(pretokenizerRegex == TOKENIZER_REGEX) {
      "tokenizer.json pre-tokenizer regex differs from the Kev tokenizer's: $pretokenizerRegex"
    }
    splitPattern = Pattern.compile(javaPattern(pretokenizerRegex))
    val vocabulary = loaded.vocabulary
    bpeVocabularySize = vocabulary.size
    val bytes = byteCharacters()
    for (byte in 0 until BYTE_VALUES) {
      byteIds[byte] =
        requireNotNull(vocabulary[bytes[byte].toString()]) { "No vocabulary entry for byte $byte" }
    }
    merges = MergeTable(loaded.mergeLefts.size)
    for (rank in loaded.mergeLefts.indices) {
      val left = loaded.mergeLefts[rank]
      val right = loaded.mergeRights[rank]
      val leftId = requireNotNull(vocabulary[left]) { "Merge $rank: unknown token $left" }
      val rightId = requireNotNull(vocabulary[right]) { "Merge $rank: unknown token $right" }
      val merged = requireNotNull(vocabulary[left + right]) { "Merge $rank: no token $left$right" }
      // As tokenizers' merge map: a repeated pair keeps its last rank.
      merges.put(leftId, rightId, rank, merged)
    }
    var size = (vocabulary.values.maxOrNull() ?: -1) + 1
    for ((content, id) in loaded.addedTokens) {
      insert(content, id)
      addedIds[content] = id
      size = maxOf(size, id + 1)
    }
    vocabularySize = size
  }

  /** ID of an added token (`<|fim_prefix|>`, `<|endoftext|>`, …), or null. */
  fun tokenId(token: String): Int? = addedIds[token]

  /** Token IDs of [text], as `tokenizer(text, add_special_tokens=False).input_ids`. */
  fun encode(text: String): IntArray {
    val ids = ArrayList<Int>(text.length / CHARACTERS_PER_TOKEN_ESTIMATE + 1)
    splitOnAddedTokens(
      text,
      onText = { segment -> splitPieces(normalize(segment)) { piece -> bpe(piece, ids) } },
      onToken = { ids.add(it) },
    )
    return ids.toIntArray()
  }

  /** Splits [text] on the added tokens (leftmost-longest, non-overlapping). */
  private fun splitOnAddedTokens(text: String, onText: (String) -> Unit, onToken: (Int) -> Unit) {
    var unprocessed = 0
    var position = 0
    while (position < text.length) {
      var node = addedTokens
      var end = position
      var matchedId = NO_TOKEN
      var matchedEnd = position
      while (end < text.length) {
        node = node.children[text[end]] ?: break
        end++
        if (node.tokenId != NO_TOKEN) {
          matchedId = node.tokenId
          matchedEnd = end
        }
      }
      if (matchedId == NO_TOKEN) {
        position++
        continue
      }
      if (position > unprocessed) {
        onText(text.substring(unprocessed, position))
      }
      onToken(matchedId)
      position = matchedEnd
      unprocessed = matchedEnd
    }
    if (unprocessed < text.length) {
      onText(text.substring(unprocessed))
    }
  }

  /** The `Isolated` Split pre-tokenizer: every regex match and every gap between matches. */
  private fun splitPieces(text: String, onPiece: (String) -> Unit) {
    val matcher = splitPattern.matcher(text)
    var last = 0
    while (matcher.find()) {
      if (matcher.start() > last) {
        onPiece(text.substring(last, matcher.start()))
      }
      onPiece(matcher.group())
      last = matcher.end()
    }
    if (last < text.length) {
      onPiece(text.substring(last))
    }
  }

  /**
   * BPE on the UTF-8 bytes of one piece: start from each byte's character, then repeatedly apply
   * the lowest-rank merge, leftmost first, as `tokenizers`' `Word::merge_all`.
   */
  private fun bpe(piece: String, output: MutableList<Int>) {
    val bytes = piece.toByteArray(Charsets.UTF_8)
    if (bytes.isEmpty()) return
    val ids = IntArray(bytes.size) { byteIds[bytes[it].toInt() and BYTE_MASK] }
    if (ids.size == 1) {
      output.add(ids[0])
      return
    }
    val previous = IntArray(ids.size) { it - 1 }
    val next = IntArray(ids.size) { if (it + 1 < ids.size) it + 1 else -1 }
    val alive = BooleanArray(ids.size) { true }
    val queue = PriorityQueue<Candidate>(compareBy<Candidate> { it.rank }.thenBy { it.left })
    fun offer(left: Int) {
      if (left < 0 || !alive[left]) return
      val right = next[left]
      if (right < 0) return
      val slot = merges.find(ids[left], ids[right])
      if (slot < 0) return
      queue.add(
        Candidate(merges.rank(slot), left, right, ids[left], ids[right], merges.mergedId(slot))
      )
    }
    for (index in 0 until ids.size - 1) {
      offer(index)
    }
    while (queue.isNotEmpty()) {
      val candidate = queue.remove()
      val left = candidate.left
      val right = candidate.right
      // Skip entries made stale by an earlier merge on either side.
      if (
        !alive[left] ||
          !alive[right] ||
          next[left] != right ||
          ids[left] != candidate.leftId ||
          ids[right] != candidate.rightId
      ) {
        continue
      }
      ids[left] = candidate.mergedId
      alive[right] = false
      next[left] = next[right]
      if (next[right] >= 0) {
        previous[next[right]] = left
      }
      offer(previous[left])
      offer(left)
    }
    for (index in ids.indices) {
      if (alive[index]) {
        output.add(ids[index])
      }
    }
  }

  private fun insert(content: String, id: Int) {
    require(content.isNotEmpty()) { "Empty added token" }
    var node = addedTokens
    for (character in content) {
      node = node.children.getOrPut(character) { TrieNode() }
    }
    node.tokenId = id
  }

  /** Open-addressing map from a (left, right) ID pair to the merge's rank and merged ID. */
  private class MergeTable(entries: Int) {
    private val capacity = Integer.highestOneBit(maxOf(entries, 1) * 2) * 2
    private val keys = LongArray(capacity) { EMPTY }
    private val ranks = IntArray(capacity)
    private val mergedIds = IntArray(capacity)

    fun put(left: Int, right: Int, rank: Int, mergedId: Int) {
      val key = key(left, right)
      var slot = start(key)
      while (keys[slot] != EMPTY && keys[slot] != key) {
        slot = (slot + 1) and (capacity - 1)
      }
      keys[slot] = key
      ranks[slot] = rank
      mergedIds[slot] = mergedId
    }

    /** Slot of the pair, or -1. */
    fun find(left: Int, right: Int): Int {
      val key = key(left, right)
      var slot = start(key)
      while (keys[slot] != EMPTY) {
        if (keys[slot] == key) return slot
        slot = (slot + 1) and (capacity - 1)
      }
      return -1
    }

    fun rank(slot: Int): Int = ranks[slot]

    fun mergedId(slot: Int): Int = mergedIds[slot]

    private fun start(key: Long): Int = ((key * HASH_MULTIPLIER) ushr HASH_SHIFT).toInt() and (capacity - 1)

    private fun key(left: Int, right: Int): Long = (left.toLong() shl 32) or right.toLong()

    private companion object {
      const val EMPTY = -1L
      // Fibonacci hashing: the 64-bit golden-ratio multiplier spreads consecutive IDs.
      const val HASH_MULTIPLIER = -7046029254386353131L
      const val HASH_SHIFT = 32
    }
  }

  /** What `init` needs from tokenizer.json, streamed so the vocabulary never becomes a tree. */
  private class Loaded(
    val regex: String,
    val vocabulary: HashMap<String, Int>,
    val mergeLefts: List<String>,
    val mergeRights: List<String>,
    val addedTokens: List<Pair<String, Int>>,
  )

  private class Loader(private val reader: KevJsonReader) {
    private var regex: String? = null
    private var vocabulary: HashMap<String, Int>? = null
    private val mergeLefts = ArrayList<String>()
    private val mergeRights = ArrayList<String>()
    private val addedTokens = ArrayList<Pair<String, Int>>()
    private var sawModel = false

    fun load(): Loaded {
      reader.beginObject()
      while (reader.hasNext()) {
        when (val name = reader.nextName()) {
          "added_tokens" -> readAddedTokens(reader.readValue())
          "normalizer" -> checkNormalizer(reader.readValue())
          "pre_tokenizer" -> regex = readPreTokenizer(reader.readValue())
          "post_processor" -> checkPostProcessor(reader.readValue())
          "model" -> readModel()
          "truncation",
          "padding" -> require(reader.readValue() == null) { "tokenizer.json sets $name" }
          else -> reader.readValue()
        }
      }
      reader.endObject()
      reader.endDocument()
      require(sawModel) { "tokenizer.json has no model" }
      return Loaded(
        requireNotNull(regex) { "tokenizer.json has no pre-tokenizer" },
        requireNotNull(vocabulary) { "tokenizer.json model has no vocab" },
        mergeLefts,
        mergeRights,
        addedTokens,
      )
    }

    private fun readAddedTokens(value: Any?) {
      for (entry in value as List<*>) {
        val token = entry as Map<*, *>
        val content = token["content"] as String
        for (flag in listOf("normalized", "lstrip", "rstrip", "single_word")) {
          require(token[flag] == false) { "Unsupported added-token flag $flag on $content" }
        }
        addedTokens.add(content to (token["id"] as JsonNumber).toInt())
      }
    }

    private fun checkNormalizer(value: Any?) {
      require(value is Map<*, *> && value["type"] == "NFC" && value.size == 1) {
        "tokenizer.json normalizer is not NFC: $value"
      }
    }

    /** Sequence[Split(Regex, Isolated), ByteLevel(no prefix space, no regex)] -> the regex. */
    private fun readPreTokenizer(value: Any?): String {
      val pre = value as Map<*, *>
      require(pre["type"] == "Sequence") { "Unexpected pre-tokenizer ${pre["type"]}" }
      val steps = pre["pretokenizers"] as List<*>
      require(steps.size == 2) { "Unexpected pre-tokenizer steps: $steps" }
      val split = steps[0] as Map<*, *>
      require(
        split["type"] == "Split" && split["behavior"] == "Isolated" && split["invert"] == false
      ) {
        "Unexpected split step: $split"
      }
      val pattern = split["pattern"] as Map<*, *>
      val byteLevel = steps[1] as Map<*, *>
      // trim_offsets only moves offsets, never IDs.
      require(
        byteLevel["type"] == "ByteLevel" &&
          byteLevel["add_prefix_space"] == false &&
          byteLevel["use_regex"] == false
      ) {
        "Unexpected byte-level step: $byteLevel"
      }
      return requireNotNull(pattern["Regex"] as String?) { "Split pattern is not a regex" }
    }

    private fun checkPostProcessor(value: Any?) {
      require(value == null || (value is Map<*, *> && value["type"] == "ByteLevel")) {
        "Unexpected post-processor: $value"
      }
    }

    private fun readModel() {
      sawModel = true
      reader.beginObject()
      while (reader.hasNext()) {
        when (val name = reader.nextName()) {
          "vocab" -> vocabulary = readVocabulary()
          "merges" -> readMerges()
          "type" -> require(reader.nextString() == "BPE") { "Model is not BPE" }
          "dropout",
          "unk_token" -> require(reader.readValue() == null) { "Unsupported BPE $name" }
          "continuing_subword_prefix",
          "end_of_word_suffix" -> {
            val affix = reader.readValue()
            require(affix == null || affix == "") { "Unsupported BPE $name: $affix" }
          }
          "byte_fallback",
          "ignore_merges" -> require(reader.readValue() == false) { "Unsupported BPE $name" }
          else -> reader.readValue()
        }
      }
      reader.endObject()
    }

    private fun readVocabulary(): HashMap<String, Int> {
      val map = HashMap<String, Int>(VOCABULARY_CAPACITY)
      reader.beginObject()
      while (reader.hasNext()) {
        val token = reader.nextName()
        map[token] = reader.nextInt()
      }
      reader.endObject()
      return map
    }

    /** Merges written as `"left right"` or as `["left", "right"]`. */
    private fun readMerges() {
      reader.beginArray()
      while (reader.hasNext()) {
        if (reader.peek() == KevJsonReader.Kind.ARRAY) {
          reader.beginArray()
          check(reader.hasNext()) { "Empty merge" }
          mergeLefts.add(reader.nextString())
          check(reader.hasNext()) { "Merge without a right side" }
          mergeRights.add(reader.nextString())
          check(!reader.hasNext()) { "Merge is not a pair" }
          reader.endArray()
        } else {
          val merge = reader.nextString()
          val space = merge.indexOf(' ', 1)
          require(space > 0) { "Merge is not a pair: $merge" }
          mergeLefts.add(merge.substring(0, space))
          mergeRights.add(merge.substring(space + 1))
        }
      }
      reader.endArray()
    }
  }

  companion object {
    /**
     * The Split regex of Kev's tokenizer.json: transformers 5.17's `PRETOKENIZE_REGEX` for Qwen2
     * (letters without `\p{M}`, unlike the regex in the Qwen3.5 base repo's tokenizer.json).
     */
    const val TOKENIZER_REGEX =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|" +
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    /** Distinct byte values; GPT-2 also maps unprintable bytes to code points from here up. */
    private const val BYTE_VALUES = 256
    private const val BYTE_MASK = 0xff
    private const val NO_TOKEN = -1

    /** Initial size of the ID list: about one token per three characters of English text. */
    private const val CHARACTERS_PER_TOKEN_ESTIMATE = 3

    /** HashMap capacity for the 248,044-entry vocabulary without a rehash. */
    private const val VOCABULARY_CAPACITY = 1 shl 19

    /** onig's Unicode `\s` (Unicode White_Space). */
    private const val WHITESPACE =
      "\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}" +
        "\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"

    private const val CONTRACTIONS = "(?i:'s|'t|'re|'ve|'m|'ll|'d)"

    /** [CONTRACTIONS] with onig's case folding: only U+017F (long s) folds into these letters. */
    private const val FOLDED_CONTRACTIONS =
      "(?:'[sS\u017f]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"

    /**
     * The java.util.regex / ICU spelling of an onig [regex] from tokenizer.json: the contractions
     * group with case folding spelled out, `\s` and `\S` as explicit White_Space classes.
     */
    fun javaPattern(regex: String): String {
      require(regex.indexOf(CONTRACTIONS) >= 0 && regex.indexOf("(?i") == regex.indexOf(CONTRACTIONS)) {
        "Unsupported case-insensitive group in $regex"
      }
      val folded = regex.replace(CONTRACTIONS, FOLDED_CONTRACTIONS)
      val out = StringBuilder()
      var inClass = false
      var index = 0
      while (index < folded.length) {
        val character = folded[index]
        if (character == '\\' && index + 1 < folded.length) {
          when (val escaped = folded[index + 1]) {
            's' -> out.append(if (inClass) WHITESPACE else "[$WHITESPACE]")
            'S' -> {
              require(!inClass) { "\\S inside a character class is not supported" }
              out.append("[^$WHITESPACE]")
            }
            else -> out.append(character).append(escaped)
          }
          index += 2
          continue
        }
        if (character == '[' && !inClass) {
          inClass = true
        } else if (character == ']' && inClass) {
          inClass = false
        }
        out.append(character)
        index++
      }
      return out.toString()
    }

    private fun normalize(text: String): String = Normalizer.normalize(text, Normalizer.Form.NFC)

    /** GPT-2 `bytes_to_unicode`: every byte maps to one printable character. */
    private fun byteCharacters(): CharArray {
      val printable = (0x21..0x7e) + (0xa1..0xac) + (0xae..0xff)
      val map = CharArray(BYTE_VALUES)
      var extra = 0
      for (byte in 0 until BYTE_VALUES) {
        map[byte] =
          if (byte in printable) {
            byte.toChar()
          } else {
            (BYTE_VALUES + extra++).toChar()
          }
      }
      return map
    }
  }
}

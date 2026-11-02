package com.d1_omni_2b

import java.io.File
import java.util.PriorityQueue
import java.util.regex.Pattern

/** Token IDs of a text and, for each, the characters it covers: `[starts[i], ends[i])` of the text. */
class Encoding(val ids: IntArray, val starts: IntArray, val ends: IntArray) {
  val size: Int
    get() = ids.size
}

/**
 * The model's tokenizer, read from its `tokenizer.json` (Hugging Face `tokenizers`: byte-level BPE,
 * `ignore_merges`, added tokens), called as the model's Python code calls it:
 * `tokenizer(text, add_special_tokens=False)`, with `return_offsets_mapping=True` where it asks for
 * offsets. `encode` follows the `tokenizers` library's order:
 * 1. split the text on the added tokens (all marked `normalized: false`), leftmost-longest;
 * 2. no normalizer;
 * 3. split each remaining piece with the pre-tokenizer regex, keeping matches and the text between
 *    them (`Isolated`);
 * 4. map each piece's UTF-8 bytes to GPT-2's byte characters; a piece that is a vocabulary entry as a
 *    whole is that one token (`ignore_merges`), else the BPE merges apply by rank.
 *
 * Offsets are character offsets into the text (UTF-16 indices; a character split across tokens is
 * covered by each of them, as `tokenizers` maps byte offsets back to characters). The regex is
 * compiled after two rewrites that keep its meaning under java.util.regex and Android's ICU: `\s` /
 * `\S` become Oniguruma's Unicode White_Space class, and the case-insensitive contractions spell out
 * Oniguruma's case folding.
 */
class Tokenizer(tokenizerJson: File) {
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
  private val addedSpecial = LinkedHashMap<String, Boolean>()
  private val idToToken = HashMap<Int, String>()
  private val addedTrie = TrieNode()
  private val vocabulary: HashMap<String, Int>
  private val merges: MergeTable
  private val byteIds = IntArray(BYTE_VALUES)
  private val byteChars = byteCharacters()
  private val charBytes = HashMap<Char, Int>()
  private val splitPattern: Pattern
  private val ignoreMerges: Boolean

  /** The added tokens marked special, as `all_special_tokens` and `added_tokens_decoder` list them. */
  val specialTokens: List<String>
    get() = addedSpecial.filterValues { it }.keys.toList()

  init {
    val loaded = Loader(D1JsonReader(tokenizerJson.readBytes())).load()
    require(loaded.regex == TOKENIZER_REGEX) { "tokenizer.json pre-tokenizer regex differs: ${loaded.regex}" }
    splitPattern = Pattern.compile(javaPattern(loaded.regex))
    vocabulary = loaded.vocabulary
    ignoreMerges = loaded.ignoreMerges
    for (byte in 0 until BYTE_VALUES) {
      byteIds[byte] = requireNotNull(vocabulary[byteChars[byte].toString()]) { "No vocabulary entry for byte $byte" }
      charBytes[byteChars[byte]] = byte
    }
    for ((token, id) in vocabulary) idToToken[id] = token
    merges = MergeTable(loaded.mergeLefts.size)
    for (rank in loaded.mergeLefts.indices) {
      val left = loaded.mergeLefts[rank]
      val right = loaded.mergeRights[rank]
      val leftId = requireNotNull(vocabulary[left]) { "Merge $rank: unknown token $left" }
      val rightId = requireNotNull(vocabulary[right]) { "Merge $rank: unknown token $right" }
      val merged = requireNotNull(vocabulary[left + right]) { "Merge $rank: no token $left$right" }
      merges.put(leftId, rightId, rank, merged)
    }
    for (token in loaded.addedTokens) {
      insert(addedTrie, token.content, token.id)
      addedIds[token.content] = token.id
      addedSpecial[token.content] = token.special
      idToToken[token.id] = token.content
    }
  }

  /** ID of [token] as `convert_tokens_to_ids`: an added token first, else a vocabulary entry; null for neither. */
  fun tokenId(token: String): Int? = addedIds[token] ?: vocabulary[token]

  /** Token IDs of [text], as `tokenizer.encode(text, add_special_tokens=False)`. */
  fun encode(text: String): IntArray = encodeWithOffsets(text).ids

  /** Token IDs and character offsets of [text], as `tokenizer(text, return_offsets_mapping=True)`. */
  fun encodeWithOffsets(text: String): Encoding {
    val ids = ArrayList<Int>(text.length / CHARACTERS_PER_TOKEN_ESTIMATE + 1)
    val starts = ArrayList<Int>(ids.size)
    val ends = ArrayList<Int>(ids.size)
    var position = 0
    var unprocessed = 0
    fun flushText(end: Int) {
      if (end > unprocessed) splitPieces(text, unprocessed, end) { a, b -> bpe(text, a, b, ids, starts, ends) }
    }
    while (position < text.length) {
      var node = addedTrie
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
      flushText(position)
      ids.add(matchedId)
      starts.add(position)
      ends.add(matchedEnd)
      position = matchedEnd
      unprocessed = matchedEnd
    }
    flushText(text.length)
    return Encoding(ids.toIntArray(), starts.toIntArray(), ends.toIntArray())
  }

  /** The text of [ids], as `tokenizer.decode(ids)` (byte-level decode; added tokens as their text). */
  fun decode(ids: IntArray): String {
    val out = StringBuilder()
    val bytes = java.io.ByteArrayOutputStream()
    fun flush() {
      if (bytes.size() > 0) {
        out.append(String(bytes.toByteArray(), Charsets.UTF_8))
        bytes.reset()
      }
    }
    for (id in ids) {
      val token = idToToken[id] ?: continue
      if (addedIds.containsKey(token)) {
        flush()
        out.append(token)
      } else {
        for (character in token) bytes.write(charBytes[character] ?: continue)
      }
    }
    flush()
    return out.toString()
  }

  /** The `Isolated` Split pre-tokenizer over `text[from, to)`: every regex match and every gap between matches. */
  private fun splitPieces(text: String, from: Int, to: Int, onPiece: (Int, Int) -> Unit) {
    val matcher = splitPattern.matcher(text).region(from, to)
    matcher.useTransparentBounds(false)
    matcher.useAnchoringBounds(true)
    var last = from
    while (matcher.find()) {
      if (matcher.start() > last) onPiece(last, matcher.start())
      if (matcher.end() > matcher.start()) onPiece(matcher.start(), matcher.end())
      last = matcher.end()
    }
    if (last < to) onPiece(last, to)
  }

  /** BPE on the UTF-8 bytes of `text[from, to)`, with each token's character offsets. */
  private fun bpe(
    text: String,
    from: Int,
    to: Int,
    output: MutableList<Int>,
    starts: MutableList<Int>,
    ends: MutableList<Int>,
  ) {
    val piece = text.substring(from, to)
    val bytes = piece.toByteArray(Charsets.UTF_8)
    if (bytes.isEmpty()) return
    // The character (UTF-16 index into text) and its end for every byte of the piece.
    val byteStart = IntArray(bytes.size)
    val byteEnd = IntArray(bytes.size)
    run {
      var b = 0
      var c = 0
      while (c < piece.length) {
        val codePoint = piece.codePointAt(c)
        val units = Character.charCount(codePoint)
        val length = String(Character.toChars(codePoint)).toByteArray(Charsets.UTF_8).size
        for (k in 0 until length) {
          byteStart[b + k] = from + c
          byteEnd[b + k] = from + c + units
        }
        b += length
        c += units
      }
    }
    if (ignoreMerges) {
      val whole = StringBuilder(bytes.size)
      for (byte in bytes) whole.append(byteChars[byte.toInt() and BYTE_MASK])
      val id = vocabulary[whole.toString()]
      if (id != null) {
        output.add(id)
        starts.add(byteStart[0])
        ends.add(byteEnd[bytes.size - 1])
        return
      }
    }
    val ids = IntArray(bytes.size) { byteIds[bytes[it].toInt() and BYTE_MASK] }
    val first = IntArray(bytes.size) { it }
    val last = IntArray(bytes.size) { it }
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
      queue.add(Candidate(merges.rank(slot), left, right, ids[left], ids[right], merges.mergedId(slot)))
    }
    for (index in 0 until ids.size - 1) offer(index)
    while (queue.isNotEmpty()) {
      val candidate = queue.remove()
      val left = candidate.left
      val right = candidate.right
      if (!alive[left] || !alive[right] || next[left] != right || ids[left] != candidate.leftId ||
        ids[right] != candidate.rightId
      ) {
        continue
      }
      ids[left] = candidate.mergedId
      last[left] = last[right]
      alive[right] = false
      next[left] = next[right]
      if (next[right] >= 0) previous[next[right]] = left
      offer(previous[left])
      offer(left)
    }
    for (index in ids.indices) {
      if (alive[index]) {
        output.add(ids[index])
        starts.add(byteStart[first[index]])
        ends.add(byteEnd[last[index]])
      }
    }
  }

  private fun insert(trie: TrieNode, content: String, id: Int) {
    require(content.isNotEmpty()) { "Empty added token" }
    var node = trie
    for (character in content) node = node.children.getOrPut(character) { TrieNode() }
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
      while (keys[slot] != EMPTY && keys[slot] != key) slot = (slot + 1) and (capacity - 1)
      keys[slot] = key
      ranks[slot] = rank
      mergedIds[slot] = mergedId
    }

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
      const val HASH_MULTIPLIER = -7046029254386353131L
      const val HASH_SHIFT = 32
    }
  }

  private class AddedToken(val content: String, val id: Int, val special: Boolean)

  private class Loaded(
    val regex: String,
    val vocabulary: HashMap<String, Int>,
    val mergeLefts: List<String>,
    val mergeRights: List<String>,
    val addedTokens: List<AddedToken>,
    val ignoreMerges: Boolean,
  )

  /** What `init` needs from tokenizer.json, streamed so the vocabulary never becomes a tree. */
  private class Loader(private val reader: D1JsonReader) {
    private var regex: String? = null
    private var vocabulary: HashMap<String, Int>? = null
    private val mergeLefts = ArrayList<String>()
    private val mergeRights = ArrayList<String>()
    private val addedTokens = ArrayList<AddedToken>()
    private var ignoreMerges = false
    private var sawModel = false

    fun load(): Loaded {
      reader.beginObject()
      while (reader.hasNext()) {
        when (val name = reader.nextName()) {
          "added_tokens" -> readAddedTokens(reader.readValue())
          "normalizer" -> require(reader.readValue() == null) { "tokenizer.json has a normalizer" }
          "pre_tokenizer" -> regex = readPreTokenizer(reader.readValue())
          "post_processor" -> reader.readValue()
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
        ignoreMerges,
      )
    }

    private fun readAddedTokens(value: Any?) {
      for (entry in value as List<*>) {
        val token = entry as Map<*, *>
        val content = token["content"] as String
        for (flag in listOf("lstrip", "rstrip", "single_word", "normalized")) {
          require(token[flag] == false) { "Unsupported added-token flag $flag on $content" }
        }
        addedTokens.add(AddedToken(content, (token["id"] as JsonNumber).toInt(), token["special"] == true))
      }
    }

    /** Sequence[Split(Regex, Isolated), ByteLevel(no prefix space, no regex)] -> the regex. */
    private fun readPreTokenizer(value: Any?): String {
      val pre = value as Map<*, *>
      require(pre["type"] == "Sequence") { "Unexpected pre-tokenizer ${pre["type"]}" }
      val steps = pre["pretokenizers"] as List<*>
      require(steps.size == 2) { "Unexpected pre-tokenizer steps: $steps" }
      val split = steps[0] as Map<*, *>
      require(split["type"] == "Split" && split["behavior"] == "Isolated" && split["invert"] == false) {
        "Unexpected split step: $split"
      }
      val byteLevel = steps[1] as Map<*, *>
      require(
        byteLevel["type"] == "ByteLevel" && byteLevel["add_prefix_space"] == false && byteLevel["use_regex"] == false
      ) {
        "Unexpected byte-level step: $byteLevel"
      }
      return requireNotNull((split["pattern"] as Map<*, *>)["Regex"] as String?) { "Split pattern is not a regex" }
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
          "ignore_merges" -> ignoreMerges = reader.readValue() == true
          "byte_fallback",
          "fuse_unk" -> require(reader.readValue() == false) { "Unsupported BPE $name" }
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
        if (reader.peek() == D1JsonReader.Kind.ARRAY) {
          reader.beginArray()
          check(reader.hasNext()) { "Merge is not a pair" }
          mergeLefts.add(reader.nextString())
          check(reader.hasNext()) { "Merge is not a pair" }
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
    /** The Split regex of the model's tokenizer.json (digits in runs of one to three). */
    const val TOKENIZER_REGEX =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|" +
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    private const val BYTE_VALUES = 256
    private const val BYTE_MASK = 0xff
    private const val NO_TOKEN = -1
    private const val CHARACTERS_PER_TOKEN_ESTIMATE = 3
    private const val VOCABULARY_CAPACITY = 1 shl 18

    /** Oniguruma's Unicode `\s` (Unicode White_Space). */
    private const val WHITESPACE =
      "\\x{09}-\\x{0d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}" +
        "\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}"

    private const val CONTRACTIONS = "(?i:'s|'t|'re|'ve|'m|'ll|'d)"

    /** [CONTRACTIONS] with Oniguruma's case folding: only U+017F (long s) folds into these letters. */
    private const val FOLDED_CONTRACTIONS = "(?:'[sSſ]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"

    /** The java.util.regex / ICU spelling of the tokenizer.json regex. */
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
        if (character == '[' && !inClass) inClass = true else if (character == ']' && inClass) inClass = false
        out.append(character)
        index++
      }
      return out.toString()
    }

    /** GPT-2 `bytes_to_unicode`: every byte maps to one printable character. */
    private fun byteCharacters(): CharArray {
      val printable = (0x21..0x7e) + (0xa1..0xac) + (0xae..0xff)
      val map = CharArray(BYTE_VALUES)
      var extra = 0
      for (byte in 0 until BYTE_VALUES) {
        map[byte] = if (byte in printable) byte.toChar() else (BYTE_VALUES + extra++).toChar()
      }
      return map
    }
  }
}

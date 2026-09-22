package com.sopro

import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

/** Direct protobuf + compiled NFKC charsmap reader for the shipped UNIGRAM model. No JNI. */
class SpTokenizer(model: File, private val maxLength: Int = 512) {
  private data class Piece(val id: Int, val text: String, val score: Float, val type: Int)

  private class Trie {
    val next = HashMap<Int, Trie>()
    var piece: Piece? = null
  }

  private val trie = Trie()
  private val byteIds = IntArray(256) { -1 }
  private val pieces: List<Piece>
  private val userSymbols: List<ByteArray>
  private val charsMap: IntArray
  private val normalizedBytes: ByteArray
  private val unknownScore: Float
  private val maxScore: Float
  private val bosId: Int
  private val eosId: Int
  private val unkId: Int
  private val dummyPrefix: Boolean
  private val removeSpaces: Boolean
  private val escapeSpaces: Boolean

  init {
    val proto = Proto(model.readBytes())
    val trainer = Proto(proto.bytes(2))
    require(trainer.number(3, 1) == 1) { "This port requires a SentencePiece UNIGRAM model." }
    require(trainer.number(35, 0) == 1) { "Expected byte fallback." }
    require(trainer.number(24, 0) == 0) { "Whitespace suffix mode is unsupported." }
    bosId = trainer.number(41, 1)
    eosId = trainer.number(42, 2)
    unkId = trainer.number(40, 0)
    pieces =
      proto.allBytes(1).mapIndexed { id, bytes ->
        val p = Proto(bytes)
        Piece(id, String(p.bytes(1), Charsets.UTF_8), Float.fromBits(p.fixed32(2)), p.number(3, 1))
      }
    for (piece in pieces) {
      if (piece.type == 6) byteIds[piece.text.substring(3, 5).toInt(16)] = piece.id
      if (piece.type == 1 || piece.type == 4) {
        var node = trie
        for (byte in piece.text.toByteArray(Charsets.UTF_8)) node =
          node.next.getOrPut(byte.toInt() and 255) { Trie() }
        node.piece = piece
      }
    }
    require(byteIds.all { it >= 0 })
    unknownScore = pieces.filter { it.type == 1 }.minOf { it.score } - 10f
    maxScore = pieces.filter { it.type == 1 }.maxOf { it.score }
    userSymbols =
      pieces
        .filter { it.type == 4 }
        .map { it.text.toByteArray(Charsets.UTF_8) }
        .sortedByDescending { it.size }
    val normalizer = Proto(proto.bytes(3))
    dummyPrefix = normalizer.number(3, 1) != 0
    removeSpaces = normalizer.number(4, 1) != 0
    escapeSpaces = normalizer.number(5, 1) != 0
    val map = ByteBuffer.wrap(normalizer.bytes(2)).order(ByteOrder.LITTLE_ENDIAN)
    val trieBytes = map.int
    require(trieBytes >= 1024 && trieBytes % 1024 == 0 && trieBytes < map.remaining())
    charsMap = IntArray(trieBytes / 4) { map.int }
    normalizedBytes = ByteArray(map.remaining()).also { map.get(it) }
  }

  fun encode(text: String, lang: String? = null): IntArray {
    val tag = languageTag(lang)
    val prepared = normalizeText(if (tag.isEmpty()) text else "$tag $text")
    val normalized = normalizeSentencePiece(prepared)
    val size = normalized.size
    val best = FloatArray(size + 1) { Float.NEGATIVE_INFINITY }.also { it[0] = 0f }
    val back = IntArray(size + 1) { -1 }
    val ids = IntArray(size + 1)
    var start = 0
    while (start < size) {
      val width = utf8Width(normalized[start]).coerceAtMost(size - start)
      var node = trie
      var end = start
      var hasSingle = false
      while (end < size) {
        node = node.next[normalized[end].toInt() and 255] ?: break
        end++
        val piece = node.piece ?: continue
        val score = if (piece.type == 4) (end - start) * maxScore - .1f else piece.score
        val candidate = score + best[start]
        if (back[end] == -1 || candidate > best[end]) {
          best[end] = candidate
          back[end] = start
          ids[end] = piece.id
        }
        if (end - start == width) hasSingle = true
      }
      // SentencePiece inserts one UNK per whole code point only when no single-character piece
      // exists.
      if (!hasSingle) {
        end = start + width
        val candidate = unknownScore + best[start]
        if (back[end] == -1 || candidate > best[end]) {
          best[end] = candidate
          back[end] = start
          ids[end] = unkId
        }
      }
      start += width
    }
    val spans = ArrayList<Triple<Int, Int, Int>>()
    var end = size
    while (end > 0) {
      val begin = back[end]
      check(begin >= 0)
      spans += Triple(begin, end, ids[end])
      end = begin
    }
    spans.reverse()
    val out = ArrayList<Int>()
    out += bosId
    for ((begin, limit, id) in spans) {
      if (id == unkId) for (i in begin until limit) out += byteIds[normalized[i].toInt() and 255]
      else out += id
    }
    out += eosId
    return out.take(maxLength).toIntArray()
  }

  private fun normalizeSentencePiece(text: String): ByteArray {
    val input = text.toByteArray(Charsets.UTF_8)
    val decoded = ByteArrayOutputStream()
    var start = 0
    var previousSpace = removeSpaces
    while (start < input.size) {
      val (replacement, consumed) = normalizePrefix(input, start)
      var offset = 0
      if (previousSpace)
        while (offset < replacement.size && replacement[offset] == 32.toByte()) offset++
      if (offset < replacement.size) {
        decoded.write(replacement, offset, replacement.size - offset)
        previousSpace = removeSpaces && replacement.last() == 32.toByte()
      }
      start += consumed
    }
    var normalized = decoded.toByteArray()
    if (removeSpaces) {
      var end = normalized.size
      while (end > 0 && normalized[end - 1] == 32.toByte()) end--
      normalized = normalized.copyOf(end)
    }
    if (normalized.isEmpty()) return normalized
    val out = ByteArrayOutputStream()
    val space = if (escapeSpaces) "▁".toByteArray(Charsets.UTF_8) else byteArrayOf(32)
    if (dummyPrefix) out.write(space)
    for (byte in normalized) if (byte == 32.toByte()) out.write(space) else out.write(byte.toInt())
    return out.toByteArray()
  }

  private fun normalizePrefix(input: ByteArray, start: Int): Pair<ByteArray, Int> {
    for (symbol in userSymbols) if (
      start + symbol.size <= input.size && symbol.indices.all { input[start + it] == symbol[it] }
    )
      return symbol to symbol.size
    fun offset(unit: Int) = (unit ushr 10) shl ((unit and 512) ushr 6)
    var node = offset(charsMap[0])
    var longest = 0
    var value = 0
    for (i in start until input.size) {
      node = node xor (input[i].toInt() and 255)
      if (node !in charsMap.indices) break
      val unit = charsMap[node]
      if ((unit and -2147483393) != (input[i].toInt() and 255)) break
      node = node xor offset(unit)
      if ((unit and 256) != 0) {
        longest = i - start + 1
        value = charsMap[node] and Int.MAX_VALUE
      }
    }
    if (longest > 0) {
      var end = value
      while (end < normalizedBytes.size && normalizedBytes[end] != 0.toByte()) end++
      return normalizedBytes.copyOfRange(value, end) to longest
    }
    val width = utf8Width(input[start]).coerceAtMost(input.size - start)
    return input.copyOfRange(start, start + width) to width
  }

  companion object {
    private fun utf8Width(byte: Byte): Int {
      val b = byte.toInt() and 255
      return when {
        b < 128 -> 1
        b < 224 -> 2
        b < 240 -> 3
        else -> 4
      }
    }

    fun languageTag(lang: String?): String {
      if (lang.isNullOrEmpty()) return ""
      val key = strip(lang).lowercase(Locale.ROOT)
      require(key in setOf("en", "pt", "fr", "de")) {
        "Unsupported language '$lang'; expected en, pt, fr or de."
      }
      return "<|lang_${key}|>"
    }

    private fun whitespace(c: Char): Boolean =
      Character.isWhitespace(c) || Character.isSpaceChar(c) || c == '\u0085'

    private fun strip(s: String) = s.trim { whitespace(it) }

    private fun joinWhitespace(s: String): String {
      val out = StringBuilder()
      var pending = false
      for (c in s) if (whitespace(c)) pending = out.isNotEmpty()
      else {
        if (pending) out.append(' ')
        out.append(c)
        pending = false
      }
      return out.toString()
    }

    /** Mirrors sopro.text.normalize_text, including leading special-tag recursion. */
    fun normalizeText(input: String): String {
      var text = strip(input)
      if (text.isEmpty()) return "You need to add some text for me to talk."
      var end = 0
      while (text.startsWith("<|", end)) {
        val close = text.indexOf("|>", end + 2)
        if (close <= end + 2 || text.substring(end + 2, close).any { it == '|' || whitespace(it) })
          break
        end = close + 2
        while (end < text.length && whitespace(text[end])) end++
      }
      // Python '.' in the source's anchored regex does not consume newlines.
      if (end > 0 && '\n' !in text.substring(end)) {
        val prefix = strip(joinWhitespace(text.substring(0, end)))
        val body = strip(text.substring(end))
        return if (body.isEmpty()) prefix else "$prefix ${normalizeText(body)}"
      }
      val first = text.codePointAt(0)
      if (Character.isLowerCase(first)) {
        val width = Character.charCount(first)
        text = text.substring(0, width).uppercase(Locale.ROOT) + text.substring(width)
      }
      text = joinWhitespace(text).replace("…", "...")
      for (punct in listOf(",", ".", "!", "?", ";", ":")) text = text.replace(" $punct", punct)
      text = text.replace('“', '"').replace('”', '"').replace('‘', '\'').replace('’', '\'')
      text = strip(joinWhitespace(text))
      if (text.last() !in ".!?-,;:") text += "."
      return text
    }
  }

  private class Proto(bytes: ByteArray) {
    private val fields = HashMap<Int, MutableList<Pair<Int, ByteArray>>>()

    init {
      var pos = 0
      fun varint(): Int {
        var result = 0
        var shift = 0
        while (true) {
          val b = bytes[pos++].toInt() and 255
          if (shift < 32) result = result or ((b and 127) shl shift)
          if (b < 128) return result
          shift += 7
          require(shift < 70)
        }
      }
      while (pos < bytes.size) {
        val tag = varint()
        val wire = tag and 7
        val data =
          when (wire) {
            0 -> ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(varint()).array()
            1 -> bytes.copyOfRange(pos, pos + 8).also { pos += 8 }
            2 -> {
              val length = varint()
              bytes.copyOfRange(pos, pos + length).also { pos += length }
            }
            5 -> bytes.copyOfRange(pos, pos + 4).also { pos += 4 }
            else -> error("Unsupported protobuf wire type $wire")
          }
        fields.getOrPut(tag ushr 3) { ArrayList() } += wire to data
      }
    }

    fun bytes(id: Int) = fields[id]?.first()?.second ?: byteArrayOf()

    fun allBytes(id: Int) = fields[id]?.map { it.second } ?: emptyList()

    fun number(id: Int, default: Int): Int =
      fields[id]?.first()?.second?.let { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).int }
        ?: default

    fun fixed32(id: Int) = number(id, 0)
  }
}

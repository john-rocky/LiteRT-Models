package com.d1omni

import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * The model repository's `contract.json` (version 2), the file a host reads everything from: the
 * file of each graph with its bytes and sha256, the decision graphs' signatures and buckets, the
 * token IDs, the temperatures, the settings of each request kind, the position limit and the
 * recommended Android GPU precision.
 */
class D1Contract(json: Any?) {
  /** One entry of `files`: a graph file and its signature, or the bytes and sha256 of a file. */
  class FileEntry(
    val name: String,
    val bytes: Long,
    val sha256: String,
    val graph: String?,
    val bucket: Int?,
    val signature: String?,
  )

  /** The settings of one request kind (`modes.text`, `.image`, `.audio`). */
  class Mode(
    val maxLength: Int,
    val calibrate: Boolean,
    val noulDefault: Map<*, *>?,
    val audio: Boolean,
    /** What a state of null becomes for this kind (`{}` for audio), or [NOT_SET]. */
    val stateNoneBecomes: Any?,
  )

  val version: Int
  val files: List<FileEntry>

  /** The decision graph file of each bucket L, ascending. */
  val decisionFiles: Map<Int, FileEntry>

  /** The decision buckets, ascending (128 … 4096). */
  val buckets: List<Int>

  val tokenizerFile: String
  val tokenizerSha256: String

  /** Token name -> ID (`token_ids`), checked against tokenizer.json by [checkTokenizer]. */
  val tokenIds: Map<String, Int>

  /** Temperature key -> value (`temperatures`), for text requests only. */
  val temperatures: Map<String, Double>

  val modes: Map<String, Mode>

  /** The provider's position limit (16,384). */
  val maxLength: Int

  /** `precision.android_opencl.recommended`, as written. */
  val androidRecommended: String?

  init {
    val root = json as Map<*, *>
    version = (root["contract_version"] as JsonNumber).toInt()
    require(version == SUPPORTED_VERSION) { "contract.json version $version; this app reads 2" }
    files =
      (root["files"] as List<*>).map {
        val entry = it as Map<*, *>
        FileEntry(
          entry["name"] as String,
          (entry["bytes"] as JsonNumber).literal.toLong(),
          entry["sha256"] as String,
          entry["graph"] as String?,
          (entry["bucket"] as JsonNumber?)?.toInt(),
          entry["signature"] as String?,
        )
      }
    val byName = files.associateBy { it.name }
    val decide = (root["graphs"] as Map<*, *>)["decide"] as Map<*, *>
    decisionFiles =
      (decide["by_L"] as Map<*, *>)
        .entries
        .associate { (bucket, name) ->
          val entry = requireNotNull(byName[name as String]) { "contract.json lists no file $name" }
          val length = (bucket as String).toInt()
          require(entry.bucket == length && entry.signature == "decide_$length") {
            "contract.json: $name is not decide_$length"
          }
          length to entry
        }
        .toSortedMap()
    buckets = decisionFiles.keys.toList()
    val tokenizer = root["tokenizer"] as Map<*, *>
    tokenizerFile = tokenizer["file"] as String
    tokenizerSha256 = tokenizer["sha256"] as String
    tokenIds =
      (root["token_ids"] as Map<*, *>).entries.associate { (token, id) ->
        token as String to (id as JsonNumber).toInt()
      }
    temperatures =
      (root["temperatures"] as Map<*, *>).entries.associate { (key, value) ->
        key as String to (value as JsonNumber).toDouble()
      }
    modes =
      (root["modes"] as Map<*, *>)
        .entries
        .filter { (_, value) -> value is Map<*, *> }
        .associate { (kind, value) ->
          val mode = value as Map<*, *>
          kind as String to
            Mode(
              (mode["max_len"] as JsonNumber).toInt(),
              mode["calibrate"] as Boolean,
              mode["noul_default"] as Map<*, *>?,
              mode["audio"] as Boolean,
              if (mode.containsKey("state_none_becomes")) mode["state_none_becomes"] else NOT_SET,
            )
        }
    maxLength = (root["max_length"] as JsonNumber).toInt()
    androidRecommended =
      ((root["precision"] as Map<*, *>?)?.get("android_opencl") as Map<*, *>?)?.get("recommended")
        as String?
  }

  /** The [FileEntry] of [name], or null. */
  fun file(name: String): FileEntry? = files.firstOrNull { it.name == name }

  /** The temperature of [question] for a text request (`d1_host.temperature`). */
  fun temperature(question: D1Question): Double =
    temperatures[D1Prompt.temperatureKey(question)]
      ?: temperatures[question.type.wireName]
      ?: 1.0

  /**
   * Checks [tokenizer] against `token_ids` (every role token has the contract's ID and BOS is
   * `<|startoftext|>`); throws on a difference.
   */
  fun checkTokenizer(tokenizer: D1Tokenizer) {
    for ((token, id) in tokenIds) {
      check(tokenizer.tokenId(token) == id) {
        "tokenizer.json gives $token ${tokenizer.tokenId(token)}, contract.json $id"
      }
    }
    check(tokenIds[D1Prompt.BOS_TOKEN] == tokenizer.tokenId(D1Prompt.BOS_TOKEN)) {
      "the tokenizer's BOS is not ${D1Prompt.BOS_TOKEN}"
    }
  }

  companion object {
    const val FILE = "contract.json"
    private const val SUPPORTED_VERSION = 2

    /** A [Mode.stateNoneBecomes] that the contract does not set (a null state stays null). */
    val NOT_SET = Any()

    fun read(file: File): D1Contract = D1Contract(D1Json.parse(file.readBytes()))

    /** The smallest of [buckets] (ascending) that holds [positions], or null when none does. */
    fun bucketFor(positions: Int, buckets: List<Int>): Int? = buckets.firstOrNull { positions <= it }

    /** sha256 of [file] as lower-case hex. */
    fun sha256(file: File): String {
      val digest = MessageDigest.getInstance("SHA-256")
      FileInputStream(file).use { stream ->
        val buffer = ByteArray(BUFFER_BYTES)
        while (true) {
          val read = stream.read(buffer)
          if (read < 0) break
          digest.update(buffer, 0, read)
        }
      }
      return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** sha256 of [text]'s UTF-8 bytes as lower-case hex. */
    fun sha256(text: String): String =
      MessageDigest.getInstance("SHA-256").digest(text.toByteArray(Charsets.UTF_8)).joinToString(
        ""
      ) {
        "%02x".format(it)
      }

    private const val BUFFER_BYTES = 1 shl 20
  }
}

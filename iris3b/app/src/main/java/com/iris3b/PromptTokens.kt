package com.iris3b

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.ShortBuffer
import java.nio.channels.FileChannel

/**
 * The iris-3b prompt layout and the host-side embedding lookup.
 *
 * Layout (upstream `iris3b/text/qwen3_vl.py`, `_run`): the system-prompt prefix (34 ids), the
 * caption cut to 295 ids, the assistant-turn suffix (5 ids), right-padded with `<|endoftext|>` to
 * 334; the attention mask is 1 on prefix + caption + suffix. The suffix is placed after the cut, so
 * a long caption never loses the assistant marker.
 *
 * The embedding table is the checkpoint's bf16 `embed_tokens` as raw little-endian bytes,
 * memory-mapped; bf16 -> fp32 is a 16-bit shift, so the lookup is exact.
 */
class PromptTokens(assetsDir: File) : Closeable {

    private val tokenizer = QwenTokenizer(assetsDir)
    private val prefix = readIds(File(assetsDir, "prompt_prefix_ids.txt"))
    private val suffix = readIds(File(assetsDir, "prompt_suffix_ids.txt"))
    private val embeddingFile = RandomAccessFile(File(assetsDir, EMBEDDING_FILE), "r")
    private val embeddings: ShortBuffer = embeddingFile.channel
        .map(FileChannel.MapMode.READ_ONLY, 0, embeddingFile.length())
        .order(ByteOrder.LITTLE_ENDIAN)
        .asShortBuffer()

    /** Token ids and attention mask, both [SEQ_LEN], for one caption. */
    fun encode(caption: String): Tokens {
        val captionIds = tokenizer.encode(caption)
        val budget = WINDOW - suffix.size
        val n = minOf(captionIds.size, budget)
        val ids = IntArray(SEQ_LEN) { PAD_TOKEN_ID }
        prefix.copyInto(ids, 0)
        for (i in 0 until n) {
            ids[prefix.size + i] = captionIds[i]
        }
        suffix.copyInto(ids, prefix.size + n)
        val realCount = prefix.size + n + suffix.size
        return Tokens(ids, realCount, captionIds.size)
    }

    /** inputs_embeds [1, SEQ_LEN, HIDDEN] as fp32, gathered from the bf16 table. */
    fun embed(tokens: Tokens): FloatArray {
        val out = FloatArray(SEQ_LEN * HIDDEN)
        for (position in 0 until SEQ_LEN) {
            val base = tokens.ids[position] * HIDDEN
            for (channel in 0 until HIDDEN) {
                out[position * HIDDEN + channel] = Float.fromBits(embeddings.get(base + channel).toInt() shl 16)
            }
        }
        return out
    }

    override fun close() {
        embeddingFile.close()
    }

    /** Padded ids, how many are real (attention mask = 1), and the untruncated caption length. */
    class Tokens(val ids: IntArray, val realCount: Int, val captionTokens: Int) {
        fun mask(): IntArray = IntArray(SEQ_LEN) { if (it < realCount) 1 else 0 }
    }

    companion object {
        const val SEQ_LEN = 334
        const val WINDOW = 300
        const val HIDDEN = 2560
        const val PAD_TOKEN_ID = 151643
        const val EMBEDDING_FILE = "embed_bf16.bin"

        private fun readIds(file: File): IntArray =
            file.readText().trim().split(',').map { it.trim().toInt() }.toIntArray()
    }
}

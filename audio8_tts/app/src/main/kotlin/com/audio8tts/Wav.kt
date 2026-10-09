package com.audio8tts

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Minimal RIFF/WAVE helpers (from the Fun-ASR demo): walk the chunks, read 16-bit PCM, write 16-bit mono. */
object Wav {
    class Pcm(val data: ByteArray, val sampleRate: Int, val channels: Int)

    private class Info(val sampleRate: Int, val channels: Int, val bits: Int, val byteRate: Int, val dataSize: Long,
                       val data: ByteArray?)

    private fun read(file: File, withData: Boolean): Info {
        RandomAccessFile(file, "r").use { f ->
            val head = ByteArray(12).also { f.readFully(it) }
            require(String(head, 0, 4) == "RIFF" && String(head, 8, 4) == "WAVE") { "not a wav: $file" }
            var fmt: ByteBuffer? = null
            while (f.filePointer + 8 <= f.length()) {
                val hdr = ByteArray(8).also { f.readFully(it) }
                val id = String(hdr, 0, 4)
                val size = ByteBuffer.wrap(hdr, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xffffffffL
                when (id) {
                    "fmt " -> {
                        fmt = ByteBuffer.wrap(ByteArray(size.toInt()).also { f.readFully(it) }).order(ByteOrder.LITTLE_ENDIAN)
                        if (size % 2 == 1L) f.skipBytes(1)
                    }
                    "data" -> {
                        val m = requireNotNull(fmt) { "data before fmt in $file" }
                        val data = if (withData) ByteArray(size.toInt()).also { f.readFully(it) } else null
                        return Info(sampleRate = m.getInt(4), channels = m.getShort(2).toInt(), bits = m.getShort(14).toInt(),
                            byteRate = m.getInt(8), dataSize = size, data = data)
                    }
                    else -> f.seek(f.filePointer + size + (size % 2))
                }
            }
        }
        error("no data chunk in $file")
    }

    fun seconds(file: File): Double = read(file, withData = false).let { it.dataSize.toDouble() / it.byteRate }

    /** The samples of a 16-bit PCM wav, for playback through AudioTrack. */
    fun pcm16(file: File): Pcm = read(file, withData = true).let {
        require(it.bits == 16) { "not 16-bit PCM: $file" }
        Pcm(it.data!!, it.sampleRate, it.channels)
    }

    /** A 16-bit PCM wav as mono float in [-1, 1) (channels averaged), as soundfile float32 + mean(1) reads it. */
    fun monoFloat(file: File): Pair<FloatArray, Int> {
        val p = pcm16(file)
        val s = ByteBuffer.wrap(p.data).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val n = s.limit() / p.channels
        val out = FloatArray(n)
        for (i in 0 until n) {
            var acc = 0f
            for (c in 0 until p.channels) acc += s.get(i * p.channels + c) / 32768f
            out[i] = acc / p.channels
        }
        return out to p.sampleRate
    }

    /** Float [-1, 1] -> 16-bit PCM bytes (soundfile's PCM_16 write: scale 32767, clip). */
    fun floatToPcm16(x: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(x.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in x) {
            val s = Math.round(v.coerceIn(-1f, 1f) * 32767f)
            bb.putShort(s.toShort())
        }
        return bb.array()
    }

    fun writeMono16(file: File, pcm: ByteArray, sampleRate: Int) {
        file.outputStream().use { it.write(mono16Header(pcm.size, sampleRate)); it.write(pcm) }
    }

    /** A complete 16-bit mono RIFF/WAVE file in memory: the 44-byte header followed by [pcm]. */
    fun mono16Bytes(pcm: ByteArray, sampleRate: Int): ByteArray = mono16Header(pcm.size, sampleRate) + pcm

    private fun mono16Header(dataBytes: Int, sampleRate: Int): ByteArray =
        ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + dataBytes); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(1)
            putInt(sampleRate); putInt(sampleRate * 2); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(dataBytes)
        }.array()
}

package com.mimi

import android.app.Activity
import android.graphics.Color
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

/**
 * Mimi neural codec demo: round-trips a 2-second 24 kHz clip through the on-device hybrid codec
 * (SEANet convs on GPU, transformers + RVQ on CPU) and plays original vs. reconstructed for A/B.
 */
class MainActivity : Activity() {

    private val tag = "MIMI"
    private val bg = Executors.newSingleThreadExecutor()
    private var codec: MimiCodec? = null
    private lateinit var original: FloatArray
    private var recon: FloatArray? = null

    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(56, 120, 56, 56)
        }
        status = TextView(this).apply { textSize = 15f; text = "Loading Mimi codec (GPU convs + CPU transformers)…" }
        val playOrig = Button(this).apply {
            text = "▶ Play original"; isEnabled = false; setOnClickListener { play(original) }
        }
        val playRecon = Button(this).apply {
            text = "▶ Play reconstructed (codec)"; isEnabled = false
            setOnClickListener { recon?.let { play(it) } }
        }
        root.addView(status); root.addView(playOrig); root.addView(playRecon)
        setContentView(root)

        original = readFloats(assets.open("test_audio.bin").readBytes())

        bg.execute {
            try {
                val c = MimiCodec(this); codec = c
                c.roundTrip(original)            // warm up GPU + JIT the RVQ
                val r = c.roundTrip(original)    // measured
                recon = r.audio
                runCatching {                                   // dump for desktop A/B verification
                    java.io.File(filesDir, "recon.bin").outputStream().use { os ->
                        val bb = ByteBuffer.allocate(r.audio.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                        r.audio.forEach { bb.putFloat(it) }; os.write(bb.array())
                    }
                    java.io.File(filesDir, "codes.bin").outputStream().use { os ->
                        val bb = ByteBuffer.allocate(r.codes.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                        r.codes.forEach { bb.putInt(it) }; os.write(bb.array())
                    }
                }
                val rtf = (r.encodeMs + r.decodeMs) / 2000.0   // clip is 2 s
                Log.i(tag, "encode=${r.encodeMs}ms decode=${r.decodeMs}ms codes=${r.codes.size} rtf=$rtf")
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "On-device hybrid codec ✓\n\n" +
                        "encode ${r.encodeMs} ms · decode ${r.decodeMs} ms · RTF ${"%.2f".format(rtf)}\n" +
                        "${MimiRvq.NQ} codebooks × ${MimiCodec.TC} frames = ${r.codes.size} ints · 24 kHz\n\n" +
                        "Playing reconstructed… tap to A/B."
                    playOrig.isEnabled = true; playRecon.isEnabled = true
                    play(r.audio)
                }
            } catch (e: Throwable) {
                Log.e(tag, "codec failed", e)
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
                    status.text = "FAIL: ${e.message}"
                }
            }
        }
    }

    private fun readFloats(b: ByteArray): FloatArray {
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 4) { bb.float }
    }

    private fun play(audio: FloatArray) {
        bg.execute {
            try {
                val track = AudioTrack(
                    AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
                    AudioFormat.Builder()
                        .setSampleRate(24000)
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build(),
                    audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
                )
                track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
                track.play()
                Thread.sleep((audio.size / 24L) + 250)
                track.release()
            } catch (e: Throwable) {
                Log.e(tag, "play failed: ${e.message}")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        codec?.close()
    }
}

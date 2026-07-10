package com.vibevoice

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * Minimal text-to-speech UI: type a sentence, tap Generate, hear it in the bundled voice.
 * Model load and generation run on a background thread; audio plays via AudioTrack (float PCM).
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var synth: VibeVoiceSynthesizer? = null

    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var button: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true                       // inset content below the status bar
            setPadding(48, 48, 48, 48)
        }
        // The ActionBar already shows the app name, so the content starts with the text field.
        input = EditText(this).apply {
            hint = "Enter text to speak"
            setText("Hello, this is a test of on device text to speech.")
            minLines = 2
        }
        button = Button(this).apply { text = "Generate"; isEnabled = false }
        status = TextView(this).apply { text = "Loading model…"; textSize = 14f }
        val topMargins = intArrayOf(0, 32, 24)
        for ((index, view) in listOf(input, button, status).withIndex()) {
            val params = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            params.topMargin = topMargins[index]
            root.addView(view, params)
        }
        setContentView(root)

        bg.execute {
            val s = try {
                VibeVoiceSynthesizer(this)
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Load failed: ${e.message}" }
                return@execute
            }
            synth = s
            runOnUiThread { status.text = "Ready."; button.isEnabled = true }
        }

        button.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            button.isEnabled = false
            status.text = "Generating…"
            bg.execute {
                val s = synth ?: return@execute
                try {
                    val r = s.synthesize(text)
                    saveWav(r.audio)
                    runOnUiThread {
                        val secs = r.audio.size.toFloat() / VibeVoiceSynthesizer.SAMPLE_RATE
                        status.text =
                            "Spoke %.1fs (%d tokens) in %d ms"
                                .format(secs, r.speechTokens, r.ms)
                        button.isEnabled = true
                    }
                    play(r.audio)
                } catch (e: Throwable) {
                    runOnUiThread { status.text = "Error: ${e.message}"; button.isEnabled = true }
                }
            }
        }
    }

    /** Save the last output as a 24 kHz mono 16-bit WAV in filesDir (for off-device inspection). */
    private fun saveWav(audio: FloatArray) {
        val sr = VibeVoiceSynthesizer.SAMPLE_RATE
        val data = audio.size * 2
        val bb = java.nio.ByteBuffer.allocate(44 + data).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + data); bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16); bb.putShort(1); bb.putShort(1)
        bb.putInt(sr); bb.putInt(sr * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(data)
        for (v in audio) bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        java.io.File(filesDir, "output.wav").writeBytes(bb.array())
    }

    private fun play(audio: FloatArray) {
        val track = AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
            AudioFormat.Builder()
                .setSampleRate(VibeVoiceSynthesizer.SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
            audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        track.play()
        Thread.sleep((audio.size * 1000L / VibeVoiceSynthesizer.SAMPLE_RATE) + 250)
        track.release()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        synth?.close()
    }
}

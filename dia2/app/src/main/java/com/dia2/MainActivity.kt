package com.dia2

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
import java.io.File
import java.util.concurrent.Executors

/** Dia2-1B dialogue TTS: type a [S1]/[S2] script, generate on-device, play it back. */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var synth: Dia2Synthesizer? = null
    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var button: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        input = EditText(this).apply {
            setText("[S1] Hello, how are you today? [S2] I'm great, thanks for asking.")
            minLines = 2
        }
        button = Button(this).apply { text = "Generate"; isEnabled = false }
        status = TextView(this).apply { text = "Loading Dia2-1B…"; textSize = 14f }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            fitsSystemWindows = true
            setPadding(48, 48, 48, 48)
        }
        val margins = intArrayOf(0, 32, 24)
        for ((i, v) in listOf(input, button, status).withIndex()) {
            val lp = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            lp.topMargin = margins[i]
            root.addView(v, lp)
        }
        setContentView(root)

        bg.execute {
            synth = try {
                Dia2Synthesizer(this)
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Load failed: ${e.message}" }
                return@execute
            }
            runOnUiThread { status.text = "Ready."; button.isEnabled = true }
        }

        button.setOnClickListener {
            val text = input.text.toString().ifBlank { return@setOnClickListener }
            button.isEnabled = false
            status.text = "Generating… (slow — Moshi on CPU)"
            bg.execute {
                val s = synth ?: return@execute
                try {
                    val r = s.synthesize(text)
                    saveWav(r.audio)
                    val secs = r.audio.size.toFloat() / Dia2Synthesizer.SAMPLE_RATE
                    runOnUiThread {
                        status.text =
                            "Spoke %.1fs (%d frames) in %d ms".format(secs, r.frames, r.ms)
                        button.isEnabled = true
                    }
                    play(r.audio)
                } catch (e: Throwable) {
                    runOnUiThread { status.text = "Error: ${e.message}"; button.isEnabled = true }
                }
            }
        }
    }

    private fun saveWav(audio: FloatArray) {
        val sr = Dia2Synthesizer.SAMPLE_RATE
        val data = audio.size * 2
        val bb = java.nio.ByteBuffer.allocate(44 + data).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + data); bb.put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()); bb.putInt(16); bb.putShort(1); bb.putShort(1)
        bb.putInt(sr); bb.putInt(sr * 2); bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(data)
        for (v in audio) bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        File(filesDir, "output.wav").writeBytes(bb.array())
    }

    private fun play(audio: FloatArray) {
        if (audio.isEmpty()) return
        val track = AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
            AudioFormat.Builder()
                .setSampleRate(Dia2Synthesizer.SAMPLE_RATE)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
            audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        track.play()
        Thread.sleep((audio.size * 1000L / Dia2Synthesizer.SAMPLE_RATE) + 250)
        track.release()
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdownNow()
        synth?.close()
    }
}

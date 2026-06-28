package com.asr

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * Fully-GPU on-device speech recognition (wav2vec2-CTC). Hold the button and talk, or transcribe the bundled
 * sample. The model runs as one CompiledModel GPU graph; CTC greedy decode is on the host.
 */
class MainActivity : Activity() {

    private val tag = "ASR"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: Wav2Vec2CTC? = null

    private lateinit var status: TextView
    private lateinit var result: TextView
    @Volatile private var recording = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 90, 36, 36) }
        status = TextView(this).apply { textSize = 15f; text = "Loading wav2vec2-CTC on GPU…" }
        result = TextView(this).apply {
            textSize = 22f; setPadding(0, 48, 0, 48); gravity = Gravity.START
            setTextColor(Color.rgb(0x11, 0x11, 0x11)); text = ""
        }
        val talk = Button(this).apply {
            text = "🎤  Hold to Talk"; textSize = 18f; isEnabled = false
            setOnTouchListener { _, e ->
                when (e.action) {
                    MotionEvent.ACTION_DOWN -> { startRecording(); true }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { stopRecording(); true }
                    else -> false
                }
            }
        }
        val sample = Button(this).apply {
            text = "▶  Transcribe sample clip"; isEnabled = false
            setOnClickListener { transcribeSample() }
        }
        root.addView(status); root.addView(talk); root.addView(sample); root.addView(result)
        setContentView(root)

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)

        bg.execute {
            try {
                net = Wav2Vec2CTC(this)
                runOnUiThread { talk.isEnabled = true; sample.isEnabled = true; status.text = "Ready — hold to talk, or try the sample." }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    private var recordThread: Thread? = null
    private val pcmOut = ByteArrayOutputStream()

    private fun startRecording() {
        if (recording || net == null) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            status.text = "Microphone permission needed."; return
        }
        recording = true; pcmOut.reset()
        status.text = "● Listening…"; result.text = ""
        recordThread = Thread {
            val min = AudioRecord.getMinBufferSize(Wav2Vec2CTC.SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val rec = AudioRecord(MediaRecorder.AudioSource.MIC, Wav2Vec2CTC.SR,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(min, Wav2Vec2CTC.SR * 2))
            val buf = ShortArray(1600)
            try {
                rec.startRecording()
                var total = 0
                while (recording && total < Wav2Vec2CTC.SAMPLES) {
                    val r = rec.read(buf, 0, buf.size)
                    for (i in 0 until r) { pcmOut.write(buf[i].toInt() and 0xFF); pcmOut.write((buf[i].toInt() shr 8) and 0xFF) }
                    total += r
                }
            } catch (e: Throwable) { Log.e(tag, "record", e) } finally { rec.stop(); rec.release() }
        }.also { it.start() }
    }

    private fun stopRecording() {
        if (!recording) return
        recording = false
        bg.execute {
            recordThread?.join()
            val pcm = pcmBytesToFloat(pcmOut.toByteArray())
            if (pcm.size < Wav2Vec2CTC.SR / 2) { runOnUiThread { status.text = "Too short — hold longer." }; return@execute }
            runInfer(pcm, "mic")
        }
    }

    private fun transcribeSample() {
        runOnUiThread { status.text = "Transcribing sample…"; result.text = "" }
        bg.execute {
            try {
                val bytes = assets.open("sample_speech.wav").readBytes()
                val pcm = pcmBytesToFloat(bytes.copyOfRange(44, bytes.size))   // skip WAV header
                runInfer(pcm, "sample")
            } catch (e: Throwable) { Log.e(tag, "sample", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun runInfer(pcm: FloatArray, src: String) {
        val n = net ?: return
        val t0 = System.nanoTime()
        val text = n.transcribe(pcm)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "transcribe($src) ${ms}ms: $text")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU ASR ✓  ${ms} ms  ·  wav2vec2-CTC, CompiledModel GPU"
            result.text = if (text.isBlank()) "(no speech detected)" else text
        }
    }

    /** little-endian PCM16 bytes -> mono float [-1,1]. */
    private fun pcmBytesToFloat(b: ByteArray): FloatArray {
        val n = b.size / 2
        val out = FloatArray(n)
        for (i in 0 until n) {
            val s = (b[i * 2].toInt() and 0xFF) or (b[i * 2 + 1].toInt() shl 8)
            out[i] = s / 32768f
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); recording = false; bg.shutdown(); net?.close() }
}

package com.emalightning

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTimestamp
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

/**
 * EMA Lightning (Turkish TTS) on LiteRT: type Turkish, tap Speak, hear it. The four graphs come from
 * litert-community/EMA-Lightning-LiteRT (downloaded once into filesDir, 27 MB); the acoustic graphs run on the
 * CPU, the decoder on the GPU. The audio plays under a "● PLAYING" pill and is saved as files/last.wav.
 *
 * Recording harness (the same code path a finger drives): `adb shell am start -n com.emalightning/.MainActivity
 * --es text "<sentence>" --ez autoplay true` synthesizes and plays the sentence after READY; every run writes
 * files/run.json (the sentence, letters / frames / buckets, unrounded ms per stage, the audio length, the PLAYING
 * time) and logs EMA_READY / EMA_PLAYING / EMA_DONE.
 */
class MainActivity : Activity() {

    private val tag = "EmaLightning"
    private val bg = Executors.newSingleThreadExecutor()
    private var tts: EmaSynthesizer? = null
    private var frontend: EmaFrontend? = null
    private var busy = false

    private lateinit var status: TextView
    private lateinit var pill: TextView
    private lateinit var input: EditText
    private lateinit var speak: Button
    private lateinit var download: Button

    companion object {
        const val EXAMPLE = "Merhaba, bu ses telefonunuzun üzerinde üretiliyor."
        private const val C_IDLE = 0xFF9E9E9E.toInt()
        private const val C_WORK = 0xFF1E88E5.toInt()
        private const val C_LIVE = 0xFFE53935.toInt()   // PLAYING (the recording harness looks for this colour)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        keepVisible()
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(56, 120, 56, 56); setBackgroundColor(Color.WHITE) }
        status = TextView(this).apply { textSize = 18f; text = "Loading…"; setTextColor(Color.DKGRAY); setPadding(0, 24, 0, 24) }
        pill = TextView(this).apply {
            textSize = 18f; gravity = Gravity.CENTER; setTextColor(Color.WHITE); setPadding(40, 20, 40, 20)
            background = GradientDrawable().apply { cornerRadius = 48f; setColor(C_IDLE) }
            text = "● IDLE"
        }
        input = EditText(this).apply { setText(EXAMPLE); textSize = 20f; minLines = 3; setTextColor(Color.BLACK) }
        speak = Button(this).apply { text = "▶ Speak"; textSize = 20f; isEnabled = false; setOnClickListener { say(input.text.toString(), fromIntent = false) } }
        download = Button(this).apply { text = "Download the model (27 MB)"; isEnabled = false; setOnClickListener { fetchModels() } }
        root.addView(status); root.addView(input); root.addView(speak); root.addView(pill); root.addView(download)
        setContentView(root)
        if (EmaSynthesizer.missing(this).isEmpty()) { download.visibility = android.view.View.GONE; loadEngine() } else {
            status.text = "The four graphs are not on the phone yet. Download them from litert-community/EMA-Lightning-LiteRT (27 MB, once)."
            download.isEnabled = true
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (tts != null && !busy) handleIntent(intent)
    }

    private fun handleIntent(intent: Intent) {
        val text = intent.getStringExtra("text")
        if (text != null) runOnUiThread { input.setText(text) }
        if (intent.getBooleanExtra("autoplay", false)) say(text ?: input.text.toString(), fromIntent = true)
    }

    /** Debug / recording only: show over the keyguard and keep the screen on, so an adb-driven take runs on a locked phone. */
    private fun keepVisible() {
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun fetchModels() {
        download.isEnabled = false
        bg.execute {
            try {
                for (f in EmaSynthesizer.missing(this)) {
                    runOnUiThread { status.text = "Downloading $f …" }
                    val tmp = File(filesDir, "$f.part")
                    (URL(EmaSynthesizer.HUB + f).openConnection() as HttpURLConnection).apply { instanceFollowRedirects = true; connectTimeout = 15000; readTimeout = 60000 }
                        .inputStream.use { ins -> tmp.outputStream().use { ins.copyTo(it) } }
                    check(tmp.renameTo(File(filesDir, f)))
                }
                runOnUiThread { download.visibility = android.view.View.GONE }
                loadEngine()
            } catch (e: Throwable) {
                Log.e(tag, "download failed", e)
                runOnUiThread { status.text = "Download failed: ${e.message}"; download.isEnabled = true }
            }
        }
    }

    private fun loadEngine() {
        speak.isEnabled = false
        bg.execute {
            try {
                val t0 = System.nanoTime()
                val t = EmaSynthesizer(this)
                val fe = EmaFrontend(t.vocab)
                t.synthesize(fe.piece(fe.alphabet(EXAMPLE)), seed = 0)  // warm-up on the example's buckets (L64 / T96, P48 + P120): CPU graphs + the GPU compile
                tts = t; frontend = fe
                val ms = (System.nanoTime() - t0) / 1e6
                Log.i(tag, "EMA_READY load_ms=$ms")
                runOnUiThread {
                    status.text = "Ready · acoustic graphs on the CPU, decoder on the GPU (fp16) · LiteRT 2.2.0\nType Turkish and tap Speak."
                    speak.isEnabled = true
                    handleIntent(intent)
                }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread { status.text = "FAIL: ${e.message}"; download.isEnabled = EmaSynthesizer.missing(this).isNotEmpty() }
            }
        }
    }

    private fun say(raw: String, fromIntent: Boolean) {
        val t = tts ?: return
        val fe = frontend ?: return
        if (busy) return
        busy = true
        speak.isEnabled = false
        setPill("● SYNTHESIZING", C_WORK)
        status.text = "Synthesizing…"
        bg.execute {
            try {
                val text = fe.alphabet(raw)
                val pieces = fe.chunk(text, 1.0)
                val results = ArrayList<EmaSynthesizer.Result>()
                val audio = ArrayList<FloatArray>()
                val wall0 = System.currentTimeMillis()
                for ((i, pp) in pieces.withIndex()) {
                    val r = t.synthesize(fe.piece(pp.first), seed = i.toLong())
                    results.add(r); audio.add(r.audio)
                    if (pp.second > 0) audio.add(FloatArray((pp.second * EmaSynthesizer.SAMPLE_RATE).toInt()))
                }
                val wav = FloatArray(audio.sumOf { it.size })
                var o = 0
                for (a in audio) { System.arraycopy(a, 0, wav, o, a.size); o += a.size }
                val secs = wav.size.toDouble() / EmaSynthesizer.SAMPLE_RATE
                val total = results.sumOf { it.timing.totalMs }
                val first = results.firstOrNull()?.timing?.firstAudioMs ?: 0.0
                val shown = "%.2f s of speech in %.0f ms · first audio %.0f ms · RTF %.3f".format(secs, total, first, total / 1000.0 / secs)
                Log.i(tag, "EMA_SYNTH pieces=${pieces.size} total_ms=$total first_ms=$first secs=$secs")
                saveWav(File(filesDir, "last.wav"), wav)
                runOnUiThread { status.text = shown }
                val playNs = play(wav)
                val run = JSONObject()
                    .put("text", raw).put("alphabet", text).put("pieces", JSONArray(pieces.map { it.first }))
                    .put("letters", JSONArray(results.map { it.letters })).put("frames", JSONArray(results.map { it.frames }))
                    .put("buckets", JSONArray(results.map { "L${it.lb}_T${it.tb}" }))
                    .put("text_ms", JSONArray(results.map { it.timing.textMs })).put("align_ms", JSONArray(results.map { it.timing.alignMs }))
                    .put("steps_ms", JSONArray(results.map { it.timing.stepsMs })).put("decode_ms", JSONArray(results.map { JSONArray(it.timing.decodeMs) }))
                    .put("first_audio_ms", first).put("total_ms", total).put("audio_samples", wav.size).put("audio_seconds", secs)
                    .put("shown", shown).put("from_intent", fromIntent).put("wall_start_epoch_ms", wall0)
                    .put("playing_ns_after_start", playNs).put("litert", "2.2.0").put("device", android.os.Build.MODEL)
                File(filesDir, "run.json").writeText(run.toString(1))
                Log.i(tag, "EMA_DONE")
                runOnUiThread { speak.isEnabled = true; busy = false }
            } catch (e: Throwable) {
                Log.e(tag, "synth failed", e)
                runOnUiThread { status.text = "FAIL: ${e.message}"; setPill("● IDLE", C_IDLE); speak.isEnabled = true; busy = false }
            }
        }
    }

    /** Plays the wav; the pill turns PLAYING at the playback head's first movement (the moment sound leaves the speaker). */
    private fun play(audio: FloatArray): Long {
        val rate = EmaSynthesizer.SAMPLE_RATE
        val track = AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build(),
            AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_FLOAT).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build(),
            audio.size * 4, AudioTrack.MODE_STATIC, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
        track.write(audio, 0, audio.size, AudioTrack.WRITE_BLOCKING)
        val t0 = System.nanoTime()
        track.play()
        var playingNs = -1L
        val ts = AudioTimestamp()
        while (true) {
            if (playingNs < 0 && (track.playbackHeadPosition > 0 || (track.getTimestamp(ts) && ts.framePosition > 0))) {
                playingNs = System.nanoTime() - t0
                runOnUiThread { setPill("● PLAYING", C_LIVE) }
                Log.i(tag, "EMA_PLAYING ns=$playingNs")
            }
            if (track.playbackHeadPosition >= audio.size || System.nanoTime() - t0 > (audio.size.toLong() * 1_000_000_000L / rate) + 1_500_000_000L) break
            Thread.sleep(5)
        }
        track.release()
        runOnUiThread { setPill("● IDLE", C_IDLE) }
        return playingNs
    }

    private fun setPill(text: String, color: Int) {
        pill.text = text
        (pill.background as GradientDrawable).setColor(color)
    }

    private fun saveWav(f: File, audio: FloatArray) {
        val rate = EmaSynthesizer.SAMPLE_RATE
        val pcm = ByteArray(audio.size * 2)
        for (i in audio.indices) {
            val v = (audio[i].coerceIn(-1f, 1f) * 32767f).toInt()
            pcm[2 * i] = (v and 0xff).toByte(); pcm[2 * i + 1] = ((v shr 8) and 0xff).toByte()
        }
        val header = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray()).putInt(36 + pcm.size).put("WAVE".toByteArray()).put("fmt ".toByteArray())
            .putInt(16).putShort(1).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16).put("data".toByteArray()).putInt(pcm.size)
        f.outputStream().use { it.write(header.array()); it.write(pcm) }
    }

    override fun onDestroy() {
        super.onDestroy()
        bg.shutdown()
        tts?.close()
    }
}

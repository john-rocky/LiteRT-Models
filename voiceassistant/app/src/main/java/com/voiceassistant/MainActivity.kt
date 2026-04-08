package com.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.AutomaticGainControl
import android.media.audiofx.NoiseSuppressor
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import java.util.ArrayDeque
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

private const val TAG = "VoiceAssistant"

/**
 * Number of preroll VAD chunks (~160 ms) kept in a ring buffer while listening.
 * When SPEECH_START fires, the preroll is prepended to the captured audio so
 * Whisper sees the leading phoneme that triggered the detector.
 */
private const val PREROLL_CHUNKS = 5

/**
 * Hard cap on a single captured turn (in 32 ms chunks). At 16 kHz / 512 samples
 * per chunk this is ~30 s, matching Whisper's 30-second window.
 */
private const val MAX_TURN_CHUNKS = 30 * 1000 / 32

/**
 * VAD probability threshold required to barge in while the assistant is
 * actively speaking. Higher than the normal 0.5 to give AEC a fighting chance
 * against TTS bleed-through into the mic.
 */
private const val BARGE_IN_THRESHOLD = 0.7f

/** UI state for the listen/speak/think turn machine. */
private enum class UiState { IDLE, LISTENING, CAPTURING, THINKING, SPEAKING }

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var stateLabel: TextView
    private lateinit var voiceSpinner: Spinner
    private lateinit var listenButton: Button
    private lateinit var transcriptText: TextView
    private lateinit var responseText: TextView

    private var assistant: VoiceAssistant? = null
    private val initExecutor = Executors.newSingleThreadExecutor()
    private val turnExecutor = Executors.newSingleThreadExecutor()

    // ── Listen-mode state (mic loop owns these) ───────────────────────────
    @Volatile private var listening = false
    private var micThread: Thread? = null
    private val tracker = SegmentTracker()

    // ── Per-turn cancellation + active player handle (for barge-in) ───────
    /** Cancellation flag for the currently in-flight processing turn, if any. */
    private val currentTurnCancel = AtomicReference<AtomicBoolean?>(null)
    /** AudioTrack currently playing back, so we can hard-stop it on barge-in. */
    private val currentTrack = AtomicReference<AudioTrack?>(null)

    // ── UI state (main thread only, except via runOnUiThread) ─────────────
    @Volatile private var uiState: UiState = UiState.IDLE

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startListening()
        else statusText.text = "Microphone permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildUi())

        // Load all models in the background
        statusText.text = "Loading models (Whisper + SmolLM2 + Kokoro + Silero VAD)..."
        initExecutor.execute {
            try {
                val a = VoiceAssistant(this)
                runOnUiThread {
                    assistant = a
                    val voices = a.voiceIds
                    if (voices.isEmpty()) {
                        statusText.text = "No voices found. Push them via adb."
                    } else {
                        voiceSpinner.adapter = ArrayAdapter(
                            this, android.R.layout.simple_spinner_dropdown_item, voices
                        )
                        statusText.text = "Ready (TTS ${a.ttsProvider})"
                        listenButton.isEnabled = true
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Init failed", e)
                runOnUiThread { statusText.text = "Init failed: ${e.message}" }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopListening()
        cancelInFlightTurn()
        assistant?.close()
        initExecutor.shutdown()
        turnExecutor.shutdown()
    }

    // ── Listen toggle ─────────────────────────────────────────────────────

    private fun onListenClick() {
        if (listening) {
            stopListening()
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            } else {
                startListening()
            }
        }
    }

    private fun startListening() {
        val a = assistant ?: return
        listening = true
        tracker.reset()
        a.vad.reset()

        listenButton.text = "Stop listening"
        listenButton.setBackgroundColor(0xFFB71C1C.toInt())
        setUiState(UiState.LISTENING)
        transcriptText.text = "..."
        responseText.text = "..."

        micThread = thread(name = "va-mic", isDaemon = true) { runMicLoop(a) }
    }

    private fun stopListening() {
        if (!listening) return
        listening = false
        micThread?.join(500)
        micThread = null
        cancelInFlightTurn()

        listenButton.text = "Listen"
        listenButton.setBackgroundColor(0xFF1565C0.toInt())
        setUiState(UiState.IDLE)
    }

    /**
     * Hard-stop whatever the assistant is currently doing: flips the in-flight
     * turn's cancellation flag and stops the current AudioTrack so playback
     * dies on the spot. Used for barge-in and onDestroy.
     */
    private fun cancelInFlightTurn() {
        currentTurnCancel.getAndSet(null)?.set(true)
        currentTrack.getAndSet(null)?.let { t ->
            try { t.pause(); t.flush(); t.stop(); t.release() } catch (_: Exception) {}
        }
    }

    // ── Mic loop ──────────────────────────────────────────────────────────

    private fun runMicLoop(a: VoiceAssistant) {
        val sampleRate = VadDetector.SAMPLE_RATE
        val chunk = VadDetector.CHUNK_SAMPLES

        // VOICE_COMMUNICATION enables the platform AEC/AGC/NS chain on most
        // modern Android devices. Combined with the post-creation AEC effect
        // below, this gives barge-in a fighting chance against the assistant's
        // own TTS bleeding into the mic.
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val readSamples = maxOf(minBuf / 2, chunk * 4)

        val record = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                readSamples * 4,
            )
        } catch (e: SecurityException) {
            runOnUiThread { statusText.text = "Microphone permission denied" }
            return
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            runOnUiThread { statusText.text = "AudioRecord init failed" }
            record.release()
            return
        }
        attachAudioEffects(record.audioSessionId)

        record.startRecording()
        Log.i(TAG, "Mic loop started: rate=$sampleRate readSamples=$readSamples")

        val readBuf = ShortArray(readSamples)
        val pending = FloatArray(readSamples * 2)
        var pendingLen = 0

        // Preroll ring buffer of recent silent chunks
        val preroll: ArrayDeque<FloatArray> = ArrayDeque(PREROLL_CHUNKS + 1)

        // Capture buffer for the in-progress utterance, or null if not capturing
        var capture: ArrayList<FloatArray>? = null

        try {
            while (listening) {
                val n = record.read(readBuf, 0, readBuf.size)
                if (n <= 0) continue
                if (pendingLen + n > pending.size) pendingLen = 0  // defensive reset
                for (i in 0 until n) pending[pendingLen++] = readBuf[i].toFloat() / 32768f

                var off = 0
                while (pendingLen - off >= chunk) {
                    val frame = FloatArray(chunk)
                    System.arraycopy(pending, off, frame, 0, chunk)
                    off += chunk

                    val rawProb = a.vad.processChunk(frame)

                    // Echo guard: when the assistant is currently SPEAKING its
                    // own TTS, raise the SPEECH_START bar so AEC bleed-through
                    // does not false-trigger barge-in. Once a real barge-in
                    // does fire, uiState transitions to CAPTURING and the
                    // normal threshold takes over for SPEECH_END detection.
                    val prob = if (uiState == UiState.SPEAKING && !tracker.isSpeaking
                        && rawProb < BARGE_IN_THRESHOLD
                    ) {
                        0f
                    } else rawProb
                    val event = tracker.update(prob)

                    when (event) {
                        SegmentTracker.Event.SPEECH_START -> {
                            // Barge-in: kill any in-flight processing/playback
                            // before starting a new capture.
                            cancelInFlightTurn()
                            capture = ArrayList<FloatArray>(64).apply {
                                addAll(preroll)
                                add(frame)
                            }
                            preroll.clear()
                            setUiState(UiState.CAPTURING)
                        }
                        SegmentTracker.Event.SPEECH_END -> {
                            val captured = capture
                            capture = null
                            if (captured != null && captured.isNotEmpty()) {
                                submitTurn(a, flatten(captured))
                            }
                        }
                        SegmentTracker.Event.NONE -> {
                            val cap = capture
                            if (cap != null) {
                                cap.add(frame)
                                if (cap.size >= MAX_TURN_CHUNKS) {
                                    // Hard cap: force-end the capture as if VAD had emitted SPEECH_END.
                                    capture = null
                                    submitTurn(a, flatten(cap))
                                }
                            } else {
                                preroll.addLast(frame)
                                while (preroll.size > PREROLL_CHUNKS) preroll.removeFirst()
                            }
                        }
                    }
                }
                if (off > 0) {
                    val rem = pendingLen - off
                    if (rem > 0) System.arraycopy(pending, off, pending, 0, rem)
                    pendingLen = rem
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Mic loop error", e)
            runOnUiThread { statusText.text = "Mic loop error: ${e.message}" }
        } finally {
            try { record.stop() } catch (_: Exception) {}
            record.release()
            Log.i(TAG, "Mic loop stopped")
        }
    }

    /** Flatten a list of equal-length chunks into one contiguous FloatArray. */
    private fun flatten(chunks: List<FloatArray>): FloatArray {
        var total = 0
        for (c in chunks) total += c.size
        val out = FloatArray(total)
        var idx = 0
        for (c in chunks) { System.arraycopy(c, 0, out, idx, c.size); idx += c.size }
        return out
    }

    private fun attachAudioEffects(sessionId: Int) {
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                AcousticEchoCanceler.create(sessionId)?.enabled = true
                Log.i(TAG, "AEC enabled")
            }
            if (NoiseSuppressor.isAvailable()) {
                NoiseSuppressor.create(sessionId)?.enabled = true
            }
            if (AutomaticGainControl.isAvailable()) {
                AutomaticGainControl.create(sessionId)?.enabled = true
            }
        } catch (e: Exception) {
            Log.w(TAG, "Audio effects attach failed: ${e.message}")
        }
    }

    // ── Turn processing ───────────────────────────────────────────────────

    private fun submitTurn(a: VoiceAssistant, samples: FloatArray) {
        // Each turn carries its own cancel flag. We hand the flag's reference
        // off to currentTurnCancel so a future SPEECH_START can flip it.
        val cancel = AtomicBoolean(false)
        currentTurnCancel.set(cancel)
        setUiState(UiState.THINKING)
        runOnUiThread {
            transcriptText.text = "..."
            responseText.text = "..."
        }

        val voiceId = (voiceSpinner.selectedItem as? String) ?: "af_heart"

        turnExecutor.execute {
            val chunkQueue = LinkedBlockingQueue<FloatArray>()
            val EOS = FloatArray(0)

            // Player thread: consumes synthesized chunks until EOS or cancel.
            val player = thread(name = "va-player", isDaemon = true) {
                var startedPlayback = false
                while (true) {
                    if (cancel.get()) break
                    val c = try { chunkQueue.take() } catch (_: InterruptedException) { break }
                    if (c === EOS) break
                    if (cancel.get()) break
                    if (!startedPlayback) {
                        startedPlayback = true
                        setUiState(UiState.SPEAKING)
                    }
                    playChunkBlocking(c, cancel)
                    if (cancel.get()) break
                }
            }

            try {
                a.processStreaming(
                    samples = samples,
                    voiceId = voiceId,
                    cancelled = { cancel.get() },
                    onTranscript = { value ->
                        runOnUiThread { transcriptText.text = value }
                    },
                    onResponse = { value ->
                        runOnUiThread { responseText.text = value }
                    },
                    onAudioChunk = { c -> if (!cancel.get()) chunkQueue.put(c) },
                )
                chunkQueue.put(EOS)
                player.join()

                runOnUiThread {
                    if (!cancel.get()) {
                        statusText.text = "STT ${a.lastWhisperMs}ms | " +
                            "LM ${a.lastLmMs}ms | TTS ${a.lastTtsMs}ms"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Turn failed", e)
                player.interrupt()
                runOnUiThread { statusText.text = "Turn error: ${e.message}" }
            } finally {
                // Only return to LISTENING if a newer turn hasn't taken over
                // and the listen toggle is still on.
                if (currentTurnCancel.compareAndSet(cancel, null) && listening) {
                    setUiState(UiState.LISTENING)
                }
            }
        }
    }

    /**
     * Play one float-PCM chunk. Stores the AudioTrack reference in
     * [currentTrack] so a barge-in can hard-stop it from another thread.
     * Polls [cancel] roughly every 50 ms during the playback wait.
     */
    private fun playChunkBlocking(samples: FloatArray, cancel: AtomicBoolean) {
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(KokoroSynthesizer.SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(samples.size * 4)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        currentTrack.set(track)
        try {
            track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
            track.play()
            val totalMs = (samples.size * 1000L) / KokoroSynthesizer.SAMPLE_RATE
            val deadline = System.currentTimeMillis() + totalMs + 100
            while (System.currentTimeMillis() < deadline) {
                if (cancel.get()) break
                try { Thread.sleep(40) } catch (_: InterruptedException) { break }
            }
        } finally {
            try { track.pause(); track.flush(); track.stop() } catch (_: Exception) {}
            try { track.release() } catch (_: Exception) {}
            currentTrack.compareAndSet(track, null)
        }
    }

    // ── UI helpers ────────────────────────────────────────────────────────

    private fun setUiState(s: UiState) {
        uiState = s
        runOnUiThread {
            stateLabel.text = when (s) {
                UiState.IDLE      -> "○ idle"
                UiState.LISTENING -> "● listening"
                UiState.CAPTURING -> "◉ speaking…"
                UiState.THINKING  -> "💭 thinking…"
                UiState.SPEAKING  -> "🔊 replying…"
            }
            stateLabel.setTextColor(when (s) {
                UiState.IDLE      -> 0xFF888888.toInt()
                UiState.LISTENING -> 0xFF4FC3F7.toInt()
                UiState.CAPTURING -> 0xFF4CAF50.toInt()
                UiState.THINKING  -> 0xFFFFB74D.toInt()
                UiState.SPEAKING  -> 0xFFBA68C8.toInt()
            })
        }
    }

    private fun buildUi(): LinearLayout {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF1A1A1A.toInt())
            setPadding(24, 48, 24, 24)
        }

        statusText = TextView(this).apply {
            textSize = 14f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(0, 8, 0, 8)
            text = "Loading models..."
        }
        root.addView(statusText)

        stateLabel = TextView(this).apply {
            textSize = 22f
            setPadding(0, 16, 0, 8)
            text = "○ idle"
            setTextColor(0xFF888888.toInt())
        }
        root.addView(stateLabel)

        addLabel(root, "Voice")
        voiceSpinner = Spinner(this)
        root.addView(voiceSpinner)

        listenButton = Button(this).apply {
            text = "Listen"
            isEnabled = false
            gravity = Gravity.CENTER
            setBackgroundColor(0xFF1565C0.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 18f
            setOnClickListener { onListenClick() }
        }
        root.addView(listenButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = 24; bottomMargin = 16 })

        addLabel(root, "You said")
        transcriptText = makeOutputView()
        root.addView(transcriptText)

        addLabel(root, "Assistant replied")
        responseText = makeOutputView()
        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }
        scrollView.addView(responseText)
        root.addView(scrollView)

        return root
    }

    private fun addLabel(parent: LinearLayout, text: String) {
        val label = TextView(this).apply {
            this.text = text
            textSize = 13f
            setTextColor(0xFF999999.toInt())
            setPadding(0, 16, 0, 4)
        }
        parent.addView(label)
    }

    private fun makeOutputView(): TextView = TextView(this).apply {
        textSize = 17f
        setTextColor(0xFFEEEEEE.toInt())
        setBackgroundColor(0xFF2A2A2A.toInt())
        setPadding(24, 24, 24, 24)
        text = "..."
        setTextIsSelectable(true)
    }
}

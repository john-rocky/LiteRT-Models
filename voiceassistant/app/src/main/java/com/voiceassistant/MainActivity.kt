package com.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
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
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue

private const val TAG = "VoiceAssistant"
private const val MAX_RECORD_SEC = 10

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var voiceSpinner: Spinner
    private lateinit var recordButton: Button
    private lateinit var transcriptText: TextView
    private lateinit var responseText: TextView

    private var assistant: VoiceAssistant? = null
    private val initExecutor = Executors.newSingleThreadExecutor()
    private val pipelineExecutor = Executors.newSingleThreadExecutor()

    private var audioRecord: AudioRecord? = null
    @Volatile private var isRecording = false
    private var recordedSamples = mutableListOf<Float>()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) toggleRecording()
        else statusText.text = "Microphone permission denied"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFF1A1A1A.toInt())
            setPadding(24, 48, 24, 24)
        }

        statusText = TextView(this).apply {
            textSize = 14f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(0, 8, 0, 8)
            text = "Loading models (Whisper + SmolLM2 + Kokoro)..."
        }
        root.addView(statusText)

        addLabel(root, "Voice")
        voiceSpinner = Spinner(this)
        root.addView(voiceSpinner)

        recordButton = Button(this).apply {
            text = "Hold to talk"
            isEnabled = false
            gravity = Gravity.CENTER
            setBackgroundColor(0xFF1565C0.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            textSize = 18f
            setOnClickListener { onRecordClick() }
        }
        root.addView(recordButton, LinearLayout.LayoutParams(
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

        setContentView(root)

        // Load models in background
        initExecutor.execute {
            try {
                assistant = VoiceAssistant(this)
                runOnUiThread {
                    val voices = assistant!!.voiceIds
                    if (voices.isEmpty()) {
                        statusText.text = "No voices found. Push via adb."
                    } else {
                        voiceSpinner.adapter = ArrayAdapter(
                            this, android.R.layout.simple_spinner_dropdown_item, voices
                        )
                        statusText.text = "Ready (TTS: ${assistant!!.ttsProvider})"
                        recordButton.isEnabled = true
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Init failed", e)
                runOnUiThread { statusText.text = "Init failed: ${e.message}" }
            }
        }
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

    private fun onRecordClick() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        toggleRecording()
    }

    private fun toggleRecording() {
        if (isRecording) stopRecording() else startRecording()
    }

    private fun startRecording() {
        val bufferSize = AudioRecord.getMinBufferSize(
            MelSpectrogram.SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                MelSpectrogram.SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize * 4
            )
        } catch (e: SecurityException) {
            statusText.text = "Microphone permission denied"
            return
        }

        audioRecord?.startRecording()
        isRecording = true
        synchronized(recordedSamples) { recordedSamples.clear() }
        recordButton.text = "Stop & process"
        recordButton.setBackgroundColor(0xFFB71C1C.toInt())
        statusText.text = "Recording..."
        transcriptText.text = "..."
        responseText.text = "..."

        initExecutor.execute {
            val buffer = ShortArray(bufferSize / 2)
            val maxSamples = MelSpectrogram.SAMPLE_RATE * MAX_RECORD_SEC
            while (isRecording && (recordedSamples.size < maxSamples)) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: -1
                if (read > 0) {
                    synchronized(recordedSamples) {
                        for (i in 0 until read) {
                            if (recordedSamples.size >= maxSamples) break
                            recordedSamples.add(buffer[i].toFloat() / 32768f)
                        }
                    }
                    val sec = recordedSamples.size.toFloat() / MelSpectrogram.SAMPLE_RATE
                    runOnUiThread {
                        if (isRecording) statusText.text = "Recording ${"%.1f".format(sec)}s"
                    }
                }
            }
            if (isRecording) runOnUiThread { stopRecording() }
        }
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        recordButton.text = "Hold to talk"
        recordButton.setBackgroundColor(0xFF1565C0.toInt())
        recordButton.isEnabled = false

        val samples: FloatArray
        synchronized(recordedSamples) { samples = recordedSamples.toFloatArray() }
        if (samples.isEmpty()) {
            statusText.text = "No audio captured"
            recordButton.isEnabled = true
            return
        }

        statusText.text = "Transcribing..."
        val voiceId = (voiceSpinner.selectedItem as? String) ?: "af_heart"

        pipelineExecutor.execute {
            // Producer/consumer queue: LM/TTS thread pushes chunks, playback thread pops.
            // Sentinel `EMPTY` marks "no more chunks coming".
            val chunkQueue = LinkedBlockingQueue<FloatArray>()
            val EMPTY = FloatArray(0)

            // Player thread: pop chunks and play each via MODE_STATIC AudioTrack
            // (the same path that worked reliably for the non-streaming demo).
            val playerThread = Thread {
                var startedPlayback = false
                while (true) {
                    val chunk = try { chunkQueue.take() } catch (_: InterruptedException) { break }
                    if (chunk === EMPTY) break
                    if (!startedPlayback) {
                        startedPlayback = true
                        runOnUiThread { statusText.text = "Speaking..." }
                    }
                    playChunkBlocking(chunk)
                }
            }
            playerThread.start()

            try {
                assistant!!.processStreaming(
                    samples = samples,
                    voiceId = voiceId,
                    onTranscript = { value ->
                        runOnUiThread {
                            transcriptText.text = value
                            statusText.text = "Generating response..."
                        }
                    },
                    onResponse = { value ->
                        runOnUiThread { responseText.text = value }
                    },
                    onAudioChunk = { chunk -> chunkQueue.put(chunk) },
                )

                // Signal end-of-stream and wait for playback to finish
                chunkQueue.put(EMPTY)
                playerThread.join()

                runOnUiThread {
                    statusText.text = "STT ${assistant!!.lastWhisperMs}ms | " +
                        "LM ${assistant!!.lastLmMs}ms | " +
                        "TTS ${assistant!!.lastTtsMs}ms"
                    recordButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Pipeline failed", e)
                playerThread.interrupt()
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    recordButton.isEnabled = true
                }
            }
        }
    }

    /**
     * Play a single audio chunk synchronously using MODE_STATIC AudioTrack.
     * Blocks until playback completes. This is the same code path the
     * non-streaming demo used and is known to work on Pixel 8a.
     */
    private fun playChunkBlocking(samples: FloatArray) {
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
        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
        track.play()
        val durMs = (samples.size * 1000L) / KokoroSynthesizer.SAMPLE_RATE
        try { Thread.sleep(durMs + 100) } catch (_: InterruptedException) {}
        try { track.stop(); track.release() } catch (_: Exception) {}
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            isRecording = false
            audioRecord?.stop()
            audioRecord?.release()
        }
        assistant?.close()
        initExecutor.shutdown()
        pipelineExecutor.shutdown()
    }
}

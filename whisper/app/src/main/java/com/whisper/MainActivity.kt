package com.whisper

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import java.nio.ByteOrder
import java.util.concurrent.Executors

private const val TAG = "Whisper"

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var pickButton: Button
    private lateinit var recordButton: Button
    private lateinit var resultText: TextView
    private lateinit var langSpinner: Spinner

    private var transcriber: WhisperTranscriber? = null
    private val executor = Executors.newSingleThreadExecutor()
    private var isProcessing = false

    // Recording state
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordedSamples = mutableListOf<Float>()

    private val audioPicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { loadAndTranscribe(it) } }

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
            textSize = 15f
            setTextColor(0xFFCCCCCC.toInt())
            setPadding(0, 8, 0, 8)
            text = "Loading model..."
        }
        root.addView(statusText)

        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 4, 0, 4)
        }

        pickButton = Button(this).apply {
            text = "Select Audio"
            isEnabled = false
            setOnClickListener { audioPicker.launch("audio/*") }
        }
        buttonRow.addView(pickButton)

        recordButton = Button(this).apply {
            text = "Record"
            isEnabled = false
            setOnClickListener { onRecordClick() }
            setBackgroundColor(0xFF333333.toInt())
            setTextColor(0xFFFFFFFF.toInt())
        }
        buttonRow.addView(recordButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { marginStart = 8 })

        // Language selector
        langSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("en", "ja", "zh", "ko", "fr", "de", "es", "it", "pt", "ru")
            )
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { marginStart = 16 }
        }
        buttonRow.addView(langSpinner)
        root.addView(buttonRow)

        // Transcription result
        val resultLabel = TextView(this).apply {
            text = "Transcription"
            textSize = 16f
            setTextColor(0xFFFFFFFF.toInt())
            setPadding(0, 24, 0, 8)
        }
        root.addView(resultLabel)

        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f
            )
        }

        resultText = TextView(this).apply {
            textSize = 18f
            setTextColor(0xFFEEEEEE.toInt())
            setBackgroundColor(0xFF2A2A2A.toInt())
            setPadding(24, 24, 24, 24)
            text = "Select audio or tap Record..."
            setTextIsSelectable(true)
        }
        scrollView.addView(resultText)
        root.addView(scrollView)

        setContentView(root)

        // Load model
        executor.execute {
            try {
                transcriber = WhisperTranscriber(this)
                runOnUiThread {
                    statusText.text = "Ready (encoder: ${transcriber!!.acceleratorName})"
                    pickButton.isEnabled = true
                    recordButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
            }
        }
    }

    // ── Record button ──────────────────────────────────────────────────────

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
        if (isRecording) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        if (isProcessing) return

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
        recordedSamples.clear()
        recordButton.text = "Stop"
        recordButton.setBackgroundColor(0xFFB71C1C.toInt())
        pickButton.isEnabled = false
        statusText.text = "Recording... (max 30s)"

        // Read audio in background
        executor.execute {
            val buffer = ShortArray(bufferSize / 2)
            val maxSamples = MelSpectrogram.N_SAMPLES  // 30s max

            while (isRecording && recordedSamples.size < maxSamples) {
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
                        statusText.text = "Recording... ${String.format("%.1f", sec)}s"
                    }
                }
            }

            // Auto-stop at 30s
            if (isRecording) {
                runOnUiThread { stopRecording() }
            }
        }
    }

    private fun stopRecording() {
        isRecording = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null

        recordButton.text = "Record"
        recordButton.setBackgroundColor(0xFF333333.toInt())

        val samples: FloatArray
        synchronized(recordedSamples) {
            samples = recordedSamples.toFloatArray()
        }

        if (samples.isEmpty()) {
            statusText.text = "No audio recorded"
            pickButton.isEnabled = true
            return
        }

        transcribeSamples(samples)
    }

    // ── Transcribe from file ───────────────────────────────────────────────

    private fun loadAndTranscribe(uri: Uri) {
        if (isProcessing || transcriber == null) return
        isProcessing = true
        pickButton.isEnabled = false
        recordButton.isEnabled = false
        statusText.text = "Decoding audio..."
        resultText.text = ""

        executor.execute {
            try {
                val samples = decodeAudio(uri)
                    ?: throw Exception("Failed to decode audio")
                runOnUiThread { transcribeSamples(samples) }
            } catch (e: Exception) {
                Log.e(TAG, "Audio decode failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                    recordButton.isEnabled = true
                    isProcessing = false
                }
            }
        }
    }

    // ── Common transcribe ──────────────────────────────────────────────────

    private fun transcribeSamples(samples: FloatArray) {
        if (isProcessing && !isRecording) return  // prevent double-run from file path
        isProcessing = true
        pickButton.isEnabled = false
        recordButton.isEnabled = false

        val language = langSpinner.selectedItem as String
        val durationSec = samples.size.toFloat() / MelSpectrogram.SAMPLE_RATE
        statusText.text = "Transcribing ${String.format("%.1f", durationSec)}s audio..."
        resultText.text = ""

        executor.execute {
            try {
                val text = transcriber!!.transcribe(samples, language)
                runOnUiThread {
                    resultText.text = text.ifEmpty { "(no speech detected)" }
                    statusText.text = "Encode: ${transcriber!!.lastEncodeMs}ms | " +
                        "Decode: ${transcriber!!.lastDecodeMs}ms | " +
                        "${String.format("%.1f", durationSec)}s audio"
                    pickButton.isEnabled = true
                    recordButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(TAG, "Transcription failed", e)
                runOnUiThread {
                    statusText.text = "Error: ${e.message}"
                    pickButton.isEnabled = true
                    recordButton.isEnabled = true
                }
            }
            isProcessing = false
        }
    }

    // ── Audio file decoding ────────────────────────────────────────────────

    private fun decodeAudio(uri: Uri): FloatArray? {
        val extractor = MediaExtractor()
        extractor.setDataSource(this, uri, null)

        var audioTrackIdx = -1
        var audioFormat: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) {
                audioTrackIdx = i
                audioFormat = format
                break
            }
        }
        if (audioTrackIdx < 0 || audioFormat == null) return null

        extractor.selectTrack(audioTrackIdx)
        val mime = audioFormat.getString(MediaFormat.KEY_MIME)!!
        val sampleRate = audioFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = audioFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(audioFormat, null, null, 0)
        codec.start()

        val pcmSamples = mutableListOf<Float>()
        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false

        while (!outputDone) {
            if (!inputDone) {
                val inputIdx = codec.dequeueInputBuffer(10_000)
                if (inputIdx >= 0) {
                    val inputBuf = codec.getInputBuffer(inputIdx)!!
                    val size = extractor.readSampleData(inputBuf, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(inputIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inputIdx, 0, size, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outputIdx = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            if (outputIdx >= 0) {
                val outputBuf = codec.getOutputBuffer(outputIdx)!!
                val shortBuf = outputBuf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                while (shortBuf.hasRemaining()) {
                    pcmSamples.add(shortBuf.get().toFloat() / 32768f)
                }
                codec.releaseOutputBuffer(outputIdx, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    outputDone = true
                }
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        val monoSamples = if (channels > 1) {
            FloatArray(pcmSamples.size / channels) { i ->
                var sum = 0f
                for (ch in 0 until channels) sum += pcmSamples[i * channels + ch]
                sum / channels
            }
        } else {
            pcmSamples.toFloatArray()
        }

        return if (sampleRate != MelSpectrogram.SAMPLE_RATE) {
            resample(monoSamples, sampleRate, MelSpectrogram.SAMPLE_RATE)
        } else {
            monoSamples
        }
    }

    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return input
        val ratio = fromRate.toDouble() / toRate
        val outputLen = (input.size / ratio).toInt()
        val output = FloatArray(outputLen)
        for (i in 0 until outputLen) {
            val srcPos = i * ratio
            val srcIdx = srcPos.toInt()
            val frac = (srcPos - srcIdx).toFloat()
            output[i] = if (srcIdx + 1 < input.size) {
                input[srcIdx] * (1f - frac) + input[srcIdx + 1] * frac
            } else {
                input[srcIdx.coerceAtMost(input.size - 1)]
            }
        }
        return output
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            isRecording = false
            audioRecord?.stop()
            audioRecord?.release()
        }
        transcriber?.close()
        executor.shutdown()
    }
}

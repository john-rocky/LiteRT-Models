package com.kokoro

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.view.Gravity
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import org.json.JSONArray
import java.util.concurrent.Executors

private const val TAG = "Kokoro"

class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var phraseSpinner: Spinner
    private lateinit var voiceSpinner: Spinner
    private lateinit var inputEdit: EditText
    private lateinit var phraseText: TextView
    private lateinit var playButton: Button

    private var synth: KokoroSynthesizer? = null
    private var enPhonemizer: EnglishPhonemizer? = null
    private var jaPhonemizer: JapanesePhonemizer? = null
    private val executor = Executors.newSingleThreadExecutor()

    /** Each entry: { text, language, token_ids: [int...] } */
    private lateinit var demoPhrases: List<DemoPhrase>

    data class DemoPhrase(val text: String, val language: String, val tokenIds: IntArray)

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

        // Free-form text input (overrides demo phrase if non-empty)
        addLabel(root, "Custom text (overrides preset)")
        inputEdit = EditText(this).apply {
            setBackgroundColor(0xFF2A2A2A.toInt())
            setTextColor(0xFFEEEEEE.toInt())
            setHintTextColor(0xFF888888.toInt())
            setPadding(24, 16, 24, 16)
            hint = "Type any text — leave empty to use preset below"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            maxLines = 3
            textSize = 16f
        }
        root.addView(inputEdit)

        // Demo phrase picker
        addLabel(root, "Preset phrase")
        phraseSpinner = Spinner(this)
        root.addView(phraseSpinner)

        // Voice picker
        addLabel(root, "Voice")
        voiceSpinner = Spinner(this)
        root.addView(voiceSpinner)

        // Selected text preview
        addLabel(root, "Will speak")
        val scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0, 1f
            )
        }
        phraseText = TextView(this).apply {
            textSize = 18f
            setTextColor(0xFFEEEEEE.toInt())
            setBackgroundColor(0xFF2A2A2A.toInt())
            setPadding(24, 24, 24, 24)
            text = "..."
            setTextIsSelectable(true)
        }
        scrollView.addView(phraseText)
        root.addView(scrollView)

        // Play button
        playButton = Button(this).apply {
            text = "Generate & Play"
            isEnabled = false
            gravity = Gravity.CENTER
            setBackgroundColor(0xFF2E7D32.toInt())
            setTextColor(0xFFFFFFFF.toInt())
            setOnClickListener { onPlayClick() }
        }
        root.addView(playButton, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = 24 })

        setContentView(root)

        // Load demo phrases (small JSON in assets)
        try {
            demoPhrases = loadDemoPhrases()
            phraseSpinner.adapter = ArrayAdapter(
                this, android.R.layout.simple_spinner_dropdown_item,
                demoPhrases.map { "[${it.language}] ${it.text.take(40)}" }
            )
            phraseSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                    if (inputEdit.text.isNullOrBlank()) {
                        phraseText.text = demoPhrases[position].text
                    }
                }
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load demo phrases", e)
            statusText.text = "demo_phrases.json missing — run scripts/convert_kokoro.py"
            return
        }

        // Update preview live as the user types
        inputEdit.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: android.text.Editable?) {
                val txt = s?.toString().orEmpty()
                if (txt.isBlank()) {
                    val pos = phraseSpinner.selectedItemPosition.coerceAtLeast(0)
                    if (pos < demoPhrases.size) phraseText.text = demoPhrases[pos].text
                } else {
                    phraseText.text = txt
                }
            }
        })

        // Load model + phonemizers in background
        executor.execute {
            try {
                synth = KokoroSynthesizer(this)
                val vocab = Phonemizer.loadVocab(this)
                enPhonemizer = EnglishPhonemizer.load(this, vocab)
                jaPhonemizer = JapanesePhonemizer.load(this, vocab)
                runOnUiThread {
                    val voices = synth!!.voiceIds
                    if (voices.isEmpty()) {
                        statusText.text = "No voices found in files/voices/. Push via adb."
                    } else {
                        voiceSpinner.adapter = ArrayAdapter(
                            this, android.R.layout.simple_spinner_dropdown_item, voices
                        )
                        statusText.text = "Ready [${synth!!.providerName}] (${voices.size} voices)"
                        playButton.isEnabled = true
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                runOnUiThread { statusText.text = "Failed: ${e.message}" }
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

    /** Detect language ("en" or "ja") from voice ID prefix. */
    private fun detectLanguage(voiceId: String): String =
        if (voiceId.startsWith("j")) "ja" else "en"

    private fun onPlayClick() {
        val voiceId = voiceSpinner.selectedItem as? String ?: return
        val customText = inputEdit.text?.toString()?.trim().orEmpty()

        playButton.isEnabled = false
        statusText.text = "Synthesizing..."

        executor.execute {
            try {
                val tokenIds: IntArray
                val language: String
                val textForLog: String

                if (customText.isNotEmpty()) {
                    language = detectLanguage(voiceId)
                    val phonemizer = if (language == "ja") jaPhonemizer else enPhonemizer
                    if (phonemizer == null) throw IllegalStateException("Phonemizer not loaded")
                    tokenIds = phonemizer.phonemize(customText)
                    textForLog = "[$language live] ${tokenIds.size} tokens"
                    if (tokenIds.isEmpty()) throw IllegalStateException("Phonemizer produced 0 tokens — all OOV?")
                } else {
                    val phrase = demoPhrases[phraseSpinner.selectedItemPosition]
                    tokenIds = phrase.tokenIds
                    language = phrase.language
                    textForLog = "[$language preset] ${tokenIds.size} tokens"
                }

                val audio = synth!!.synthesize(tokenIds, voiceId, 1.0f)
                val durSec = audio.size.toFloat() / KokoroSynthesizer.SAMPLE_RATE
                val rtf = synth!!.lastInferMs / (durSec * 1000f)
                runOnUiThread {
                    statusText.text = "[${synth!!.providerName}] $textForLog | " +
                        "Inf ${synth!!.lastInferMs}ms | ${"%.2f".format(durSec)}s | RTF ${"%.2f".format(rtf)}"
                }
                playPcm(audio)
            } catch (e: Exception) {
                Log.e(TAG, "Synthesis failed", e)
                runOnUiThread { statusText.text = "Error: ${e.message}" }
            } finally {
                runOnUiThread { playButton.isEnabled = true }
            }
        }
    }

    private fun playPcm(samples: FloatArray) {
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
        // Auto-release after playback finishes
        executor.execute {
            val durMs = (samples.size * 1000L) / KokoroSynthesizer.SAMPLE_RATE
            Thread.sleep(durMs + 200)
            try {
                track.stop()
                track.release()
            } catch (_: Exception) {}
        }
    }

    private fun loadDemoPhrases(): List<DemoPhrase> {
        val json = assets.open("demo_phrases.json").bufferedReader().use { it.readText() }
        val arr = JSONArray(json)
        val result = mutableListOf<DemoPhrase>()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            val text = obj.getString("text")
            val lang = obj.getString("language")
            val ids = obj.getJSONArray("token_ids")
            val intIds = IntArray(ids.length()) { ids.getInt(it) }
            result.add(DemoPhrase(text, lang, intIds))
        }
        return result
    }

    override fun onDestroy() {
        super.onDestroy()
        synth?.close()
        executor.shutdown()
    }
}

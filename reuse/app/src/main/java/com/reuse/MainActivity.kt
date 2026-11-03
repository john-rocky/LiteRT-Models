package com.reuse

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.util.Log
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.abs
import org.json.JSONArray
import org.json.JSONObject

/**
 * RE-USE speech restoration with LiteRT: pick a recording, restore it on the GPU, play both.
 *
 * Headless mode (measurements, one model load for all files):
 *   adb shell am start -n com.reuse/.MainActivity --es input in/a.wav,in/b.wav [--es model <asset .tflite>]
 *     [--es accel gpu|gpu_fp32|cpu] [--ei threads 4] [--es hop <step as a fraction of the chunk, e.g. 0.75>]
 *     [--ez gpu_cache true] [--ez clear_gpu_cache true] [--es out <tag>]
 * reads 16 kHz WAV files (paths relative to the app's files dir), writes files/out/<tag>/<name>.wav (float32) and
 * files/out/<tag>/result.json (written last: status, error, load time, per-chunk times, the GPU cache files), and
 * logs CLIP / RESULT / ERROR lines under the tag "ReUSE".
 *
 * gpu_cache true turns on LiteRT's GPU program cache in files/gpu_cache (off by default). With LiteRT 2.2.0 and the
 * 2 s graph, the Galaxy S26 and the Pixel 11 Pro wrote a 167 MB cache on the first load but could not read it back
 * ("Deserialization failed: DATA_LOSS"): the second load compiled again, took longer and added another 167 MB.
 */
class MainActivity : Activity() {

  private val worker = Executors.newSingleThreadExecutor() // the model is loaded and run on this one thread
  private var env: Environment? = null
  private var restorer: SpeechRestorer? = null
  private var loadedKey = ""

  private lateinit var status: TextView
  private lateinit var modelPicker: Spinner
  private lateinit var accelPicker: Spinner
  private lateinit var playOriginal: Button
  private lateinit var playRestored: Button
  private var original: FloatArray? = null
  private var restored: FloatArray? = null
  private var player: AudioTrack? = null

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    val models = modelAssets()
    Log.i(TAG, "MODELS default=${models.firstOrNull() ?: "none"} reuse=$models all_tflite=" +
      "${assets.list("")?.filter { it.endsWith(".tflite") }}")
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(36, 90, 36, 36)
    }
    status = TextView(this).apply { textSize = 15f }
    modelPicker = Spinner(this).apply {
      adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, models.ifEmpty { listOf("-") })
    }
    accelPicker = Spinner(this).apply {
      adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, listOf("GPU", "CPU"))
    }
    val pick = Button(this).apply {
      text = "Pick a recording and restore it"
      setOnClickListener { pickAudio() }
    }
    playOriginal = Button(this).apply {
      text = "Play original"
      isEnabled = false
      setOnClickListener { play(original) }
    }
    playRestored = Button(this).apply {
      text = "Play restored"
      isEnabled = false
      setOnClickListener { play(restored) }
    }
    listOf(status, modelPicker, accelPicker, pick, playOriginal, playRestored).forEach { root.addView(it) }
    setContentView(root)

    val input = intent.getStringExtra("input")
    val fromHistory = (intent.flags and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY) != 0
    if (input != null && savedInstanceState == null && !fromHistory) {
      runHeadless(input)
    } else if (models.isEmpty()) {
      showError("No model in this app: put the RE-USE .tflite in app/src/main/assets/ and rebuild (see README).")
    } else {
      status.text = "Pick a speech recording (any format and sample rate, up to $MAX_SECONDS s) to restore it."
    }
  }

  // Only the RE-USE graphs: some phones (Galaxy S26) merge their own .tflite files into every app's asset list.
  // Descending, so that the 2 s graph (reuse_16k_2s_…) comes before the 1 s one and is the default.
  private fun modelAssets() =
    assets.list("")?.filter { it.startsWith("reuse_") && it.endsWith(".tflite") }?.sortedDescending() ?: emptyList()

  private fun gpuCacheDir() = File(filesDir, "gpu_cache")

  /** The files in the GPU cache dir and their sizes in bytes. */
  private fun gpuCacheFiles() =
    JSONObject().apply { gpuCacheDir().listFiles()?.sortedBy { it.name }?.forEach { put(it.name, it.length()) } }

  // One cache per graph file and precision: the asset name and size, and gpu or gpu_fp32.
  private fun gpuCacheKey(model: String, accel: String) =
    "${model.substringBeforeLast('.')}_${assets.openFd(model).use { it.length }}_$accel"

  private fun closeRestorer() {
    restorer?.close()
    restorer = null
    loadedKey = ""
  }

  /** Loads the restorer (or reuses the loaded one). The exception messages are written for the user. */
  private fun open(model: String, accel: String, threads: Int, hopPortion: Double?, gpuCache: Boolean): SpeechRestorer {
    val key = "$model|$accel|$threads|$hopPortion|$gpuCache"
    restorer?.let { if (key == loadedKey) return it }
    closeRestorer()
    val models = modelAssets()
    check(model in models) {
      "Model not found: assets/$model is not in this app (models: ${models.ifEmpty { listOf("none") }.joinToString()})." +
        " Put the RE-USE .tflite in app/src/main/assets/ and rebuild (see README)."
    }
    val e = env ?: Environment.create().also { env = it }
    val options =
      when (accel) {
        "gpu",
        "gpu_fp32" -> {
          val available = e.getAvailableAccelerators()
          check(Accelerator.GPU in available) {
            "GPU unavailable: LiteRT found no GPU accelerator on this phone (available: ${available.joinToString()})." +
              " Choose CPU to run the model there, much slower."
          }
          CompiledModel.Options(Accelerator.GPU).apply {
            val fp32 = accel == "gpu_fp32"
            if (fp32 || gpuCache) {
              gpuOptions =
                CompiledModel.GpuOptions(
                  precision = if (fp32) CompiledModel.GpuOptions.Precision.FP32 else null,
                  serializationDir = if (gpuCache) gpuCacheDir().apply { mkdirs() }.path else null,
                  modelCacheKey = if (gpuCache) gpuCacheKey(model, accel) else null,
                  serializeProgramCache = if (gpuCache) true else null,
                )
            }
          }
        }
        "cpu" -> CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(numThreads = threads) }
        else -> throw IllegalArgumentException("Unknown accelerator '$accel' (gpu, gpu_fp32 or cpu).")
      }
    val r =
      try {
        SpeechRestorer(this, model, options, e, hopPortion)
      } catch (x: LiteRtException) {
        throw IllegalStateException("${accel.uppercase()} could not load $model: ${x.message}", x)
      }
    restorer = r
    loadedKey = key
    return r
  }

  private fun pickAudio() {
    val pick = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE)
      type = "*/*"
      putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/*"))
    }
    startActivityForResult(pick, PICK_AUDIO)
  }

  @Deprecated("Deprecated in Java")
  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    val uri: Uri = data?.data ?: return
    if (requestCode != PICK_AUDIO || resultCode != RESULT_OK) return
    val model = modelPicker.selectedItem as String
    val accel = if (accelPicker.selectedItemPosition == 0) "gpu" else "cpu"
    playOriginal.isEnabled = false
    playRestored.isEnabled = false
    post("Decoding…")
    worker.execute {
      try {
        val x = AudioDecoder.decode(this, uri, SpeechRestorer.SAMPLE_RATE, MAX_SECONDS)
        check(x.isNotEmpty()) { "The file has no audio." }
        post(
          if (accel == "gpu") "Loading $model on GPU… The first load compiles the graph and can take minutes."
          else "Loading $model on CPU…"
        )
        val r = open(model, accel, CPU_THREADS, null, false)
        post("Restoring %.1f s on %s…".format(x.size / SpeechRestorer.SAMPLE_RATE.toDouble(), accel.uppercase()))
        val t0 = System.nanoTime()
        val y = r.enhance(x)
        val s = summarize(r, y.size, (System.nanoTime() - t0) / 1e6)
        original = x
        restored = y
        runOnUiThread {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "Restored %.1f s in %.1f s on %s (RTF %.2f). Per chunk of %.1f s: graph %.0f ms, host %.1f ms."
            .format(
              s.getDouble("seconds"), s.getDouble("wall_ms") / 1000, accel.uppercase(), s.getDouble("rtf"),
              r.chunk / SpeechRestorer.SAMPLE_RATE.toDouble(), s.getDouble("graph_ms_median"),
              s.getDouble("host_ms_median"),
            )
          playOriginal.isEnabled = true
          playRestored.isEnabled = true
        }
      } catch (e: Throwable) {
        Log.e(TAG, "restore", e)
        showError(e.message ?: e.toString())
      }
    }
  }

  /** Measurements: every file with one model load; result.json is written last, also when something fails. */
  private fun runHeadless(input: String) {
    if (Build.VERSION.SDK_INT >= 27) {
      setShowWhenLocked(true)
      setTurnScreenOn(true)
    } else {
      @Suppress("DEPRECATION")
      window.addFlags(WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON)
    }
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    val model = intent.getStringExtra("model") ?: modelAssets().firstOrNull() ?: "reuse.tflite"
    val accel = intent.getStringExtra("accel") ?: "gpu"
    val threads = intent.getIntExtra("threads", CPU_THREADS)
    val hopText = intent.getStringExtra("hop")
    val gpuCache = intent.getBooleanExtra("gpu_cache", false)
    val clearGpuCache = intent.getBooleanExtra("clear_gpu_cache", false)
    val tag = intent.getStringExtra("out") ?: "run"
    val files = input.split(",").map { it.trim() }.filter { it.isNotEmpty() }
    val outDir = File(filesDir, "out/$tag")
    post("Headless run '$tag': ${files.size} file(s), $model on ${accel.uppercase()}…")
    worker.execute {
      val res = JSONObject()
      val clips = JSONArray()
      res.put("tag", tag).put("model", model).put("accel", accel).put("threads", threads)
      res.put("litert", BuildConfig.LITERT_VERSION).put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
      res.put("android", "${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})").put("fingerprint", Build.FINGERPRINT)
      if (Build.VERSION.SDK_INT >= 31) res.put("soc", "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
      res.put("started_ms", System.currentTimeMillis())
      try {
        outDir.deleteRecursively()
        outDir.mkdirs()
        val hopPortion =
          hopText?.let {
            requireNotNull(it.toDoubleOrNull()) {
              "Bad hop '$it': give the step between chunks as a fraction of the chunk, such as 0.875."
            }
          }
        if (clearGpuCache) gpuCacheDir().deleteRecursively()
        res.put("gpu_cache", gpuCache).put("gpu_cache_before", gpuCacheFiles())
        res.put("files_usable_bytes_before", filesDir.usableSpace)
        val t0 = System.nanoTime()
        val r = open(model, accel, threads, hopPortion, gpuCache)
        res.put("load_ms", (System.nanoTime() - t0) / 1e6)
        res.put("gpu_cache_after_load", gpuCacheFiles())
        res.put("available_accelerators", env?.getAvailableAccelerators()?.joinToString())
        res.put("frames", r.frames).put("chunk", r.chunk).put("hop", r.hop).put("hop_portion", r.hop.toDouble() / r.chunk)
        for (name in files) {
          val f = if (name.startsWith("/")) File(name) else File(filesDir, name)
          check(f.isFile) { "Input not found: $f" }
          val audio = WavFile.read(f)
          check(audio.sampleRate == SpeechRestorer.SAMPLE_RATE) {
            "${f.name} is ${audio.sampleRate} Hz; the headless mode takes 16 kHz WAV files."
          }
          post("Restoring ${f.name} on ${accel.uppercase()}…")
          val thermalBefore = thermalStatus()
          val t1 = System.nanoTime()
          val y = r.enhance(audio.samples)
          val wall = (System.nanoTime() - t1) / 1e6
          val stem = f.name.substringBeforeLast('.')
          WavFile.writeFloat(File(outDir, "$stem.wav"), y, SpeechRestorer.SAMPLE_RATE)
          val finite = y.all { it.isFinite() }
          val c = summarize(r, y.size, wall)
          c.put("clip", stem).put("finite", finite).put("thermal_before", thermalBefore).put("thermal_after", thermalStatus())
          c.put("peak", y.filter { it.isFinite() }.maxOfOrNull { abs(it) }?.toDouble() ?: 0.0)
          clips.put(c)
          Log.i(
            TAG,
            "CLIP $tag $stem chunks=${c.getInt("chunks")} graph_ms_median=${c.opt("graph_ms_median")} " +
              "host_ms_median=${c.opt("host_ms_median")} rtf=${c.getDouble("rtf")} finite=$finite",
          )
        }
        // Closed before result.json, so that the GPU cache files listed after it are the ones the close leaves.
        val t2 = System.nanoTime()
        closeRestorer()
        res.put("close_ms", (System.nanoTime() - t2) / 1e6).put("gpu_cache_after_close", gpuCacheFiles())
        res.put("files_usable_bytes_after", filesDir.usableSpace)
        res.put("status", "ok")
      } catch (e: Throwable) {
        res.put("status", "error").put("error", e.message ?: e.toString()).put("error_class", e.javaClass.name)
        Log.e(TAG, "ERROR $tag ${e.message}", e)
        showError(e.message ?: e.toString())
      }
      res.put("clips", clips).put("ended_ms", System.currentTimeMillis())
      outDir.mkdirs()
      val tmp = File(outDir, "result.json.tmp")
      tmp.writeText(res.toString(1))
      tmp.renameTo(File(outDir, "result.json"))
      Log.i(TAG, "RESULT $tag status=${res.getString("status")} clips=${clips.length()}")
      runOnUiThread {
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (res.getString("status") == "ok") {
          status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
          status.text = "Headless run '$tag' done: ${clips.length()} file(s) in files/out/$tag."
        }
      }
    }
  }

  /** Times of the last enhance(): per chunk graph = writeFloat + run + readFloat, host = STFT + iSTFT + overlap-add. */
  private fun summarize(r: SpeechRestorer, samples: Int, wallMs: Double): JSONObject {
    val t = r.chunkTimes
    val graph = t.map { it[1] + it[2] + it[3] }
    val host = t.map { it[0] + it[4] }
    val seconds = samples.toDouble() / SpeechRestorer.SAMPLE_RATE
    val graphTotal = graph.sum()
    val hostTotal = host.sum() + r.finalizeMs
    fun col(i: Int) = JSONArray(t.map { Math.round(it[i] * 1000) / 1000.0 })
    return JSONObject().apply {
      put("samples", samples).put("seconds", seconds).put("chunks", t.size)
      put("stft_ms", col(0)).put("write_ms", col(1)).put("run_ms", col(2)).put("read_ms", col(3))
      put("istft_ola_ms", col(4)).put("finalize_ms", r.finalizeMs)
      put("graph_ms_first", graph.first()).put("host_ms_first", host.first())
      // The median over the chunks after the first one (the first one is reported on its own).
      put("graph_ms_median", median(graph.drop(1)) ?: graph.first())
      put("host_ms_median", median(host.drop(1)) ?: host.first())
      put("graph_ms_total", graphTotal).put("host_ms_total", hostTotal)
      put("rtf", (graphTotal + hostTotal) / 1000 / seconds)
      put("wall_ms", wallMs).put("wall_rtf", wallMs / 1000 / seconds)
    }
  }

  private fun median(v: List<Double>): Double? {
    if (v.isEmpty()) return null
    val s = v.sorted()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
  }

  private fun thermalStatus(): Int =
    if (Build.VERSION.SDK_INT >= 29) (getSystemService(POWER_SERVICE) as PowerManager).currentThermalStatus else -1

  private fun post(text: String) = runOnUiThread {
    status.setBackgroundColor(Color.TRANSPARENT)
    status.text = text
  }

  private fun showError(message: String) = runOnUiThread {
    status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
    status.text = "Failed: $message"
  }

  private fun play(data: FloatArray?) {
    data ?: return
    stopPlayback()
    var peak = 1e-6f
    for (v in data) if (abs(v) > peak) peak = abs(v)
    val scaled = FloatArray(data.size) { data[it] / peak * 0.9f }
    val track =
      AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).build())
        .setAudioFormat(
          AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(SpeechRestorer.SAMPLE_RATE)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .build()
        )
        .setBufferSizeInBytes(scaled.size * 4)
        .setTransferMode(AudioTrack.MODE_STATIC)
        .build()
    track.write(scaled, 0, scaled.size, AudioTrack.WRITE_BLOCKING)
    track.play()
    player = track
  }

  private fun stopPlayback() {
    player?.let { runCatching { it.stop(); it.release() } }
    player = null
  }

  override fun onDestroy() {
    super.onDestroy()
    stopPlayback()
    worker.execute {
      restorer?.close()
      env?.close()
    }
    worker.shutdown()
  }

  companion object {
    private const val TAG = "ReUSE"
    private const val PICK_AUDIO = 1
    private const val MAX_SECONDS = 60
    private const val CPU_THREADS = 4
  }
}

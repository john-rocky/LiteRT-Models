package com.nemotron3diar

import android.app.KeyguardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Log
import com.google.ai.edge.litert.Environment
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Device gate of the whole host: streams a wav through [Nemotron3Diarizer] on the LiteRT GPU, as a microphone would
 * ([push] samples at a time), and writes what it did for scripts/gate_closed_loop.py.
 *
 * Request (files/selftest.json):
 * {"mode": "closed_loop", "wav": "wav/diarization_example_16k.wav", "b_precision": "fp32|default|accum",
 *  "push": 1600, "warmup": 1, "realtime": false, "file_mode": false, "repeats": 1,
 *  "encoder": "nemotron3_diar_encoder_low_latency_fp16.tflite"}
 *
 * "realtime": true paces the pushes at the audio rate (a microphone); timing.json then also has each step's latency
 * from the arrival of the push that completed its chunk to its logits. Without it the steps run back to back.
 * "file_mode": true runs the offline pass over the whole wav instead ([Nemotron3Diarizer.runFile],
 * StreamConfig.OFFLINE, graph B [LiteRtEngine.ENCODER_OFFLINE] by default), "repeats" times back to back with a fresh
 * diarizer each time; the outputs of the last pass are written and every pass's wall time goes to timing.json.
 *
 * Output (getExternalFilesDir(null)/n3d/closed_loop_<precision>_<wav stem>[_rt]/, file mode: file_<precision>_<wav
 * stem>/): out_rows_step<i>.bin (the logits the step emits, [frames x 8] float32 LE), steps.json (per step: chunk,
 * length, cache state, the frame ids of the rows it was fed, ms of mel / graph A / graph B / cache / total; per
 * compression: the kept frame ids and boundary scores), timing.json (compile ms, warm-up ms, medians, RTF,
 * first-chunk latency, device conditions), status.txt. The warm-up runs graph A and B once on zeros after compiling
 * (as the demo does at start-up) and is timed apart.
 */
class ClosedLoopTest(private val ctx: Context, private val request: JSONObject, private val show: (String) -> Unit) {

  private val precision = LiteRtEngine.Precision.parse(request.optString("b_precision", "fp32"))
  private val wavFile = File(ctx.filesDir, request.getString("wav"))
  private val push = request.optInt("push", 1600)
  private val warmup = request.optInt("warmup", 1)
  private val fileMode = request.optBoolean("file_mode", false)
  private val repeats = request.optInt("repeats", 1)
  private val config = if (fileMode) StreamConfig.OFFLINE else StreamConfig.LOW_LATENCY
  private val encoder =
    request.optString("encoder", if (fileMode) LiteRtEngine.ENCODER_OFFLINE else LiteRtEngine.ENCODER)
  private val realtime = request.optBoolean("realtime", false) && !fileMode
  private val outDir =
    File(File(ctx.getExternalFilesDir(null), "n3d"),
      if (fileMode) "file_${precision.label}_${wavFile.nameWithoutExtension}"
      else "closed_loop_${precision.label}_${wavFile.nameWithoutExtension}" + if (realtime) "_rt" else "")
  private val statusFile = File(outDir, "status.txt")
  private val power = ctx.getSystemService(PowerManager::class.java)

  private fun status(line: String) {
    Log.i(TAG, line)
    statusFile.appendText(line + "\n")
    show(line)
  }

  private fun conditions(): JSONObject {
    val battery = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val plugged = battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1
    val headroom = if (Build.VERSION.SDK_INT >= 30) power.getThermalHeadroom(0) else Float.NaN
    return JSONObject()
      .put("screen_interactive", power.isInteractive)
      .put("keyguard_locked", ctx.getSystemService(KeyguardManager::class.java).isKeyguardLocked)
      .put("plugged", when (plugged) {
        0 -> "battery"
        BatteryManager.BATTERY_PLUGGED_AC -> "ac"
        BatteryManager.BATTERY_PLUGGED_USB -> "usb"
        BatteryManager.BATTERY_PLUGGED_WIRELESS -> "wireless"
        else -> "unknown($plugged)"
      })
      .put("battery_level", battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1)
      .put("battery_temp_c", (battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -10) ?: -10) / 10.0)
      .put("thermal_status", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else -1)
      .put("thermal_headroom", if (headroom.isNaN()) JSONObject.NULL else headroom.toDouble())
  }

  private fun median(v: List<Double>): Double {
    val s = v.sorted()
    return if (s.isEmpty()) Double.NaN else if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
  }

  fun run() {
    outDir.mkdirs()
    outDir.listFiles()?.forEach { it.delete() }
    val timing =
      JSONObject()
        .put("litert_version", BuildConfig.LITERT_VERSION)
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
        .put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else "")
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("b_precision", precision.label)
        .put("a_precision", "fp32")
        .put("encoder", encoder)
        .put("wav", wavFile.name)
        .put("push_samples", push)
        .put("realtime", realtime)
        .put("file_mode", fileMode)
        .put("config", config.name)
        .put("conditions_start", conditions())
    status("BEGIN ${if (fileMode) "file" else "closed_loop"} ${precision.label} ${wavFile.name} " +
      "litert=${BuildConfig.LITERT_VERSION}")
    val wav = WavReader.read(wavFile)
    check(wav.sampleRate == MelFrontend.SAMPLE_RATE) { "${wavFile.name}: ${wav.sampleRate} Hz" }
    val assets = ctx.assets
    val fb = StepLog.floatsFromStream(assets.open("frontend_mel128_257.bin"))
    val hann = StepLog.floatsFromStream(assets.open("hann400.bin"))
    val silence = StepLog.floatsFromStream(assets.open("silence_embeds.bin"))
    val env = Environment.create()
    try {
      val engine = LiteRtEngine(env, File(ctx.filesDir, "models"), encoderFile = encoder, precision = precision)
      engine.use {
        timing.put("frontend_compile_ms", it.frontendCompileMs).put("encoder_compile_ms", it.encoderCompileMs)
        status("compiled A %.0f ms, B %.0f ms".format(it.frontendCompileMs, it.encoderCompileMs))
        val warm = JSONArray()
        repeat(warmup) { _ ->
          val t0 = System.nanoTime()
          it.frontend(FloatArray(Nemotron3Diarizer.FRONTEND_FRAMES * MelFrontend.N_MELS))
          val t1 = System.nanoTime()
          val t = config.maxRows
          val (c, s) = Nemotron3Diarizer.ropeTables(t)
          it.encoder(FloatArray(t * Nemotron3Diarizer.HIDDEN), FloatArray(t) { r -> if (r < 13) 0f else config.padBias },
            c, s)
          val t2 = System.nanoTime()
          warm.put(JSONObject().put("frontend_ms", (t1 - t0) / 1e6).put("encoder_ms", (t2 - t1) / 1e6))
        }
        timing.put("warmup", warm)
        if (fileMode) {
          runFileMode(it, fb, hann, silence, wav.samples, timing)
          return@use
        }
        val diarizer = Nemotron3Diarizer(it, MelFrontend(fb, hann), silence, StreamConfig.LOW_LATENCY)
        val stepJson = mutableListOf<String>()
        val totals = mutableListOf<Double>()
        val parts = mapOf("mel" to mutableListOf<Double>(), "frontend" to mutableListOf(), "encoder" to mutableListOf(),
          "cache" to mutableListOf(), "total" to mutableListOf())
        val latency = JSONArray()
        var wall0 = 0L
        fun take(steps: List<Step>, rowIds: IntArray, arrivedNs: Long) {
          check(steps.size <= 1) { "one push ran ${steps.size} steps" }
          for (s in steps) {
            val endNs = System.nanoTime()
            latency.put(JSONObject().put("k", s.index).put("arrival_ms", (arrivedNs - wall0) / 1e6)
              .put("end_ms", (endNs - wall0) / 1e6).put("latency_ms", (endNs - arrivedNs) / 1e6))
            StepLog.writeFloats(File(outDir, "out_rows_step${s.index}.bin"), s.logits)
            stepJson += StepLog.stepJson(s, rowIds)
            parts.getValue("mel") += s.melMs
            parts.getValue("frontend") += s.frontendMs
            parts.getValue("encoder") += s.encoderMs
            parts.getValue("cache") += s.cacheMs
            parts.getValue("total") += s.totalMs
            totals += s.totalMs
            if (s.index % 20 == 0) {
              status("step ${s.index} L=${s.length} %.1f ms (mel %.1f A %.2f B %.1f cache %.2f)".format(
                s.totalMs, s.melMs, s.frontendMs, s.encoderMs, s.cacheMs))
            }
          }
        }
        wall0 = System.nanoTime()
        timing.put("wall0_epoch_ms", System.currentTimeMillis()) // aligns step_latency with device-side logs
        var pos = 0
        while (pos < wav.samples.size) {
          val n = minOf(push, wav.samples.size - pos)
          if (realtime) { // the push arrives when its last sample would have been recorded
            val due = wall0 + ((pos + n) * 1e9 / MelFrontend.SAMPLE_RATE).toLong()
            val wait = due - System.nanoTime()
            if (wait > 0) Thread.sleep(wait / 1_000_000, (wait % 1_000_000).toInt())
          }
          val arrived = System.nanoTime()
          val ids = diarizer.cache.rowIds()
          take(diarizer.push(wav.samples, pos, n), ids, arrived)
          pos += n
        }
        val ids = diarizer.cache.rowIds()
        take(diarizer.finish(), ids, System.nanoTime())
        val wallMs = (System.nanoTime() - wall0) / 1e6
        val seconds = wav.samples.size.toDouble() / MelFrontend.SAMPLE_RATE
        val meta =
          JSONObject()
            .put("wav", wavFile.name)
            .put("samples", wav.samples.size)
            .put("push", push)
            .put("config", diarizer.config.name)
            .put("T", diarizer.config.maxRows)
            .put("b_precision", precision.label)
        File(outDir, "steps.json").writeText(StepLog.runJson(meta.toString(), stepJson, diarizer.cache.compressions))
        val med = JSONObject()
        for ((k, v) in parts) med.put(k, median(v))
        timing
          .put("steps", totals.size)
          .put("audio_seconds", seconds)
          .put("step_ms_median", med)
          .put("step_ms_max", JSONObject().put("total", totals.maxOrNull() ?: Double.NaN))
          .put("sum_step_ms", totals.sum())
          .put("rtf_steps", totals.sum() / 1000.0 / seconds)
          .put("wall_ms", wallMs)
          .put("rtf_wall", wallMs / 1000.0 / seconds)
          .put("first_chunk_ms", totals.firstOrNull() ?: Double.NaN)
          .put("first_chunk_audio_ms", 1000.0 * (Nemotron3Diarizer.FRONTEND_FRAMES - 1) * MelFrontend.HOP / MelFrontend.SAMPLE_RATE +
            1000.0 * MelFrontend.WIN_LENGTH / 2 / MelFrontend.SAMPLE_RATE)
          .put("compressions", diarizer.cache.compressions.size)
          .put("step_latency", latency)
          .put("conditions_end", conditions())
        File(outDir, "timing.json").writeText(timing.toString(1))
        status("steps ${totals.size}, median %.1f ms, RTF %.3f, first chunk %.1f ms".format(
          med.getDouble("total"), totals.sum() / 1000.0 / seconds, totals.firstOrNull() ?: Double.NaN))
      }
    } finally {
      env.close()
    }
    status("DONE")
  }

  /** The offline pass over the whole wav, [repeats] times back to back; writes the last pass. */
  private fun runFileMode(
    engine: LiteRtEngine,
    fb: FloatArray,
    hann: FloatArray,
    silence: FloatArray,
    samples: FloatArray,
    timing: JSONObject,
  ) {
    val seconds = samples.size.toDouble() / MelFrontend.SAMPLE_RATE
    val passes = JSONArray()
    var steps = emptyList<Step>()
    repeat(repeats) { r ->
      val diarizer = Nemotron3Diarizer(engine, MelFrontend(fb, hann), silence, config)
      val startEpochMs = System.currentTimeMillis()
      val t0 = System.nanoTime()
      steps = diarizer.runFile(samples)
      val wallMs = (System.nanoTime() - t0) / 1e6
      val first = steps.first()
      passes.put(JSONObject().put("pass", r).put("start_epoch_ms", startEpochMs).put("wall_ms", wallMs)
        .put("rtf", wallMs / 1000.0 / seconds)
        .put("mel_ms", first.melMs).put("frontend_ms", first.frontendMs)
        .put("encoder_ms", JSONArray(steps.map { it.encoderMs }))
        .put("cache_ms", JSONArray(steps.map { it.cacheMs }))
        .put("end_ms_since_start", (System.nanoTime() - t0) / 1e6))
      status("pass $r: ${steps.size} chunks, wall %.1f ms, RTF %.4f (mel %.1f, A %.1f, B %s ms)".format(wallMs,
        wallMs / 1000.0 / seconds, first.melMs, first.frontendMs, steps.joinToString("/") { "%.0f".format(it.encoderMs) }))
    }
    val stepJson = steps.map { StepLog.stepJson(it, IntArray(0)) }
    for (s in steps) StepLog.writeFloats(File(outDir, "out_rows_step${s.index}.bin"), s.logits)
    val meta = JSONObject().put("wav", wavFile.name).put("samples", samples.size).put("config", config.name)
      .put("T", config.maxRows).put("b_precision", precision.label).put("file_mode", true)
    File(outDir, "steps.json").writeText(StepLog.runJson(meta.toString(), stepJson, emptyList()))
    timing
      .put("audio_seconds", seconds)
      .put("steps", steps.size)
      .put("passes", passes)
      .put("conditions_end", conditions())
    File(outDir, "timing.json").writeText(timing.toString(1))
  }

  companion object {
    const val TAG = "N3D"
  }
}

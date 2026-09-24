package com.nemotron3diar

import android.app.KeyguardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/** One graph-B variant: a model file under files/models and the requested GPU precision. */
data class Variant(val name: String, val file: String, val precision: String)

/**
 * The self-test request (files/selftest.json), e.g.
 * {"tag": "r2_safe_fp16", "frontend": "n3d_frontend.tflite", "steps": [0, 30, 128],
 *  "timing_step": 128, "warmup": 5, "iters": 20,
 *  "variants": [{"name": "safe_fp16", "file": "n3d_encoder_ll_safe_fp16.tflite", "precision": "default"}]}
 */
data class Spec(
  val tag: String,
  val frontend: String,
  val frontendPrecision: String,
  val steps: List<Int>,
  val timingStep: Int,
  val warmup: Int,
  val iters: Int,
  val variants: List<Variant>,
  /** graph B's fixed T (541 low_latency, 684 offline) and the step-bin folder under files/. */
  val t: Int = 541,
  val stepsDir: String = "steps",
  val skipFrontend: Boolean = false,
) {
  companion object {
    fun parse(text: String): Spec {
      val o = JSONObject(text)
      val steps = o.getJSONArray("steps").let { a -> List(a.length()) { a.getInt(it) } }
      val variants =
        o.getJSONArray("variants").let { a ->
          List(a.length()) {
            val v = a.getJSONObject(it)
            Variant(v.getString("name"), v.getString("file"), v.optString("precision", "default"))
          }
        }
      return Spec(
        tag = o.optString("tag", "selftest"),
        frontend = o.optString("frontend", "n3d_frontend.tflite"),
        frontendPrecision = o.optString("frontend_precision", variants.firstOrNull()?.precision ?: "default"),
        steps = steps,
        timingStep = o.optInt("timing_step", steps.first()),
        warmup = o.optInt("warmup", 5),
        iters = o.optInt("iters", 20),
        variants = variants,
        t = o.optInt("T", 541),
        stepsDir = o.optString("steps_dir", "steps"),
        skipFrontend = o.optBoolean("skip_frontend", false),
      )
    }
  }
}

/**
 * Runs the Nemotron-3-Diarization graphs on the LiteRT CompiledModel GPU for the round-2 device gate.
 *
 * graph A  mel [1,104,128] -> chunk_embeds [1,13,512]
 * graph B  packed_embeds [1,541,512], attn_bias [1,1,1,541], rope_cos / rope_sin [1,1,541,64] -> logits [1,4328,8]
 *
 * Inputs are float32 little-endian files in files/steps (step<i>_{mel,packed,bias,cos,sin}.bin); outputs go to
 * getExternalFilesDir(null)/n3d: outA_step<i>.bin, out_<variant>_<precision>_step<i>.bin, timing.json and
 * status.txt. One Environment is shared by every CompiledModel. GPU only: a graph that does not compile on the
 * GPU is recorded as an error, never retried on the CPU. Each timed iteration = write all inputs -> run() ->
 * readFloat() of every output (run() is asynchronous; the readback is the sync point).
 */
class SelfTest(private val ctx: Context, private val spec: Spec, private val show: (String) -> Unit) {

  private val stepsDir = File(ctx.filesDir, spec.stepsDir)
  private val t = spec.t
  private val modelsDir = File(ctx.filesDir, "models")
  private val outDir = File(ctx.getExternalFilesDir(null), "n3d")
  private val statusFile = File(outDir, "status.txt")
  private val power = ctx.getSystemService(PowerManager::class.java)
  private val errors = JSONArray()

  private fun status(line: String) {
    Log.i(TAG, line)
    statusFile.appendText(line + "\n")
    show(line)
  }

  private fun thermal(): Int = if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else -1

  /** Screen / lock / power state: the measurement conditions recorded next to the numbers. */
  private fun conditions(): JSONObject {
    val battery = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val plugged = battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1
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
      .put("thermal_status", thermal())
      .put("thermal_headroom", headroom())
  }

  private fun headroom(): Any {
    if (Build.VERSION.SDK_INT < 30) return JSONObject.NULL
    val h = power.getThermalHeadroom(0)
    return if (h.isNaN()) JSONObject.NULL else h.toDouble()
  }

  private fun options(precision: String): CompiledModel.Options {
    val options = CompiledModel.Options(Accelerator.GPU)
    when (precision) {
      "default" -> {}
      "fp32" -> options.gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
      "fp16" -> options.gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)
      "fp16acc32" ->
        options.gpuOptions =
          CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM)
      else -> error("unknown precision $precision")
    }
    return options
  }

  /** A compiled graph with its named I/O buffers. */
  private class Graph(val model: CompiledModel, inputs: List<String>, outputs: List<String>) : AutoCloseable {
    val ins: Map<String, TensorBuffer> = inputs.associateWith { model.createInputBuffer(it) }
    val outs: Map<String, TensorBuffer> = outputs.associateWith { model.createOutputBuffer(it) }

    /** write -> run -> readback of every output; returns the outputs by name. */
    fun invoke(feeds: Map<String, FloatArray>): Map<String, FloatArray> {
      for ((name, data) in feeds) ins.getValue(name).writeFloat(data)
      model.run(ins, outs)
      return outs.mapValues { it.value.readFloat() }
    }

    override fun close() {
      ins.values.forEach { it.close() }
      outs.values.forEach { it.close() }
      model.close()
    }
  }

  private fun compile(env: Environment, file: String, precision: String, inputs: List<String>, outputs: List<String>):
    Pair<Graph, Double> {
    val path = File(modelsDir, file)
    check(path.exists()) { "model not found: ${path.absolutePath} (run scripts/install_to_device.sh)" }
    val t0 = System.nanoTime()
    val model = CompiledModel.create(path.absolutePath, options(precision), env)
    val graph = Graph(model, inputs, outputs)
    return graph to (System.nanoTime() - t0) / 1e6
  }

  private fun stepInput(step: Int, kind: String, count: Int): FloatArray {
    val f = File(stepsDir, "step${step}_$kind.bin")
    val bytes = f.readBytes()
    check(bytes.size == count * 4) { "${f.name}: ${bytes.size} bytes, expected ${count * 4}" }
    val out = FloatArray(count)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
  }

  private fun writeOut(name: String, data: FloatArray) {
    val bb = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(data)
    File(outDir, name).writeBytes(bb.array())
  }

  private fun feedsA(step: Int) = mapOf("mel" to stepInput(step, "mel", MEL_FRAMES * MEL_BINS))

  private fun feedsB(step: Int) =
    mapOf(
      "packed_embeds" to stepInput(step, "packed", t * HIDDEN),
      "attn_bias" to stepInput(step, "bias", t),
      "rope_cos" to stepInput(step, "cos", t * HEAD_DIM),
      "rope_sin" to stepInput(step, "sin", t * HEAD_DIM),
    )

  /** warmup + timed iterations of write -> run -> readback; fills [rec] with the timing fields. */
  private fun time(graph: Graph, feeds: Map<String, FloatArray>, rec: JSONObject) {
    rec.put("thermal_before", thermal())
    rec.put("headroom_before", headroom())
    repeat(spec.warmup) { graph.invoke(feeds) }
    val ms = DoubleArray(spec.iters)
    for (i in 0 until spec.iters) {
      val t0 = System.nanoTime()
      graph.invoke(feeds)
      ms[i] = (System.nanoTime() - t0) / 1e6
    }
    val sorted = ms.sorted()
    val median = if (sorted.size % 2 == 1) sorted[sorted.size / 2]
    else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2
    rec.put("warmup", spec.warmup)
    rec.put("iters", spec.iters)
    rec.put("median_ms", median)
    rec.put("min_ms", sorted.first())
    rec.put("max_ms", sorted.last())
    rec.put("times_ms", JSONArray(ms.toList()))
    rec.put("thermal_after", thermal())
    rec.put("headroom_after", headroom())
  }

  private fun recordError(where: String, t: Throwable) {
    Log.e(TAG, "ERROR $where", t)
    errors.put(JSONObject().put("where", where).put("error", "${t.javaClass.name}: ${t.message}"))
    status("ERROR $where: ${t.javaClass.simpleName}: ${t.message}")
  }

  fun run() {
    outDir.mkdirs()
    outDir.listFiles()?.forEach { it.delete() }
    val report =
      JSONObject()
        .put("tag", spec.tag)
        .put("litert_version", BuildConfig.LITERT_VERSION)
        .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
        .put("soc", if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else "")
        .put("android_sdk", Build.VERSION.SDK_INT)
        .put("steps", JSONArray(spec.steps))
        .put("T", t)
        .put("steps_dir", spec.stepsDir)
        .put("timing_step", spec.timingStep)
        .put("conditions_start", conditions())
    val graphs = JSONArray()
    status("BEGIN ${spec.tag} litert=${BuildConfig.LITERT_VERSION} steps=${spec.steps}")
    val env = Environment.create()
    try {
      // graph A
      val recA = JSONObject().put("graph", "A").put("file", spec.frontend).put("precision", spec.frontendPrecision)
      if (spec.skipFrontend) recA.put("skipped", true)
      else try {
        status("A compile ${spec.frontend} precision=${spec.frontendPrecision}")
        val (a, ms) = compile(env, spec.frontend, spec.frontendPrecision, listOf("mel"), listOf("chunk_embeds"))
        a.use {
          recA.put("load_compile_ms", ms)
          status("A compiled in %.1f ms".format(ms))
          for (s in spec.steps) {
            val out = it.invoke(feedsA(s)).getValue("chunk_embeds")
            check(out.size == CHUNK_ROWS * HIDDEN) { "chunk_embeds size ${out.size}" }
            writeOut("outA_step$s.bin", out)
          }
          status("A parity outputs written (${spec.steps.size} steps)")
          time(it, feedsA(spec.timingStep), recA)
          status("A median %.3f ms".format(recA.getDouble("median_ms")))
        }
      } catch (t: Throwable) {
        recA.put("error", "${t.javaClass.name}: ${t.message}")
        recordError("A", t)
      }
      graphs.put(recA)

      // graph B variants
      for (v in spec.variants) {
        val recB =
          JSONObject().put("graph", "B").put("variant", v.name).put("file", v.file).put("precision", v.precision)
        try {
          status("B ${v.name} compile ${v.file} precision=${v.precision}")
          val (b, ms) =
            compile(env, v.file, v.precision, listOf("packed_embeds", "attn_bias", "rope_cos", "rope_sin"),
              listOf("logits"))
          b.use {
            recB.put("load_compile_ms", ms)
            status("B ${v.name} compiled in %.1f ms".format(ms))
            for (s in spec.steps) {
              val out = it.invoke(feedsB(s)).getValue("logits")
              check(out.size == t * STACK * SPEAKERS) { "logits size ${out.size}" }
              writeOut("out_${v.name}_${v.precision}_step$s.bin", out)
            }
            status("B ${v.name} parity outputs written (${spec.steps.size} steps)")
            time(it, feedsB(spec.timingStep), recB)
            status("B ${v.name} median %.3f ms".format(recB.getDouble("median_ms")))
          }
        } catch (t: Throwable) {
          recB.put("error", "${t.javaClass.name}: ${t.message}")
          recordError("B ${v.name}", t)
        }
        graphs.put(recB)
      }
    } finally {
      env.close()
    }
    report.put("graphs", graphs).put("errors", errors).put("conditions_end", conditions())
    File(outDir, "timing.json").writeText(report.toString(1))
    status("DONE errors=${errors.length()}")
  }

  companion object {
    const val TAG = "N3D"
    const val HIDDEN = 512
    const val HEAD_DIM = 64
    const val STACK = 8
    const val SPEAKERS = 8
    const val MEL_FRAMES = 104
    const val MEL_BINS = 128
    const val CHUNK_ROWS = 13
  }
}

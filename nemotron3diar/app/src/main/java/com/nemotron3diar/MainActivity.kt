package com.nemotron3diar

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.Choreographer
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import com.google.ai.edge.litert.Environment
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

/**
 * Streaming speaker diarization ("who spoke when", up to 8 speakers) with Nemotron-3-Diarization on the LiteRT
 * CompiledModel GPU: record from the microphone or pick a clip, and the timeline grows every 0.72 s step (1.04 s
 * first-chunk latency, low_latency mode). graph B's GPU precision is selectable (graph A always runs FP32).
 *
 * "Play live" plays a clip through the speaker while the same samples stream into the model at the audio's own pace
 * (as a microphone next to the speaker would hear them): the full-screen [LiveView] shows the playback clock, the
 * waveform colored by speaker and the eight speaker lanes growing about one step behind the audio.
 *
 * Gates: if files/selftest.json exists it is consumed (renamed to selftest.done, so a relaunch from recents never
 * repeats it) and run instead of the demo -- {"mode": "closed_loop", ...} runs [ClosedLoopTest], the round-2 form
 * with "variants" runs [SelfTest]. For scripted screenshots, `am start -n com.nemotron3diar/.MainActivity
 * --es clip <file under files/>` runs the Pick path on that file and `--ei record_seconds N` records N seconds;
 * `--es live <file under files/>` runs Play live on it (optional `--ei push <samples>` (1600), `--ei preroll_ms <ms>`
 * (1500) of the ready screen before playback) and writes the step log to getExternalFilesDir/n3d/live_<name>/.
 * None of these extras runs again when the task is resumed from recents.
 */
class MainActivity : Activity() {

  private val worker = Executors.newSingleThreadExecutor()
  private var env: Environment? = null
  private var engine: LiteRtEngine? = null
  private lateinit var tables: Triple<FloatArray, FloatArray, FloatArray>

  private lateinit var status: TextView
  private lateinit var stats: TextView
  private lateinit var summary: TextView
  private lateinit var timeline: TimelineView
  private lateinit var recordButton: Button
  private lateinit var pickButton: Button
  private lateinit var liveButton: Button
  private lateinit var precisionGroup: RadioGroup
  private lateinit var normalRoot: View
  private var live: LiveView? = null

  @Volatile private var liveRunning = false

  @Volatile private var recording = false
  @Volatile private var busy = false

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    buildUi()
    if (runSelfTestIfRequested()) return

    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_MIC)
    }
    // a task resumed from recents re-delivers the launch intent: never auto-run its extras again
    val fresh = savedInstanceState == null && intent.flags and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY == 0
    val clip = if (fresh) intent.getStringExtra("clip") else null
    val recordSeconds = if (fresh) intent.getIntExtra("record_seconds", 0) else 0
    val liveClip = if (fresh) intent.getStringExtra("live") else null
    val livePush = intent.getIntExtra("push", PUSH)
    val preroll = intent.getIntExtra("preroll_ms", 1500)
    worker.execute {
      try {
        tables = loadTables()
        val e = Environment.create()
        env = e
        val eng = LiteRtEngine(e, File(filesDir, "models"), precision = LiteRtEngine.Precision.FP32)
        engine = eng
        warmUp(eng)
        ui {
          status.text =
            "Ready · GPU compile A %.0f ms, B %.0f ms (FP32)".format(eng.frontendCompileMs, eng.encoderCompileMs)
          setControlsEnabled(true)
        }
      } catch (t: Throwable) {
        Log.e(TAG, "load", t)
        ui {
          status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2))
          status.text = "FAIL: ${t.message}"
        }
        return@execute
      }
      if (clip != null) runOnUiThread { decodeAndRun(Uri.fromFile(File(filesDir, clip))) }
      else if (recordSeconds > 0) runOnUiThread { startRecording(recordSeconds) }
      else if (liveClip != null) {
        try {
          val x = readClip(File(filesDir, liveClip))
          runOnUiThread { startLive(x, File(liveClip).nameWithoutExtension, livePush, preroll) }
        } catch (t: Throwable) {
          Log.e(TAG, "live", t)
          ui { status.text = "FAIL: ${t.message}" }
        }
      }
    }
  }

  // ------------------------------------------------------------------------------------------ UI

  private fun buildUi() {
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(36, 96, 36, 36)
    }
    val title = TextView(this).apply {
      text = "Nemotron-3-Diarization · LiteRT GPU"
      textSize = 20f
      typeface = Typeface.DEFAULT_BOLD
    }
    status = TextView(this).apply {
      textSize = 14f
      text = "Loading models…"
      setPadding(0, 16, 0, 16)
    }
    precisionGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
    for ((i, p) in LiteRtEngine.Precision.entries.withIndex()) {
      precisionGroup.addView(RadioButton(this).apply {
        id = 100 + i
        text = when (p) {
          LiteRtEngine.Precision.FP32 -> "FP32"
          LiteRtEngine.Precision.DEFAULT -> "FP16"
          LiteRtEngine.Precision.ACCUM -> "FP16+FP32 acc"
        }
        textSize = 13f
      })
    }
    precisionGroup.check(100)
    precisionGroup.setOnCheckedChangeListener { _, id -> setPrecision(LiteRtEngine.Precision.entries[id - 100]) }
    recordButton = Button(this).apply {
      text = "Record"
      setOnClickListener { if (recording) recording = false else startRecording(MAX_SECONDS) }
    }
    pickButton = Button(this).apply {
      text = "Pick clip"
      setOnClickListener {
        startActivityForResult(
          Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/*"))
          },
          REQUEST_PICK,
        )
      }
    }
    liveButton = Button(this).apply {
      text = "Play live"
      setOnClickListener {
        startActivityForResult(
          Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/*", "video/*"))
          },
          REQUEST_LIVE,
        )
      }
    }
    val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
    buttons.addView(recordButton)
    buttons.addView(pickButton)
    buttons.addView(liveButton)
    stats = TextView(this).apply {
      textSize = 12f
      typeface = Typeface.MONOSPACE
      setPadding(0, 16, 0, 16)
    }
    timeline = TimelineView(this)
    summary = TextView(this).apply {
      textSize = 14f
      setPadding(0, 16, 0, 0)
    }
    root.addView(title)
    root.addView(status)
    root.addView(precisionGroup)
    root.addView(buttons)
    root.addView(stats)
    root.addView(timeline)
    root.addView(summary)
    normalRoot = ScrollView(this).apply { addView(root) }
    setContentView(normalRoot)
    setControlsEnabled(false)
  }

  private fun ui(block: () -> Unit) = runOnUiThread(block)

  private fun setControlsEnabled(enabled: Boolean) {
    pickButton.isEnabled = enabled
    liveButton.isEnabled = enabled
    recordButton.isEnabled = enabled || recording
    for (i in 0 until precisionGroup.childCount) precisionGroup.getChildAt(i).isEnabled = enabled
  }

  // ------------------------------------------------------------------------------------------ model

  private fun loadTables() =
    Triple(
      StepLog.floatsFromStream(assets.open("frontend_mel128_257.bin")),
      StepLog.floatsFromStream(assets.open("hann400.bin")),
      StepLog.floatsFromStream(assets.open("silence_embeds.bin")),
    )

  /** One inference of each graph on zeros, so the first chunk runs warm. */
  private fun warmUp(eng: LiteRtEngine) {
    eng.frontend(FloatArray(Nemotron3Diarizer.FRONTEND_FRAMES * MelFrontend.N_MELS))
    val t = StreamConfig.LOW_LATENCY.maxRows
    val (c, s) = Nemotron3Diarizer.ropeTables(t)
    eng.encoder(FloatArray(t * Nemotron3Diarizer.HIDDEN), FloatArray(t) { if (it < 13) 0f else -3.0e4f }, c, s)
  }

  private fun newDiarizer(): Nemotron3Diarizer {
    val (fb, hann, silence) = tables
    return Nemotron3Diarizer(engine!!, MelFrontend(fb, hann), silence, StreamConfig.LOW_LATENCY)
  }

  private fun setPrecision(p: LiteRtEngine.Precision) {
    val eng = engine ?: return
    if (busy || p == eng.precision) return
    busy = true
    setControlsEnabled(false)
    status.text = "Compiling graph B (${p.label})…"
    worker.execute {
      try {
        eng.setPrecision(p)
        warmUp(eng)
        ui { status.text = "Ready · graph B %s compiled in %.0f ms".format(p.label, eng.encoderCompileMs) }
      } catch (t: Throwable) {
        Log.e(TAG, "precision", t)
        ui { status.text = "FAIL: ${t.message}" }
      } finally {
        busy = false
        ui { setControlsEnabled(true) }
      }
    }
  }

  // ------------------------------------------------------------------------------------------ streaming

  /** Sums of the steps of the current session, for the stats line. */
  private class Session(val source: String) {
    var steps = 0
    var ms = 0.0
    var lastMs = 0.0
    var firstMs = 0.0
    var samples = 0
  }

  /** Starts a session; callable from any thread (the view updates are posted to the UI thread). */
  private fun begin(source: String): Session {
    ui {
      timeline.reset()
      summary.text = ""
      stats.text = ""
      status.text = source
    }
    return Session(source)
  }

  /** Posts the steps a push produced: timeline, per-step ms, real-time factor. */
  private fun publish(session: Session, steps: List<Step>, samplesSoFar: Int) {
    if (steps.isEmpty()) return
    for (s in steps) {
      if (session.steps == 0) session.firstMs = s.totalMs
      session.steps++
      session.ms += s.totalMs
      session.lastMs = s.totalMs
    }
    session.samples = samplesSoFar
    val last = steps.last()
    val copies = steps.map { it.logits.copyOf() to it.numFrames }
    val audioS = samplesSoFar.toDouble() / MelFrontend.SAMPLE_RATE
    val line =
      "step %d · %.0f ms (mel %.1f · A %.1f · B %.0f · cache %.1f)\nRTF %.2f · first chunk %.0f ms · %.1f s audio".format(
        last.index, last.totalMs, last.melMs, last.frontendMs, last.encoderMs, last.cacheMs,
        session.ms / 1000.0 / audioS, session.firstMs, audioS)
    ui {
      for ((l, n) in copies) timeline.append(l, n)
      stats.text = line
      status.text = "${session.source} · ${timeline.speakers().size} speaker(s)"
    }
  }

  private fun finishSession(session: Session, d: Nemotron3Diarizer) {
    publish(session, d.finish(), session.samples)
    ui {
      val secs = timeline.speakerSeconds()
      summary.text = timeline.speakers().joinToString("\n") { "SPK ${it + 1}: %.1f s".format(secs[it]) }
      status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
      status.text = "✓ ${session.source} · ${timeline.speakers().size} speaker(s) · ${session.steps} steps · " +
        "graph B ${engine?.precision?.label}"
      setControlsEnabled(true)
    }
  }

  private fun startRecording(seconds: Int) {
    if (engine == null || busy) return
    if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      status.text = "Microphone permission needed."
      return
    }
    busy = true
    recording = true
    recordButton.text = "Stop"
    setControlsEnabled(false)
    status.setBackgroundColor(Color.TRANSPARENT)
    val session = begin("● Recording")
    worker.execute {
      val sr = MelFrontend.SAMPLE_RATE
      val min = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
      val rec = AudioRecord(MediaRecorder.AudioSource.MIC, sr, AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, sr * 4 * 4))
      val d = newDiarizer()
      val limit = seconds * sr
      var total = 0
      try {
        rec.startRecording()
        val buf = FloatArray(PUSH)
        while (recording && total < limit) {
          val r = rec.read(buf, 0, minOf(buf.size, limit - total), AudioRecord.READ_BLOCKING)
          if (r <= 0) continue
          total += r
          publish(session, d.push(buf, 0, r), total)
        }
      } catch (t: Throwable) {
        Log.e(TAG, "record", t)
        ui { status.text = "FAIL: ${t.message}" }
      } finally {
        rec.stop()
        rec.release()
        recording = false
      }
      session.samples = total
      ui { recordButton.text = "Record" }
      finishSession(session, d)
      busy = false
    }
  }

  @Deprecated("Deprecated in Java")
  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    val uri = data?.data ?: return
    if (resultCode != RESULT_OK) return
    if (requestCode == REQUEST_PICK) decodeAndRun(uri)
    if (requestCode == REQUEST_LIVE && engine != null && !busy) {
      status.text = "Decoding…"
      worker.execute {
        try {
          val x = AudioDecoder.decode(this, uri, MAX_SECONDS)
          ui { startLive(x, uri.lastPathSegment?.substringAfterLast('/')?.substringBeforeLast('.') ?: "clip", PUSH, 500) }
        } catch (t: Throwable) {
          Log.e(TAG, "live", t)
          ui { status.text = "FAIL: ${t.message}" }
        }
      }
    }
  }

  /** Pick path: decode to 16 kHz mono, then the same streaming loop as the microphone, as fast as it runs. */
  private fun decodeAndRun(uri: Uri) {
    if (engine == null || busy) return
    busy = true
    setControlsEnabled(false)
    status.setBackgroundColor(Color.TRANSPARENT)
    status.text = "Decoding…"
    worker.execute {
      try {
        val x = AudioDecoder.decode(this, uri, MAX_SECONDS)
        val session = begin("Clip ${uri.lastPathSegment ?: ""}")
        val d = newDiarizer()
        var pos = 0
        while (pos < x.size) {
          val n = minOf(PUSH, x.size - pos)
          pos += n
          publish(session, d.push(x, pos - n, n), pos)
        }
        session.samples = x.size
        finishSession(session, d)
      } catch (t: Throwable) {
        Log.e(TAG, "clip", t)
        ui {
          status.text = "FAIL: ${t.message}"
          setControlsEnabled(true)
        }
      } finally {
        busy = false
      }
    }
  }

  // ------------------------------------------------------------------------------------------ play live

  /** A clip under files/: .wav as 16 kHz PCM16 / float read exactly, anything else (or another rate) decoded. */
  private fun readClip(file: File): FloatArray {
    check(file.exists()) { "not found: ${file.absolutePath}" }
    if (file.extension.equals("wav", ignoreCase = true)) {
      val wav = WavReader.read(file)
      if (wav.sampleRate == MelFrontend.SAMPLE_RATE) return wav.samples
    }
    return AudioDecoder.decode(this, Uri.fromFile(file), MAX_SECONDS)
  }

  /** Shows the live screen and starts playback + streaming after [prerollMs] of the ready screen. */
  private fun startLive(x: FloatArray, name: String, push: Int, prerollMs: Int) {
    val eng = engine ?: return
    if (busy || x.isEmpty()) return
    busy = true
    setControlsEnabled(false)
    val view = LiveView(this)
    view.setClip(x, MelFrontend.SAMPLE_RATE)
    view.latencyText = "latency %.2f s + on-device \u2026".format(DESIGN_LATENCY_S)
    view.stepText = "graph B ${eng.precision.label.uppercase()} \u00B7 LiteRT GPU"
    live = view
    setContentView(view)
    setFullScreen(true)
    liveRunning = true
    worker.execute { runLive(view, x, name, push, prerollMs) }
  }

  /** One step of the live loop: when its chunk arrived (samples heard) and when the UI showed it. */
  private class LiveEntry(val step: Step, val arrival: Long, val arrivalNs: Long) {
    @Volatile var shown = -1L
    @Volatile var shownNs = 0L
  }

  private fun runLive(view: LiveView, x: FloatArray, name: String, push: Int, prerollMs: Int) {
    val outDir = File(File(getExternalFilesDir(null), "n3d"), "live_$name")
    outDir.mkdirs()
    outDir.listFiles()?.forEach { it.delete() }
    val entries = mutableListOf<LiveEntry>()
    val logits = java.io.ByteArrayOutputStream()
    val totals = mutableListOf<Double>()
    val precision = engine?.precision?.label ?: "?"
    val conditionsStart = conditions()
    var player: LivePlayer? = null
    var startNs = 0L
    try {
      Thread.sleep(prerollMs.toLong())
      val d = newDiarizer()
      val p = LivePlayer(x, MelFrontend.SAMPLE_RATE)
      player = p
      startNs = System.nanoTime()
      p.start()
      ui {
        view.state = LiveView.State.LIVE
        tick(view, p)
      }
      fun publish(steps: List<Step>, arrival: Long) {
        for (s in steps) {
          val e = LiveEntry(s, arrival, System.nanoTime())
          entries += e
          totals += s.totalMs
          val bb = java.nio.ByteBuffer.allocate(s.numFrames * NUM_SPEAKERS * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
          bb.asFloatBuffer().put(s.logits, 0, s.numFrames * NUM_SPEAKERS)
          logits.write(bb.array())
          val copy = s.logits.copyOf()
          val latency = "latency %.2f s + on-device %.2f s".format(DESIGN_LATENCY_S, median(totals) / 1000)
          val line = "step %d \u00B7 %.0f ms \u00B7 graph B %s \u00B7 LiteRT GPU".format(s.index, s.totalMs,
            precision.uppercase())
          ui {
            view.append(copy, s.numFrames)
            view.latencyText = latency
            view.stepText = line
            e.shown = p.position()
            e.shownNs = System.nanoTime()
          }
        }
      }
      var pushed = 0
      while (pushed < x.size && liveRunning) {
        val heard = p.position()
        if (heard - pushed >= push || heard >= x.size) {
          val n = minOf(push, x.size - pushed)
          val steps = d.push(x, pushed, n)
          pushed += n
          publish(steps, heard)
        } else {
          Thread.sleep(2)
        }
      }
      if (liveRunning) publish(d.finish(), p.position())
      while (liveRunning && p.position() < x.size) Thread.sleep(5)
      ui {
        view.setPlayhead(x.size.toLong())
        view.state = LiveView.State.DONE
      }
      Thread.sleep(100) // lets the last posted updates record their display position
    } catch (t: Throwable) {
      Log.e(TAG, "live", t)
      ui { view.stepText = "FAIL: ${t.message}" }
    } finally {
      liveRunning = false
      player?.close()
      busy = false
    }
    runCatching { writeLiveLog(outDir, name, x.size, push, precision, startNs, entries, totals, logits, conditionsStart) }
      .onFailure { Log.e(TAG, "live log", it) }
  }

  private fun tick(view: LiveView, p: LivePlayer) {
    Choreographer.getInstance().postFrameCallback(object : Choreographer.FrameCallback {
      override fun doFrame(frameTimeNanos: Long) {
        if (!liveRunning || live !== view) return
        view.setPlayhead(p.position())
        Choreographer.getInstance().postFrameCallback(this)
      }
    })
  }

  private fun writeLiveLog(
    dir: File,
    name: String,
    samples: Int,
    push: Int,
    precision: String,
    startNs: Long,
    entries: List<LiveEntry>,
    totals: List<Double>,
    logits: java.io.ByteArrayOutputStream,
    conditionsStart: JSONObject,
  ) {
    File(dir, "logits.bin").writeBytes(logits.toByteArray())
    val steps = org.json.JSONArray()
    for (e in entries) {
      val s = e.step
      steps.put(JSONObject()
        .put("k", s.index).put("g0", s.firstFrame).put("nout", s.numFrames).put("L", s.length)
        .put("arrival", e.arrival).put("shown", e.shown)
        .put("arrival_ms", (e.arrivalNs - startNs) / 1e6).put("shown_ms", (e.shownNs - startNs) / 1e6)
        .put("ms", JSONObject().put("mel", s.melMs).put("frontend", s.frontendMs).put("encoder", s.encoderMs)
          .put("cache", s.cacheMs).put("total", s.totalMs)))
    }
    val sorted = totals.sorted()
    val json = JSONObject()
      .put("clip", name).put("samples", samples).put("sample_rate", MelFrontend.SAMPLE_RATE).put("push", push)
      .put("b_precision", precision).put("a_precision", "fp32").put("litert_version", BuildConfig.LITERT_VERSION)
      .put("design_latency_s", DESIGN_LATENCY_S)
      .put("steps_count", totals.size)
      .put("step_ms_median", median(totals))
      .put("step_ms_p95", if (sorted.isEmpty()) Double.NaN else sorted[((sorted.size - 1) * 0.95).toInt()])
      .put("step_ms_max", sorted.lastOrNull() ?: Double.NaN)
      .put("rtf", totals.sum() / 1000.0 / (samples.toDouble() / MelFrontend.SAMPLE_RATE))
      .put("conditions_start", conditionsStart)
      .put("conditions_end", conditions())
      .put("steps", steps)
    File(dir, "live.json").writeText(json.toString(1))
    Log.i(TAG, "live log: ${dir.absolutePath} (${entries.size} steps, median %.1f ms)".format(median(totals)))
  }

  private fun median(v: List<Double>): Double {
    if (v.isEmpty()) return Double.NaN
    val s = v.sorted()
    return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
  }

  /** Screen, keyguard, power source and thermal state, for the live log. */
  private fun conditions(): JSONObject {
    val power = getSystemService(android.os.PowerManager::class.java)
    val battery = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    return JSONObject()
      .put("screen_interactive", power.isInteractive)
      .put("keyguard_locked", getSystemService(android.app.KeyguardManager::class.java).isKeyguardLocked)
      .put("plugged", battery?.getIntExtra(android.os.BatteryManager.EXTRA_PLUGGED, -1) ?: -1)
      .put("battery_level", battery?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1)
      .put("battery_temp_c", (battery?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -10) ?: -10) / 10.0)
      .put("thermal_status", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else -1)
  }

  private fun setFullScreen(on: Boolean) {
    if (Build.VERSION.SDK_INT >= 30) {
      val c = window.insetsController ?: return
      if (on) {
        c.hide(WindowInsets.Type.systemBars())
        c.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
      } else {
        c.show(WindowInsets.Type.systemBars())
      }
    } else {
      @Suppress("DEPRECATION")
      window.decorView.systemUiVisibility =
        if (on) View.SYSTEM_UI_FLAG_FULLSCREEN or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
          View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        else 0
    }
  }

  /** Back from the live screen stops playback and returns to the controls. */
  @Deprecated("Deprecated in Java")
  override fun onBackPressed() {
    if (live == null) {
      @Suppress("DEPRECATION")
      super.onBackPressed()
      return
    }
    liveRunning = false
    live = null
    setFullScreen(false)
    setContentView(normalRoot)
    worker.execute { ui { setControlsEnabled(true) } } // after the live loop has left the worker
  }

  // ------------------------------------------------------------------------------------------ self-tests

  private fun runSelfTestIfRequested(): Boolean {
    if (started) {
      status.text = "self-test already started in this process; see status.txt"
      return true
    }
    val request = File(filesDir, "selftest.json")
    if (!request.exists()) return false
    val consumed = File(filesDir, "selftest.done")
    consumed.delete()
    check(request.renameTo(consumed)) { "could not consume ${request.absolutePath}" }
    started = true
    val text = consumed.readText()
    status.typeface = Typeface.MONOSPACE
    status.textSize = 11f
    status.text = ""
    val show: (String) -> Unit = { line -> ui { status.append(line + "\n") } }
    worker.execute {
      try {
        val json = JSONObject(text)
        if (json.optString("mode") == "closed_loop") ClosedLoopTest(applicationContext, json, show).run()
        else SelfTest(applicationContext, Spec.parse(text), show).run()
      } catch (t: Throwable) {
        Log.e(SelfTest.TAG, "self-test aborted", t)
        show("ABORTED ${t.javaClass.simpleName}: ${t.message}")
        runCatching {
          val dir = File(getExternalFilesDir(null), "n3d")
          dir.mkdirs()
          File(dir, "status.txt").appendText("FAILED ${t.javaClass.simpleName}: ${t.message}\n")
        }
      }
    }
    return true
  }

  override fun onDestroy() {
    super.onDestroy()
    recording = false
    liveRunning = false
    worker.execute {
      engine?.close()
      env?.close()
    }
    worker.shutdown()
  }

  companion object {
    private const val TAG = "N3D"
    private const val REQUEST_MIC = 1
    private const val REQUEST_PICK = 2
    private const val REQUEST_LIVE = 3
    private const val MAX_SECONDS = 300
    private const val PUSH = 1600
    private const val NUM_SPEAKERS = Nemotron3Diarizer.NUM_SPEAKERS

    /** First-frame latency of low_latency: (chunk 9 + look-ahead 4) encoder frames of 80 ms. */
    private val DESIGN_LATENCY_S =
      (StreamConfig.LOW_LATENCY.chunkFrames + StreamConfig.LOW_LATENCY.rightContext) *
        StreamConfig.SUBSAMPLING * MelFrontend.HOP / MelFrontend.SAMPLE_RATE.toDouble()

    @Volatile private var started = false
  }
}

package com.d1omni

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * The Kotlin audio host against the Python host (`host/d1_audio_host.py`) on the six public clips,
 * through the dumps of `demo/scripts/audio_dump.py` (`demo/fixtures/audio/`): the Hann window and the
 * Slaney filter bank, the waveform and the preemphasis bit for bit, every float64 stage of the mel
 * (power, lin, log, mean, std) against `mel_stages(x, "float64")`, the mel against both of the
 * host's forms (`mel_f64` and the provider-like `mel_f32`), frames / T / T_b / L1-3 / T1-3 / P, the
 * graph's mask inputs bit for bit, and the 18 audio rows encoded after the app's own P against the
 * fixture's ids and markers — also through the phone's rows file (`device/r2/rows_audio.json`) and
 * the gate's checks of it. The numbers go to `build/reports/parity/audio.json`.
 */
class D1AudioTest {
  @Test
  fun windowAndFilterbankMatchPython() {
    val window = D1Audio.hannWindowF32()
    val pythonWindow = floats(ExternalTestData.demoFile("fixtures/audio/hann_f32.f32"))
    val bank = D1Audio.slaneyFilterbank()
    val pythonBank = floats(ExternalTestData.demoFile("fixtures/audio/slaney_fb.f32"))
    assertEquals(D1Audio.WIN, pythonWindow.size)
    assertEquals(D1Audio.N_MELS * D1Audio.BINS, pythonBank.size)
    val windowUlps = ulpDifferences(window, pythonWindow)
    val bankUlps = ulpDifferences(bank, pythonBank)
    println("D1_AUDIO window: ${windowUlps.first} of 400 differ (max ${windowUlps.second} ulp); filter bank: ${bankUlps.first} of ${bank.size} differ (max ${bankUlps.second} ulp)")
    assertEquals("Hann window values that differ from Python", 0, windowUlps.first)
    assertEquals("filter bank values that differ from Python", 0, bankUlps.first)
  }

  @Test
  fun clipsMatchThePythonHost() {
    val contract = ExternalTestData.contract()
    val tokenizer = ExternalTestData.tokenizer()
    val doc = ExternalTestData.json(ExternalTestData.repoFile("fixtures/public_audio.json"))
    val report = LinkedHashMap<String, Any?>()
    var rows = 0
    var rowsMatched = 0
    var worstF64 = 0.0
    var worstF32 = 0.0
    var melMillis = 0.0
    for (record in doc["records"] as List<*>) {
      val entry = record as Map<*, *>
      val id = entry["id"] as String
      val dumps = "fixtures/audio/$id"
      val info = ExternalTestData.json(ExternalTestData.demoFile("$dumps/info.json"))
      val media = entry["media"] as Map<*, *>
      val samples = D1Wav.read(ExternalTestData.repoFile("fixtures/${media["file"]}"))
      val waveform = D1Audio.waveform(samples)
      assertArrayEquals("$id waveform", floats(ExternalTestData.demoFile("$dumps/waveform.f32")), waveform, 0f)
      assertBitEqual("$id waveform", floats(ExternalTestData.demoFile("$dumps/waveform.f32")), waveform)
      val started = System.nanoTime()
      val stages = D1Audio.melStages(waveform, keepPower = true)
      melMillis += (System.nanoTime() - started) / 1e6
      val stft = stages.stftFrames
      assertBitEqual("$id preemphasis", floats(ExternalTestData.demoFile("$dumps/stages_f64/pre.f32")), stages.pre)
      val powerRel =
        relativeToFrame(requireNotNull(stages.power), doubles(ExternalTestData.demoFile("$dumps/stages_f64/power.bin")), stft)
      val linRel = relativeToFrame(stages.lin, doubles(ExternalTestData.demoFile("$dumps/stages_f64/lin.bin")), stft)
      val logAbs = absolute(stages.log, doubles(ExternalTestData.demoFile("$dumps/stages_f64/log.bin")))
      val meanAbs = absolute(stages.mean, doubles(ExternalTestData.demoFile("$dumps/stages_f64/mean.bin")))
      val stdAbs = absolute(stages.std, doubles(ExternalTestData.demoFile("$dumps/stages_f64/std.bin")))
      val melF64 = floats(ExternalTestData.demoFile("$dumps/mel_f64.f32"))
      val melF32 = floats(ExternalTestData.demoFile("$dumps/mel_f32.f32"))
      val vsF64 = D1AudioCheck.melDifference(stages.out, melF64)
      val vsF32 = D1AudioCheck.melDifference(stages.out, melF32)
      // The float32 log step instead of the float64 one (the choice is recorded, not shipped).
      val log32 = D1Audio.melStages(waveform, logInFloat32 = true).out
      val log32VsF64 = D1AudioCheck.melDifference(log32, melF64)
      val log32VsF32 = D1AudioCheck.melDifference(log32, melF32)
      worstF64 = max(worstF64, vsF64["max_abs"] as Double)
      worstF32 = max(worstF32, vsF32["max_abs"] as Double)
      // Sizes: frames, T, the bucket, its dims, the clip's lengths and P.
      val frames = (info["frames"] as JsonNumber).toInt()
      val bucket = (info["T_b"] as JsonNumber).toInt()
      assertEquals(id, (info["n"] as JsonNumber).toInt(), waveform.size)
      assertEquals(id, frames, stages.frames)
      assertEquals(id, (info["T"] as JsonNumber).toInt(), stft)
      assertEquals(id, bucket, D1Audio.bucketFor(stft))
      assertArrayEquals(id, ExternalTestData.ints(info["T123"]), D1Audio.dims(bucket))
      assertArrayEquals(id, ExternalTestData.ints(info["L123"]), D1Audio.lengths(frames))
      val clip = D1Audio.info(waveform.size, frames, stft, bucket)
      assertEquals(id, (info["P"] as JsonNumber).toInt(), clip.prefixRows)
      // The graph's inputs: the masks bit for bit, the mel padded with zeros (its values are the mel's).
      val inputs = D1Audio.buildInputs(D1Mel(stages.out, stages.frames, stft), bucket)
      for ((name, values) in inputs.feeds()) {
        val python = floats(ExternalTestData.demoFile("$dumps/inputs/$name.f32"))
        assertEquals("$id $name", python.size, values.size)
        if (name != "mel") assertBitEqual("$id $name", python, values)
      }
      for (m in 0 until D1Audio.N_MELS) {
        for (t in stft until bucket) assertEquals(0f, inputs.mel[m * bucket + t], 0f)
        for (t in 0 until stft) assertEquals(stages.out[m * stft + t], inputs.mel[m * bucket + t], 0f)
      }
      // The rows after the app's own P: ids and markers of the fixture.
      val questions = entry["questions"] as Map<*, *>
      for (row in entry["expected"] as List<*>) {
        val expected = row as Map<*, *>
        val question = D1Prompt.asQuestion(questions[expected["name"]])
        val built =
          D1Rows.rows(tokenizer, contract, entry["state"], listOf(question), clip.prefixRows, D1Kind.AUDIO).single()
        rows++
        assertEquals((expected["P"] as JsonNumber).toInt(), built.prefixRows)
        if (
          built.ids.contentEquals(ExternalTestData.ints(expected["ids"])) &&
            built.markers.contentEquals(ExternalTestData.ints(expected["markers"]))
        ) {
          rowsMatched++
        }
      }
      report[id] =
        linkedMapOf(
          "samples" to samples.size,
          "n" to waveform.size,
          "frames" to frames,
          "T" to stft,
          "T_b" to bucket,
          "L123" to clip.lengths.toList(),
          "P" to clip.prefixRows,
          "stages_vs_python_f64" to
            linkedMapOf(
              "power_rel_max" to powerRel,
              "lin_rel_max" to linRel,
              "log_max_abs" to logAbs,
              "mean_max_abs" to meanAbs,
              "std_max_abs" to stdAbs,
            ),
          "mel_vs_python_f64" to vsF64,
          "mel_vs_python_f32" to vsF32,
          "log_in_float32_vs_python_f64" to log32VsF64,
          "log_in_float32_vs_python_f32" to log32VsF32,
        )
      println(
        "D1_AUDIO $id n=${waveform.size} frames=$frames T=$stft T_b=$bucket P=${clip.prefixRows} " +
          "mel vs f64 ${vsF64["max_abs"]} vs f32 ${vsF32["max_abs"]} (log in float32: ${log32VsF64["max_abs"]} / ${log32VsF32["max_abs"]}) " +
          "power rel $powerRel lin rel $linRel log $logAbs mean $meanAbs std $stdAbs"
      )
      assertTrue("$id power rel $powerRel", powerRel <= STAGE_RELATIVE_BAR)
      assertTrue("$id lin rel $linRel", linRel <= STAGE_RELATIVE_BAR)
      assertTrue("$id log $logAbs", logAbs <= STAGE_ABSOLUTE_BAR)
      assertTrue("$id mean $meanAbs std $stdAbs", meanAbs <= STAGE_ABSOLUTE_BAR && stdAbs <= STAGE_ABSOLUTE_BAR)
    }
    ExternalTestData.writeReport(
      "audio.json",
      linkedMapOf(
        "test" to "D1AudioTest",
        "clips" to report.size,
        "rows" to rows,
        "rows_identical" to rowsMatched,
        "mel_max_abs_vs_python_f64" to worstF64,
        "mel_max_abs_vs_python_f32" to worstF32,
        "kotlin_mel_ms_all_clips" to melMillis,
        "per_clip" to report,
        "execution" to "Desktop JVM ${System.getProperty("java.version")}",
      ),
    )
    println("D1_AUDIO clips=${report.size} rows=$rowsMatched/$rows mel max|d| vs f64 $worstF64 vs f32 $worstF32 mel ms (6 clips) $melMillis")
    assertEquals(6, report.size)
    assertEquals(18, rows)
    assertEquals(18, rowsMatched)
    assertTrue("mel vs the float64 host $worstF64", worstF64 <= MEL_BAR_F64)
    assertTrue("mel vs the float32 host $worstF32", worstF32 <= D1AudioCheck.MEL_ELEMENT_BAR)
  }

  @Test
  fun sizesBucketsAndWaveformEdges() {
    assertArrayEquals(intArrayOf(251, 126, 63), D1Audio.dims(501))
    assertArrayEquals(intArrayOf(501, 251, 126), D1Audio.dims(1001))
    assertArrayEquals(intArrayOf(1001, 501, 251), D1Audio.dims(2001))
    assertArrayEquals(intArrayOf(1501, 751, 376), D1Audio.dims(3001))
    assertArrayEquals(intArrayOf(870, 871), D1Audio.frameCount(139200))
    assertEquals(501, D1Audio.bucketFor(501))
    assertEquals(1001, D1Audio.bucketFor(502))
    assertEquals(3001, D1Audio.bucketFor(3001))
    assertEquals(1001, D1Audio.bucketFor(400, listOf(1001)))
    try {
      D1Audio.bucketFor(3002)
      fail("3002 frames accepted")
    } catch (expected: IllegalArgumentException) {
      // bucket_for raises past the largest bucket.
    }
    // contract.json's clip_samples_max of each bucket is the longest clip that bucket holds.
    val audio = (ExternalTestData.json(ExternalTestData.repoFile(D1Contract.FILE))["graphs"] as Map<*, *>)["audio"] as Map<*, *>
    for (entry in audio["buckets"] as List<*>) {
      val bucket = entry as Map<*, *>
      val T = (bucket["T"] as JsonNumber).toInt()
      val most = (bucket["clip_samples_max"] as JsonNumber).toInt()
      assertEquals(T, D1Audio.bucketFor(D1Audio.frameCount(most)[1]))
      assertEquals(D1Audio.lengths(D1Audio.frameCount(most)[0])[2], (bucket["P_max"] as JsonNumber).toInt())
      if (T > 501) assertEquals(T, D1Audio.bucketFor(D1Audio.frameCount(most - 159)[1]))
    }
    // waveform: / 32768, cut to 30 s, padded to 0.5 s.
    val short = D1Audio.waveform(shortArrayOf(-32768, 32767, 1))
    assertEquals(D1Audio.MIN_SAMPLES, short.size)
    assertEquals(-1f, short[0], 0f)
    assertEquals(32767f / 32768f, short[1], 0f)
    assertEquals(0f, short[D1Audio.MIN_SAMPLES - 1], 0f)
    assertEquals(30 * 16000, D1Audio.waveform(ShortArray(31 * 16000)).size)
    assertEquals(9000, D1Audio.waveform(FloatArray(9000) { 0.25f }).size)
  }

  @Test
  fun phoneRowsFileAndTheGateChecks() {
    val contract = ExternalTestData.contract()
    val tokenizer = ExternalTestData.tokenizer()
    val bytes = ExternalTestData.demoFile("device/r2/rows_audio.json").readBytes()
    assertEquals("audio", d1RowsKind(bytes))
    assertEquals("text", d1RowsKind(ExternalTestData.demoFile(ExternalTestData.ROWS_L128).readBytes()))
    val file = D1AudioRows.parse(bytes)
    assertEquals(6, file.records.size)
    assertEquals(18, file.rowCount)
    val core = D1GateCore(tokenizer, contract)
    var matched = 0
    val buckets = LinkedHashMap<Int, Int>()
    for (record in file.records) {
      val samples = D1Wav.read(ExternalTestData.repoFile("fixtures/media/${record.mediaFile}"))
      val waveform = D1Audio.waveform(samples)
      val mel = D1Audio.mel(waveform)
      val bucket = D1Audio.bucketFor(mel.stftFrames)
      val info = D1Audio.info(waveform.size, mel.frames, mel.stftFrames, bucket)
      assertEquals(record.id, true, D1AudioCheck.infoMatches(info, record))
      assertEquals(listOf("f32", "f64"), record.melFiles.keys.toList())
      // A stand-in prefix: the gate's inputs put it first, then the row's ids.
      val prefix = FloatArray(info.prefixRows * D1Rows.PREFIX_WIDTH) { 0.5f + it / D1Rows.PREFIX_WIDTH }
      for (row in record.rows) {
        val mine = D1AudioCheck.withPrefix(row, info.prefixRows)
        val recheck = D1AudioCheck.recheck(tokenizer, contract, record, row, info.prefixRows)
        if (recheck.idsMatch == true && recheck.markersMatch == true && recheck.stateMatch == true) matched++
        val length = requireNotNull(D1Contract.bucketFor(info.prefixRows + row.ids.size, listOf(128, 256)))
        buckets[length] = (buckets[length] ?: 0) + 1
        val inputs = core.inputs(mine, length, prefix)
        val p = info.prefixRows
        assertArrayEquals(row.ids, inputs.ids.copyOfRange(p, p + row.ids.size))
        assertTrue(inputs.ids.take(p).all { it == 0 } && inputs.ids.drop(p + row.ids.size).all { it == 0 })
        assertEquals(p, inputs.media.count { it == 1f })
        assertEquals(p + row.ids.size, inputs.pad.count { it == 1f })
        assertEquals(0f, inputs.keepRight[p - 1], 0f)
        assertEquals(length - 1, inputs.keepRight.count { it == 1f })
        assertEquals(0.5f + (p - 1), requireNotNull(inputs.prefix)[(p - 1) * D1Rows.PREFIX_WIDTH], 0f)
        assertEquals(0f, requireNotNull(inputs.prefix)[p * D1Rows.PREFIX_WIDTH], 0f)
        assertTrue(!mine.calibrate)
      }
    }
    println("D1_AUDIO rows file: recheck $matched/18, buckets $buckets")
    assertEquals(18, matched)
    assertEquals(mapOf(256 to 17, 128 to 1), buckets.toMap())
    val sets = D1AudioTimingSet.parse(ExternalTestData.demoFile("device/r2/timing_audio.json").readBytes())
    assertEquals(listOf("food3"), sets.map { it.name })
    assertEquals("aud_food_03", sets.single().record.id)
    assertEquals(listOf("topic", "request", "urgency"), sets.single().record.questions.keys.toList())
    // A text row still refuses a prefix, a media row needs one.
    val text = D1GateRows.parse(ExternalTestData.demoFile(ExternalTestData.ROWS_L128).readBytes()).rows.first()
    try {
      core.inputs(text, 128, FloatArray(D1Rows.PREFIX_WIDTH))
      fail("a text row took a prefix")
    } catch (expected: IllegalArgumentException) {
      // build_inputs: a text row has no prefix.
    }
    try {
      core.inputs(file.records.first().rows.first(), 256)
      fail("an audio row ran without its prefix")
    } catch (expected: IllegalArgumentException) {
      // a media row needs its prefix
    }
  }

  @Test
  fun launchExtraPrecisionAudio() {
    class Extras(val values: Map<String, Any>) : D1Extras {
      override fun has(name: String) = values.containsKey(name)

      override fun string(name: String) = values[name] as String?

      override fun int(name: String, default: Int) = values[name] as Int? ?: default

      override fun boolean(name: String, default: Boolean) = values[name] as Boolean? ?: default
    }
    val normal = D1Launch.parse(Extras(emptyMap()), true) as D1Launch.Normal
    assertEquals(D1Precision.FP32, normal.precision)
    assertEquals(D1AudioEngine.DEFAULT_PRECISION, normal.audioPrecision)
    val gate =
      D1Launch.parse(
        Extras(
          mapOf("gate" to true, "fixture" to "rows_audio.json", "report" to "A.json", "precision" to "fp32", "precision_audio" to "fp32")
        ),
        true,
      ) as D1Launch.Gate
    assertEquals(D1Precision.FP32, gate.precision)
    assertEquals(D1Precision.FP32, gate.audioPrecision)
    val timing =
      D1Launch.parse(
        Extras(mapOf("timing" to true, "rows" to "timing_audio.json", "report" to "T.json", "precision_audio" to "fp16acc")),
        true,
      ) as D1Launch.Timing
    assertEquals(D1Precision.FP16_FP32_ACCUM, timing.audioPrecision)
    assertTrue(D1Launch.parse(Extras(mapOf("precision_audio" to "fp16")), true) is D1Launch.Invalid)
    assertNull(D1Launch.parse(Extras(mapOf("precision_audio" to "FP32")), true).let { (it as? D1Launch.Invalid)?.reason })
  }

  private fun assertBitEqual(what: String, expected: FloatArray, actual: FloatArray) {
    assertEquals(what, expected.size, actual.size)
    val first = expected.indices.firstOrNull { expected[it].toRawBits() != actual[it].toRawBits() }
    assertNull("$what differs at $first", first)
  }

  /** How many values differ and the largest difference in float32 units in the last place. */
  private fun ulpDifferences(a: FloatArray, b: FloatArray): Pair<Int, Int> {
    assertEquals(a.size, b.size)
    var count = 0
    var worst = 0
    for (i in a.indices) {
      if (a[i].toRawBits() != b[i].toRawBits()) {
        count++
        worst = max(worst, abs(a[i].toRawBits() - b[i].toRawBits()))
      }
    }
    return count to worst
  }

  /**
   * The largest |a - b| of a [rows, frames] array (row-major) relative to the largest |b| of its frame
   * (column): the FFT's float64 error scales with the frame, not with a single bin.
   */
  private fun relativeToFrame(a: DoubleArray, b: DoubleArray, frames: Int): Double {
    assertEquals(a.size, b.size)
    val rows = a.size / frames
    var worst = 0.0
    for (t in 0 until frames) {
      var scale = 0.0
      var diff = 0.0
      for (r in 0 until rows) {
        scale = max(scale, abs(b[r * frames + t]))
        diff = max(diff, abs(a[r * frames + t] - b[r * frames + t]))
      }
      if (scale > 0) worst = max(worst, diff / scale) else assertEquals(0.0, diff, 0.0)
    }
    return worst
  }

  private fun absolute(a: DoubleArray, b: DoubleArray): Double {
    assertEquals(a.size, b.size)
    var worst = 0.0
    for (i in a.indices) worst = max(worst, abs(a[i] - b[i]))
    return worst
  }

  private fun floats(file: File): FloatArray = D1AudioCheck.floats(file.readBytes())

  private fun doubles(file: File): DoubleArray {
    val bytes = file.readBytes()
    val out = DoubleArray(bytes.size / 8)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(out)
    return out
  }

  private companion object {
    /** The mel against the host's float64 form: the same math in another FFT and summation order. */
    const val MEL_BAR_F64 = 1e-4
    /** power and lin against the host's float64 stages, relative to each frame's largest value. */
    const val STAGE_RELATIVE_BAR = 1e-9
    /** log, mean and std against the host's float64 stages. */
    const val STAGE_ABSOLUTE_BAR = 1e-6
  }
}

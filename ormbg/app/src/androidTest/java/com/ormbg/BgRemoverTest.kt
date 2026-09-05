package com.ormbg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Integration check for [BgRemover]: compiles the model for the GPU on the connected device,
 * segments a fixed photo and compares the matte against the values recorded in
 * `ormbg/INTEGRATION.md`. Run with `./gradlew :app:connectedGpuDebugAndroidTest` and read the
 * `RESULT` line in logcat (tag `ormbg`).
 *
 * The fixture is one frame of the demo clip listed in `ormbg/DEMO_ASSETS.md` (Pexels license, no
 * attribution required). The test also writes the preprocessed input tensor, the raw model output
 * and the cutout PNG into the app's `filesDir/ormbg_test/`, so the run can be compared against the
 * source model on a host (`scripts/verify_device_dump.py`) and looked at.
 */
@RunWith(AndroidJUnit4::class)
class BgRemoverTest {

  private val app: Context = InstrumentationRegistry.getInstrumentation().targetContext
  private val testApk: Context = InstrumentationRegistry.getInstrumentation().context

  private fun fixture(): Bitmap =
    testApk.assets.open(FIXTURE).use { BitmapFactory.decodeStream(it) }
      ?: error("cannot decode $FIXTURE")

  private fun thermalStatus(): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      (app.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus
    } else {
      -1
    }

  /** Mean alpha over a [side]×[side] patch whose top-left corner is ([x0], [y0]). */
  private fun patchMean(m: Matte, x0: Int, y0: Int, side: Int): Float {
    var sum = 0f
    for (y in y0 until y0 + side) for (x in x0 until x0 + side) sum += m.alpha[y * m.size + x]
    return sum / (side * side)
  }

  @Test
  fun gpuCompilesAndSegmentsThePerson() {
    val bitmap = fixture()
    val thermalBefore = thermalStatus()
    BgRemover.fromAssets(app).use { remover ->
      val matte = remover.process(bitmap) ?: error("process() returned null without cancel()")
      val s = matte.size
      val cutout = matte.cutout(bitmap)

      // Everything below is written before any assertion so a failing run still leaves evidence.
      val dir = File(app.filesDir, "ormbg_test").apply { mkdirs() }
      writeFloats(File(dir, "input_nchw.f32"), remover.lastInput())
      val scale = matte.rawMax - matte.rawMin + 1e-6f
      writeFloats(
        File(dir, "raw_out.f32"),
        FloatArray(matte.alpha.size) { matte.alpha[it] * scale + matte.rawMin },
      )
      File(dir, "cutout.png").outputStream().use {
        cutout.compress(Bitmap.CompressFormat.PNG, 100, it)
      }

      // Timing: median over WARM_RUNS calls (the constructor already ran the warm-up inference).
      val modelTimes = LongArray(WARM_RUNS)
      val processTimes = LongArray(WARM_RUNS)
      for (i in 0 until WARM_RUNS) {
        val t0 = System.nanoTime()
        remover.process(bitmap) ?: error("process() returned null")
        processTimes[i] = (System.nanoTime() - t0) / 1_000_000
        modelTimes[i] = remover.modelMs
      }

      var mean = 0.0
      for (a in matte.alpha) mean += a
      mean /= matte.alpha.size
      val fg = matte.foregroundFraction()
      val corners =
        (patchMean(matte, 0, 0, PATCH) +
          patchMean(matte, s - PATCH, 0, PATCH) +
          patchMean(matte, 0, s - PATCH, PATCH) +
          patchMean(matte, s - PATCH, s - PATCH, PATCH)) / 4f
      val center = patchMean(matte, s / 2 - PATCH / 2, s / 2 - PATCH / 2, PATCH)

      Log.i(
        TAG,
        "RESULT device=${Build.MODEL} sdk=${Build.VERSION.SDK_INT} accel=GPU " +
          "load_ms=${remover.loadMs} model_ms_median=${modelTimes.median()} " +
          "process_ms_median=${processTimes.median()} runs=$WARM_RUNS " +
          "raw_min=%.4f raw_max=%.4f mean=%.4f fg_frac=%.4f center=%.4f corners=%.4f thermal=%d->%d"
            .format(
              matte.rawMin,
              matte.rawMax,
              mean,
              fg,
              center,
              corners,
              thermalBefore,
              thermalStatus(),
            ),
      )

      // Task-level gates: a person filling the middle of the frame, sky and window at the edges.
      assertTrue(
        "raw output range collapsed: ${matte.rawMin}..${matte.rawMax}",
        matte.rawMax - matte.rawMin > 0.5f,
      )
      assertTrue("center of the frame should be subject, got $center", center > 0.9f)
      assertTrue("frame corners should be background, got $corners", corners < 0.1f)
      // Values recorded on a Pixel 8a (see INTEGRATION.md); the tolerance covers fp16 GPU noise.
      assertEquals("foreground fraction", EXPECTED_FG_FRACTION, fg, 0.02f)
      assertEquals("mean alpha", EXPECTED_MEAN, mean.toFloat(), 0.02f)
      // The cutout carries the matte in its alpha channel: transparent sky, opaque subject.
      assertTrue("cutout corner should be transparent", Color.alpha(cutout.getPixel(2, 2)) < 26)
      assertTrue(
        "cutout center should be opaque",
        Color.alpha(cutout.getPixel(cutout.width / 2, cutout.height / 2)) > 230,
      )
    }
  }

  @Test
  fun loadsFromAFilePath() {
    // The same model staged in filesDir — the pattern for apps that download the file at runtime.
    val staged = File(app.filesDir, "ormbg_staged.tflite")
    if (!staged.exists()) {
      app.assets.open("ormbg.tflite").use { src -> staged.outputStream().use { src.copyTo(it) } }
    }
    val bitmap = fixture()
    BgRemover.fromFile(app, staged.absolutePath).use { remover ->
      val matte = remover.process(bitmap) ?: error("process() returned null")
      assertEquals("foreground fraction", EXPECTED_FG_FRACTION, matte.foregroundFraction(), 0.02f)
    }
    staged.delete()
  }

  @Test
  fun cancelOnlyAffectsTheCallInFlight() {
    val bitmap = fixture()
    BgRemover.fromAssets(app).use { remover ->
      remover.cancel() // nothing in flight: must not poison the next call
      val matte = remover.process(bitmap)
      assertTrue("a call started after cancel() must complete", matte != null)
    }
  }

  private fun writeFloats(file: File, data: FloatArray) {
    val buf = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buf.asFloatBuffer().put(data)
    file.outputStream().use { it.write(buf.array()) }
  }

  private fun LongArray.median(): Long = sorted()[size / 2]

  companion object {
    private const val TAG = "ormbg"
    private const val FIXTURE = "person.jpg"
    private const val WARM_RUNS = 20
    private const val PATCH = 64

    // Recorded on a Pixel 8a (Android 16, LiteRT 2.2.0, GPU) on 2026-09-05; the host CPU fp32
    // reference gives the same values to 4 digits (scripts/verify_device_dump.py). Re-record from
    // the RESULT line if you change the fixture.
    private const val EXPECTED_FG_FRACTION = 0.5213f
    private const val EXPECTED_MEAN = 0.5208f
  }
}

package com.ormbg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.util.concurrent.atomic.AtomicInteger

/**
 * Background removal (person / subject segmentation) with ormbg on LiteRT `CompiledModel`.
 *
 * This file is a drop-in for an existing app: it depends on nothing but the Android graphics
 * classes and `com.google.ai.edge.litert:litert`. Copy it, put `ormbg.tflite` next to your other
 * assets, then `BgRemover.fromAssets(context)` once and `process(bitmap)` per image. See
 * `ormbg/INTEGRATION.md` in LiteRT-Models for the full recipe and the verified numbers.
 *
 * Model contract (`litert-community/ormbg-LiteRT`, converted from `schirrmacher/ormbg`):
 * - Input: `[1, 3, 1024, 1024]` NCHW, RGB, `x / 255`, the image stretched to the square (the
 *   upstream inference script does the same; there is no letterbox).
 * - Output: `[1, 1, 1024, 1024]` sigmoid matte, min-max normalized per image (also upstream).
 *
 * Threading: [process] and [close] must run off the main thread and are serialized against each
 * other. [cancel] may be called from any thread. `Accelerator.GPU` compiles the whole graph for the
 * GPU or fails; there is no silent CPU fallback.
 */
class BgRemover
private constructor(
  context: Context,
  private val accelerator: Accelerator,
  create: (CompiledModel.Options, Environment?) -> CompiledModel,
) : AutoCloseable {

  companion object {
    private const val TAG = "ormbg"

    /** Model input and output edge length, in pixels. */
    const val SIZE = 1024

    /**
     * Loads `assets/<assetName>`. The module must keep the file uncompressed (`androidResources {
     * noCompress += "tflite" }`) so it can be memory-mapped.
     */
    fun fromAssets(
      context: Context,
      assetName: String = "ormbg.tflite",
      accelerator: Accelerator = Accelerator.GPU,
    ): BgRemover =
      BgRemover(context, accelerator) { options, env ->
        CompiledModel.create(context.assets, assetName, options, env)
      }

    /** Loads the model from an absolute path, e.g. a file staged into the app's `filesDir`. */
    fun fromFile(
      context: Context,
      path: String,
      accelerator: Accelerator = Accelerator.GPU,
    ): BgRemover =
      BgRemover(context, accelerator) { options, env -> CompiledModel.create(path, options, env) }
  }

  private var env: Environment? = null
  private val model: CompiledModel
  private val inputBuffers: List<TensorBuffer>
  private val outputBuffers: List<TensorBuffer>

  private val input = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val square = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val squareCanvas = Canvas(square)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  private var closed = false
  private val generation = AtomicInteger(0)
  @Volatile private var cancelledGeneration = 0

  /** Wall time of compile + load + one warm-up inference, in ms. */
  var loadMs: Long = 0
    private set

  /**
   * Inference time of the last [process] call, in ms: `run()` plus the readback that waits for it.
   * Pre- and post-processing are excluded.
   */
  var modelMs: Long = 0
    private set

  init {
    val t0 = System.nanoTime()
    val options = CompiledModel.Options(accelerator)
    if (accelerator == Accelerator.NPU) {
      // The NPU needs the dispatch library directory explicitly: LiteRT only warns when it is
      // missing and then runs without the NPU. On-device (JIT) compilation needs the compiler
      // plugin as well; without it the model silently runs on CPU and still returns a number.
      env =
        Environment.create(
          context,
          mapOf(
            Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
            Environment.Option.CompilerPluginLibraryDir to context.applicationInfo.nativeLibraryDir,
          ),
        )
      options.qualcommOptions =
        CompiledModel.QualcommOptions(
          htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
        )
    }
    model = create(options, env)
    inputBuffers = model.createInputBuffers()
    outputBuffers = model.createOutputBuffers()
    // Warm-up: the first inference compiles the GPU shaders (or the NPU graph). Paying for it
    // here keeps it out of the first process() call.
    inputBuffers[0].writeFloat(input)
    model.run(inputBuffers, outputBuffers)
    outputBuffers[0].readFloat()
    loadMs = (System.nanoTime() - t0) / 1_000_000
    Log.i(TAG, "$accelerator ready in ${loadMs}ms (compile + load + warm-up)")
  }

  /**
   * Segments [bitmap] and returns its matte, or null if [cancel] was called while this call was in
   * flight. Any bitmap size works; the result is [SIZE]×[SIZE] and maps back to the bitmap by plain
   * scaling ([Matte.cutout] and [Matte.composite] do that for you).
   */
  fun process(bitmap: Bitmap): Matte? =
    synchronized(this) {
      check(!closed) { "BgRemover is closed" }
      val id = generation.incrementAndGet()

      // Pre-processing: stretch to SIZE×SIZE, then split ARGB pixels into RGB planes in [0, 1].
      matrix.setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height)
      squareCanvas.drawBitmap(bitmap, matrix, paint)
      square.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
      val plane = SIZE * SIZE
      for (i in 0 until plane) {
        val p = pixels[i]
        input[i] = ((p shr 16) and 0xFF) / 255f
        input[plane + i] = ((p shr 8) and 0xFF) / 255f
        input[2 * plane + i] = (p and 0xFF) / 255f
      }
      if (cancelledGeneration == id) return null

      // Inference. run() only enqueues the work; the readback is what waits for the GPU, so the
      // two are always paired and timed together.
      val t0 = System.nanoTime()
      inputBuffers[0].writeFloat(input)
      model.run(inputBuffers, outputBuffers)
      val raw = outputBuffers[0].readFloat()
      modelMs = (System.nanoTime() - t0) / 1_000_000
      if (cancelledGeneration == id) return null

      // Post-processing: min-max normalize per image, as the upstream script does.
      var min = Float.MAX_VALUE
      var max = -Float.MAX_VALUE
      for (v in raw) {
        if (v < min) min = v
        if (v > max) max = v
      }
      val scale = 1f / (max - min + 1e-6f)
      for (i in raw.indices) raw[i] = (raw[i] - min) * scale
      Matte(raw, SIZE, min, max)
    }

  /**
   * Aborts the [process] call in flight, if any: that call returns null instead of a matte. Calls
   * that start later run normally. The GPU work already enqueued is left to finish.
   */
  fun cancel() {
    cancelledGeneration = generation.get()
  }

  /** The last preprocessed input tensor (NCHW, [0, 1]). Used by the instrumented test only. */
  internal fun lastInput(): FloatArray = input

  override fun close() {
    synchronized(this) {
      if (closed) return
      closed = true
      // TensorBuffers hold native memory of their own; closing the model does not free them.
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
      env?.close()
      square.recycle()
    }
  }
}

/**
 * Result of [BgRemover.process]: a [size]×[size] alpha matte in `[0, 1]`, row-major, 1 = subject.
 */
class Matte(
  val alpha: FloatArray,
  val size: Int,
  /** Range of the model output before normalization. A near-zero range means a broken input. */
  val rawMin: Float,
  val rawMax: Float,
) {
  /** Fraction of pixels whose alpha is above [threshold]. */
  fun foregroundFraction(threshold: Float = 0.5f): Float {
    var n = 0
    for (a in alpha) if (a > threshold) n++
    return n.toFloat() / alpha.size
  }

  /** Nearest-neighbour downsample to [outSize]×[outSize], for realtime compositing. */
  fun downsample(outSize: Int): FloatArray {
    val step = size / outSize
    val out = FloatArray(outSize * outSize)
    for (y in 0 until outSize) {
      val row = y * step * size
      for (x in 0 until outSize) out[y * outSize + x] = alpha[row + x * step]
    }
    return out
  }

  /**
   * The matte as an `ARGB_8888` bitmap whose alpha channel is the matte (color black),
   * [size]×[size]. Scale it to any bitmap's size, or draw it with `PorterDuff.Mode.DST_IN` to mask
   * one. (An `ALPHA_8` bitmap does not work for that: Skia treats an alpha-only image as coverage,
   * and a `DST_IN` draw with it leaves the destination unchanged.)
   */
  fun toMaskBitmap(): Bitmap {
    val px = IntArray(alpha.size)
    for (i in alpha.indices) px[i] = (alpha[i] * 255f + 0.5f).toInt().coerceIn(0, 255) shl 24
    return Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888).apply {
      setPixels(px, 0, size, 0, 0, size, size)
    }
  }

  /**
   * [source] with its background made transparent, at [source]'s own size: the matte is scaled to
   * it bilinearly and written into the alpha channel. [source] must be a software bitmap (not
   * `Bitmap.Config.HARDWARE`).
   */
  fun cutout(source: Bitmap): Bitmap {
    val w = source.width
    val h = source.height
    val mask = toMaskBitmap()
    val scaled = if (w == size && h == size) mask else Bitmap.createScaledBitmap(mask, w, h, true)
    val px = IntArray(w * h)
    val maskPx = IntArray(w * h)
    source.getPixels(px, 0, w, 0, 0, w, h)
    scaled.getPixels(maskPx, 0, w, 0, 0, w, h)
    for (i in px.indices) px[i] = (maskPx[i] and 0xFF000000.toInt()) or (px[i] and 0x00FFFFFF)
    if (scaled !== mask) scaled.recycle()
    mask.recycle()
    return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
      setPixels(px, 0, w, 0, 0, w, h)
    }
  }

  /** [source] composited over a solid [background] color (`0xAARRGGBB`). */
  fun composite(source: Bitmap, background: Int): Bitmap {
    val out = Bitmap.createBitmap(source.width, source.height, Bitmap.Config.ARGB_8888)
    out.eraseColor(background)
    val foreground = cutout(source)
    Canvas(out).drawBitmap(foreground, 0f, 0f, null)
    foreground.recycle()
    return out
  }
}

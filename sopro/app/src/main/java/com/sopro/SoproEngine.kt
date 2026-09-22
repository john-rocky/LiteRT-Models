// SPDX-License-Identifier: Apache-2.0
package com.sopro

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.SplittableRandom
import java.util.concurrent.CancellationException
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject

/** Single-thread confined host with exact contract overlays and per-graph placement. */
class SoproEngine(
  private val filesDir: File,
  private val placement: Map<String, Backend> = emptyMap(),
  private val defaultBackend: Backend = Backend.CPU,
  private val precision: Precision = Precision.SHIP,
  val contractSet: String = "r9",
  val styleVariant: PlacementConfig.StyleVariant = PlacementConfig.StyleVariant.FP32,
  requiredGraphs: List<String> = requiredGraphNames,
) : Closeable {
  enum class Backend {
    CPU,
    GPU,
    GPU32;

    companion object {
      fun parse(value: String) =
        entries.firstOrNull { it.name.equals(value, true) }
          ?: error("Unknown placement '$value'; expected cpu, gpu or gpu32")
    }
  }

  enum class Precision {
    FP32,
    WFP16,
    SHIP;

    companion object {
      fun parse(value: String) = valueOf(value.uppercase())
    }
  }

  data class CallTiming(
    val graph: String,
    val signature: String,
    val backend: String,
    val writeMs: Double,
    val runMs: Double,
    val readbackMs: Double,
    val totalMs: Double,
  )

  data class Reference(
    val wav24: FloatArray,
    val levelDb: Double,
    val idEmbedding: FloatArray,
    val conditioning: FloatArray,
    val tokens: IntArray,
    val mel: FloatArray,
  )

  data class Output(
    val wav: FloatArray,
    val raw: FloatArray,
    val tokens: IntArray,
    val solvedMel: FloatArray,
    val features: FloatArray,
    val logits: List<FloatArray>,
    val trim: PostProcess.Trim,
    val graphTimings: List<CallTiming>,
    val elapsedMs: Double,
    val prefixLength: Int,
    val stopReason: String,
    val reference: Reference,
    val stats: SynthesisStats? = null,
  )

  val timings = mutableListOf<CallTiming>()
  private val catalog =
    ModelCatalog(filesDir, precision, contractSet, styleVariant).also {
      it.requireFiles(requiredGraphs)
    }
  private val assetsDir = File(filesDir, "host_assets")
  val dsp = HostDsp.fromAssets(assetsDir)
  private val tokenizer = SpTokenizer(File(assetsDir, "tokenizer.model"))
  private val ar =
    ArHost(File(assetsDir, "ar_tables_fp32.bin"), File(assetsDir, "ar_tables_fp32.json"))
  private val graphs = linkedMapOf<String, Graph>()
  private val rotary = ArHost.rotaryCosSin()
  private val referenceCache = linkedMapOf<String, Reference>()
  private var cancellation: () -> Boolean = { false }
  private val stageCalls = mutableListOf<Pair<String, Double>>()
  /** Optional debug producer hook; called with the exact graph input arrays, without judging. */
  var recordInputs: ((String, String, List<Any>) -> Unit)? = null
  /** Raw producer evidence only; unset for normal synthesis and release timing. */
  var recordOutputs: ((String, String, List<Any>) -> Unit)? = null
  val creationMs = linkedMapOf<String, Double>()

  private fun modelSpec(name: String): JSONObject = catalog.spec(name)

  /**
   * Convert published recorded inputs into the selected exact graph's host contract. Already
   * adapted records are accepted, so GPU probes can replay either generation.
   */
  fun adaptRecordedInputs(name: String, signature: String, args: List<Any>): List<Any> =
    when (modelSpec(name).optString("host_contract", "unchanged")) {
      "ar_last_onehot" ->
        if (signature == "prefill" && args[2] is IntArray) {
          val last = args[2] as IntArray
          require(last.size == 1)
          listOf(args[0], args[1], ArHost.lastOneHot(last[0] + 1))
        } else args
      "acoustic_onehot" ->
        if (args[0] is IntArray) {
          val tokens = args[0] as IntArray
          listOf(
            AcousticHost.semanticOneHot(tokens),
            args[1],
            AcousticHost.frameOneHot(args[2] as IntArray, tokens.size),
          )
        } else args
      else -> args
    }

  private fun semanticTokens(output: Any): IntArray =
    when (output) {
      is FloatArray -> SemanticTokens.decode(output)
      is IntArray -> output
      else -> error("Unsupported semantic graph output")
    }

  /** Contract buffer_index controls binding; args_N/output_N controls semantic argument order. */
  fun invoke(name: String, args: List<Any>, signature: String = "serving_default"): List<Any> {
    checkActive()
    recordInputs?.invoke(name, signature, args)
    val started = System.nanoTime()
    compileGraph(name)
    val output = graphs.getValue(name).call(signature, args)
    recordOutputs?.invoke(name, signature, output)
    stageCalls += "$name/$signature" to ms(System.nanoTime() - started)
    checkActive()
    return output
  }

  fun compileGraph(name: String): Double {
    if (name !in graphs) {
      val start = System.nanoTime()
      graphs[name] = Graph(name, modelSpec(name))
      creationMs[name] = ms(System.nanoTime() - start)
    }
    return creationMs.getValue(name)
  }

  private fun checkActive() {
    if (cancellation()) throw CancellationException("Synthesis cancelled")
  }

  val placementDescription: String
    get() =
      requiredGraphNames.joinToString(", ") {
        "$it=${(placement[it] ?: defaultBackend).name.lowercase()}"
      }

  private fun cachedReference(
    wav24: FloatArray,
    referenceId: String,
    fixedLevelDb: Double?,
  ): Pair<Reference, Boolean> {
    val buffer = ByteBuffer.allocate(wav24.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    wav24.forEach { buffer.putFloat(it) }
    val digest =
      MessageDigest.getInstance("SHA-256").digest(buffer.array()).joinToString("") {
        "%02x".format(it)
      }
    val key = "$referenceId:$digest:$fixedLevelDb"
    referenceCache[key]?.let {
      return it to true
    }
    val prepared =
      if (fixedLevelDb == null) prepareReference(wav24)
      else prepareFixedReference(wav24, fixedLevelDb)
    if (referenceCache.size >= 2) referenceCache.remove(referenceCache.keys.first())
    referenceCache[key] = prepared
    return prepared to false
  }

  fun prepareReference(wav24: FloatArray): Reference {
    val normalized = dsp.fixedReference(wav24)
    return prepareFixedReference(normalized.wav, normalized.levelDb)
  }

  /** Gate input is already normalized and cropped; applying normalization twice is incorrect. */
  fun prepareFixedReference(wav24: FloatArray, levelDb: Double): Reference {
    require(wav24.size == 240000) { "The fixed reference must contain 240000 samples at 24 kHz" }
    val wav16 = dsp.resample24to16(wav24)
    val speaker = invoke("speaker_encoder", listOf(dsp.speakerMel(wav16)))
    val semantic = invoke("semantic_encoder", listOf(dsp.semanticMel(wav16)))
    return Reference(
      wav24,
      levelDb,
      speaker[0] as FloatArray,
      speaker[3] as FloatArray,
      semanticTokens(semantic[0]),
      dsp.acousticMelNormalized(wav24),
    )
  }

  /**
   * Raw PCM callbacks are emitted during decode, with reference context removed. Final exact
   * gain/trim/limiter/fade output is returned once the segment is complete.
   */
  fun synthesize(
    text: String,
    lang: String,
    reference: FloatArray,
    seed: Long = System.nanoTime(),
    onPcm: (FloatArray) -> Unit = {},
  ): Output {
    val started = System.nanoTime()
    timings.clear()
    stageCalls.clear()
    val prepared = prepareReference(reference)
    val textIds = tokenizer.encode(text, lang)
    val runner = ArSession(prepared.tokens, textIds)
    val random = SplittableRandom(seed)
    val logits = mutableListOf(runner.initial)
    val generated =
      Sampler.generate(
        runner.initial,
        { token ->
          runner.advance(token).also { logits.add(it) }
        },
        object : UniformSource {
          override fun next(): Double = random.nextDouble()
        },
      )
    val valid = 938 + generated.tokens.size * 4
    // A new independent stream supplies Gaussian acoustic noise; teacher gates supply x0 directly.
    val x0 = gaussian(100 * valid, SplittableRandom(seed xor 0x5deece66dL))
    return decode(
      prepared,
      generated.tokens,
      x0,
      logits,
      runner.prefixLength,
      generated.stopReason,
      started,
      onPcm,
    )
  }

  /**
   * Prepares and caches the reference, samples AR tokens, solves the full acoustic utterance, then
   * emits vocoder/iSTFT chunks after removing 32 x 256 reference-context samples. Playback applies
   * only output gain and soft limiting, with no trim or fade. The saved WAV equals the round-1
   * contract; the played stream differs at the trimmed edges and at the saved WAV's final 80 ms
   * fade-out. Before that fade the kept samples are exact. TTFA starts at [tapNanos] and ends
   * immediately before the first chunk is handed to [sink]. Calls, cache and CompiledModels are
   * confined to the caller's single model dispatcher.
   */
  fun synthesizeStreaming(
    text: String,
    lang: String,
    reference: FloatArray,
    sink: (FloatArray) -> Unit,
    seed: Long = System.nanoTime(),
    tapNanos: Long = System.nanoTime(),
    referenceId: String = "reference",
    shouldStop: () -> Boolean = { false },
    fixedReferenceLevelDb: Double? = null,
  ): Output {
    cancellation = shouldStop
    timings.clear()
    stageCalls.clear()
    try {
      checkActive()
      val referenceStart = System.nanoTime()
      // Fixed gate fixtures have already been normalized over the original full clip.
      val (prepared, hit) = cachedReference(reference, referenceId, fixedReferenceLevelDb)
      val prepMs = ms(System.nanoTime() - referenceStart)
      val textIds = tokenizer.encode(text, lang)
      val prefillStart = System.nanoTime()
      val runner = ArSession(prepared.tokens, textIds)
      val prefillMs = ms(System.nanoTime() - prefillStart)
      val arStepMs = mutableListOf<Double>()
      val random = SplittableRandom(seed)
      val logits = mutableListOf(runner.initial)
      val arStart = System.nanoTime()
      val generated =
        Sampler.generate(
          runner.initial,
          { token ->
            checkActive()
            val stepStart = System.nanoTime()
            runner.advance(token).also {
              arStepMs += ms(System.nanoTime() - stepStart)
              logits.add(it)
            }
          },
          object : UniformSource {
            override fun next(): Double {
              checkActive()
              return random.nextDouble()
            }
          },
        )
      val arTotalMs = ms(System.nanoTime() - arStart)
      val valid = 938 + generated.tokens.size * 4
      val x0 = gaussian(100 * valid, SplittableRandom(seed xor 0x5deece66dL))
      val output =
        decode(
          prepared,
          generated.tokens,
          x0,
          logits,
          runner.prefixLength,
          generated.stopReason,
          tapNanos,
          sink,
          StreamingContext(seed, referenceId, hit, prepMs, prefillMs, arStepMs, arTotalMs),
        )
      checkActive()
      return output
    } finally {
      cancellation = { false }
    }
  }

  private data class StreamingContext(
    val seed: Long,
    val referenceId: String,
    val cacheHit: Boolean,
    val referencePrepMs: Double,
    val prefillMs: Double,
    val arStepMs: List<Double>,
    val arTotalMs: Double,
  )

  /**
   * Replays all fixed tokens, including the final teacher prediction, as Python inference.py does.
   */
  fun teacher(
    textIds: IntArray,
    fixedWav24: FloatArray,
    levelDb: Double,
    referenceTokens: IntArray,
    tokens: IntArray,
    x0: FloatArray,
    onPcm: (FloatArray) -> Unit = {},
  ): Output {
    val started = System.nanoTime()
    timings.clear()
    stageCalls.clear()
    val computedReference = prepareFixedReference(fixedWav24, levelDb)
    val reference = computedReference.copy(tokens = referenceTokens)
    val runner = ArSession(reference.tokens, textIds)
    val logits = mutableListOf(runner.initial)
    tokens.forEach { logits += runner.advance(it) }
    return decode(
        reference,
        tokens,
        x0,
        logits,
        runner.prefixLength,
        "oracle teacher forcing",
        started,
        onPcm,
      )
      .copy(reference = computedReference)
  }

  /** Device producer uses frozen mel graph inputs, isolating graph/host parity from DSP. */
  fun teacherFromMels(
    textIds: IntArray,
    fixedWav24: FloatArray,
    levelDb: Double,
    speakerMel: FloatArray,
    semanticMel: FloatArray,
    acousticMel: FloatArray,
    referenceTokens: IntArray,
    tokens: IntArray,
    x0: FloatArray,
  ): Output {
    val started = System.nanoTime()
    timings.clear()
    stageCalls.clear()
    val speaker = invoke("speaker_encoder", listOf(speakerMel))
    val semantic = semanticTokens(invoke("semantic_encoder", listOf(semanticMel))[0])
    val computed =
      Reference(
        fixedWav24,
        levelDb,
        speaker[0] as FloatArray,
        speaker[3] as FloatArray,
        semantic,
        acousticMel,
      )
    // Published teacher pipeline fixes the reference token sequence after measuring the encoder.
    val reference = computed.copy(tokens = referenceTokens)
    val runner = ArSession(reference.tokens, textIds)
    val logits = mutableListOf(runner.initial)
    tokens.forEach { logits += runner.advance(it) }
    return decode(
        reference,
        tokens,
        x0,
        logits,
        runner.prefixLength,
        "oracle teacher forcing",
        started,
        {},
      )
      .copy(reference = computed)
  }

  private inner class ArSession(refTokens: IntArray, textIds: IntArray) {
    private val cache = ArHost.PackedKv()
    val prefixLength: Int
    val initial: FloatArray

    init {
      val style =
        invoke("style_prefix", listOf(ar.semanticEmbedding(refTokens.copyOfRange(0, 160))))[0]
          as FloatArray
      val prefix = ar.assemblePrefix(style, textIds, refTokens)
      prefixLength = prefix.length
      val inputs =
        adaptRecordedInputs(
          "ar_merged",
          "prefill",
          listOf(
            prefix.values.copyOf(256 * 512),
            ArHost.prefillBias(prefix.length),
            intArrayOf(prefix.length - 1),
          ),
        )
      val output = invoke("ar_merged", inputs, "prefill")
      initial = output[0] as FloatArray
      cache.initialize(output[1] as FloatArray, output[2] as FloatArray, prefix.length)
    }

    fun advance(token: Int): FloatArray {
      val p = cache.position
      require(p < 1024) { "AR cache capacity exceeded at position $p" }
      val output =
        invoke(
          "ar_merged",
          listOf(
            ar.semanticEmbedding(intArrayOf(token)),
            rotary.first.copyOfRange(p * 64, (p + 1) * 64),
            rotary.second.copyOfRange(p * 64, (p + 1) * 64),
            ArHost.stepBias(p),
            cache.keys,
            cache.values,
          ),
          "step",
        )
      cache.writeRow(output[1] as FloatArray, output[2] as FloatArray)
      return output[0] as FloatArray
    }
  }

  private fun decode(
    reference: Reference,
    tokens: IntArray,
    x0: FloatArray,
    logits: List<FloatArray>,
    prefixLength: Int,
    stopReason: String,
    started: Long,
    onPcm: (FloatArray) -> Unit,
    streaming: StreamingContext? = null,
  ): Output {
    val p =
      AcousticHost.prepareInputs(
        reference.tokens,
        tokens,
        x0,
        reference.conditioning,
        reference.mel,
      )
    val suffix = if (p.frames == 4096) "_t4096" else ""
    val conditionInputs =
      adaptRecordedInputs(
        "acoustic_condition$suffix",
        "serving_default",
        listOf(p.tokens, p.tokenMask, p.frameToToken),
      )
    val mu = invoke("acoustic_condition$suffix", conditionInputs)[0] as FloatArray
    AcousticHost.zeroBeyondValid(mu, p.validFrames, p.frames)
    val paddedNoise = p.x.copyOf()
    var x = p.x.copyOf()
    val grid = AcousticHost.timeGrid()
    for (i in 0 until grid.lastIndex) {
      val velocity =
        invoke(
          "acoustic_velocity$suffix",
          listOf(
            x,
            floatArrayOf(grid[i]),
            mu,
            p.condVec,
            p.condMel,
            p.condMask,
            p.keyBias,
          ),
        )[0]
          as FloatArray
      x =
        AcousticHost.eulerUpdate(
          x,
          velocity,
          paddedNoise,
          p.condMel,
          p.condMask,
          grid[i],
          grid[i + 1],
        )
      AcousticHost.zeroBeyondValid(x, p.validFrames, p.frames)
    }
    val solved = FloatArray(100 * p.validFrames)
    for (band in 0 until 100) for (frame in 0 until p.validFrames) {
      solved[band * p.validFrames + frame] =
        if (frame < p.promptFrames) p.condMel[band * p.frames + frame]
        else x[band * p.frames + frame]
    }
    val mel =
      AcousticHost.decodeMel(
        solved,
        p.validFrames,
        p.promptFrames,
        dsp.c.getValue("mel_mean"),
        dsp.c.getValue("mel_std"),
      )
    val melFrames = mel.size / 100
    require(melFrames >= 128) {
      "Streaming contract requires at least 128 mel frames; this generated segment has $melFrames"
    }
    val decoded =
      StreamingPcmDecoder(
          dsp.c.getValue("istft_window"),
          minOf(32, p.promptFrames) * 256,
          tokens.size * 1024,
          reference.levelDb,
          started,
          onPcm,
          playbackProcessing = streaming != null,
          shouldStop = cancellation,
        )
        .run(mel, melFrames) { mode, args ->
          invoke("vocoder_stream_$mode", args).map { it as FloatArray }
        }
    val raw = decoded.raw
    checkActive()
    val postStart = System.nanoTime()
    val post = PostProcess.postprocessSegment(raw, reference.levelDb)
    val postMs = ms(System.nanoTime() - postStart)
    val totalMs = ms(System.nanoTime() - started)
    val stats = streaming?.let { stream ->
      fun calls(prefix: String) =
        stageCalls.filter { it.first.startsWith(prefix) }.map { it.second }
      val seconds = post.wav.size / 24000.0
      SynthesisStats(
        stream.seed,
        stream.referenceId,
        stream.cacheHit,
        placementDescription,
        precision.name.lowercase(),
        decoded.ttfaMs,
        SynthesisStats.onsetMs(decoded.ttfaMs, post.trim.leadCutSamples),
        stream.referencePrepMs,
        stream.prefillMs,
        stream.arStepMs,
        stream.arTotalMs,
        calls("acoustic_condition").sum(),
        calls("acoustic_velocity"),
        calls("vocoder_stream_"),
        decoded.istftMs,
        decoded.playbackProcessingMs,
        postMs,
        totalMs,
        seconds,
        if (seconds > 0.0) totalMs / 1000.0 / seconds else 0.0,
        tokens.size,
        decoded.chunkLengths,
        timings.toList(),
      )
    }
    return Output(
      post.wav,
      raw,
      tokens,
      solved,
      decoded.features,
      logits,
      post.trim,
      timings.toList(),
      totalMs,
      prefixLength,
      stopReason,
      reference,
      stats,
    )
  }

  private inner class Graph(private val name: String, spec: JSONObject) : Closeable {
    private val backend = placement[name] ?: defaultBackend
    private val model: CompiledModel
    private val signatures = linkedMapOf<String, SignatureBuffers>()

    init {
      val options =
        CompiledModel.Options(if (backend == Backend.CPU) Accelerator.CPU else Accelerator.GPU)
      options.cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
      if (backend == Backend.GPU32)
        options.gpuOptions =
          CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
      model =
        CompiledModel.create(File(filesDir, spec.getString("path")).absolutePath, options, null)
      val all = spec.getJSONArray("signatures")
      for (i in 0 until all.length()) {
        val signature = all.getJSONObject(i)
        val key = signature.getString("name")
        signatures[key] =
          SignatureBuffers(signature, model.createInputBuffers(key), model.createOutputBuffers(key))
      }
    }

    fun call(signatureName: String, arrays: List<Any>): List<Any> {
      val sig = signatures.getValue(signatureName)
      require(arrays.size == sig.inputSpec.size)
      val start = System.nanoTime()
      sig.inputSpec.forEachIndexed { bufferIndex, spec ->
        val value = arrays[spec.getString("name").removePrefix("args_").toInt()]
        val elements = spec.getInt("buffer_bytes") / 4
        when (spec.getString("dtype")) {
          "float32" -> {
            require(value is FloatArray && value.size == elements)
            sig.inputs[bufferIndex].writeFloat(value)
          }
          "int32" -> {
            require(value is IntArray && value.size == elements)
            sig.inputs[bufferIndex].writeInt(value)
          }
          else -> error("Unsupported contract input dtype: ${spec.getString("dtype")}")
        }
      }
      val written = System.nanoTime()
      model.run(sig.inputs, sig.outputs, signatureName)
      val ran = System.nanoTime()
      val output = arrayOfNulls<Any>(sig.outputSpec.size)
      sig.outputSpec.forEachIndexed { bufferIndex, spec ->
        output[spec.getString("name").removePrefix("output_").toInt()] =
          when (spec.getString("dtype")) {
            "float32" -> sig.outputs[bufferIndex].readFloat()
            "int32" -> sig.outputs[bufferIndex].readInt()
            else -> error("Unsupported output dtype")
          }
      }
      val read = System.nanoTime()
      timings +=
        CallTiming(
          name,
          signatureName,
          backend.name.lowercase(),
          ms(written - start),
          ms(ran - written),
          ms(read - ran),
          ms(read - start),
        )
      return output.map { requireNotNull(it) }
    }

    override fun close() {
      signatures.values.forEach { s ->
        s.inputs.forEach { it.close() }
        s.outputs.forEach { it.close() }
      }
      signatures.clear()
      model.close()
    }
  }

  private class SignatureBuffers(
    spec: JSONObject,
    val inputs: List<TensorBuffer>,
    val outputs: List<TensorBuffer>,
  ) {
    val inputSpec = specs(spec.getJSONArray("inputs"))
    val outputSpec = specs(spec.getJSONArray("outputs"))

    init {
      check(inputSpec.size == inputs.size && outputSpec.size == outputs.size)
    }

    companion object {
      fun specs(array: JSONArray): List<JSONObject> =
        (0 until array.length())
          .map { array.getJSONObject(it) }
          .sortedBy { it.getInt("buffer_index") }
          .also { list ->
            list.forEachIndexed { index, item -> check(index == item.getInt("buffer_index")) }
          }
    }
  }

  override fun close() {
    graphs.values.forEach { it.close() }
    graphs.clear()
    referenceCache.clear()
    ar.close()
  }

  companion object {
    val requiredGraphNames =
      listOf(
        "speaker_encoder",
        "semantic_encoder",
        "style_prefix",
        "ar_merged",
        "acoustic_condition",
        "acoustic_velocity",
        "acoustic_condition_t4096",
        "acoustic_velocity_t4096",
        "vocoder_stream_start",
        "vocoder_stream_step",
        "vocoder_stream_flush",
      )

    private fun ms(ns: Long) = ns / 1_000_000.0

    private fun concatenate(parts: List<FloatArray>): FloatArray {
      val out = FloatArray(parts.sumOf { it.size })
      var offset = 0
      parts.forEach {
        it.copyInto(out, offset)
        offset += it.size
      }
      return out
    }

    private fun gaussian(size: Int, random: SplittableRandom): FloatArray =
      FloatArray(size) {
        (sqrt(-2.0 * ln(random.nextDouble().coerceAtLeast(java.lang.Double.MIN_NORMAL))) *
            cos(2.0 * PI * random.nextDouble()))
          .toFloat()
      }
  }
}

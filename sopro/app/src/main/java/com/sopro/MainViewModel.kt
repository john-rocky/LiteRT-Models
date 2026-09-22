// SPDX-License-Identifier: Apache-2.0
package com.sopro

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.net.Uri
import android.os.Build
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import java.io.Closeable
import java.io.File
import java.util.UUID
import java.util.concurrent.CancellationException
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject

/** Model calls and reference IO stay confined; PCM writes run on the player's own thread. */
class MainViewModel(private val context: Context) : ViewModel() {
  private val modelDispatcher = Dispatchers.Default.limitedParallelism(1)
  private val modelScope = CoroutineScope(Job() + modelDispatcher)
  private val preferences = context.getSharedPreferences("sopro_options", Context.MODE_PRIVATE)
  private val hardware = Build.HARDWARE
  private val socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else ""
  private var engine: SoproEngine? = null
  private var engineConfiguration: String? = null
  private var launchPlacement: Map<String, SoproEngine.Backend> = emptyMap()
  private var launchPrecision: SoproEngine.Precision? = null
  private var launchContractSet = "r9"
  private var styleVariant = PlacementConfig.StyleVariant.FP32
  private var measurement: UiMeasurementReport? = null
  private var measurementSeed: Long? = null
  private var measurementTapCount = 0L
  private var started = false
  private var reference: FloatArray? = null
  private var referenceId = "cc0_voice_0a67"
  private val cancelled = AtomicBoolean(false)
  @Volatile private var player: StreamingAudioPlayer? = null
  private val mutable =
    MutableStateFlow(
      UiState(
        text = context.getString(R.string.example_en),
        referenceLabel = context.getString(R.string.voice_default),
        status = context.getString(R.string.status_loading_models),
      )
    )
  val uiState: StateFlow<UiState> = mutable.asStateFlow()

  fun start(
    intent: Intent,
    activityStartedNanos: Long = System.nanoTime(),
    activityStartedUnixMs: Long = System.currentTimeMillis(),
  ) {
    if (started) return
    started = true
    if (intent.getBooleanExtra("gate", false)) {
      mutable.update {
        it.copy(busy = true, gateMode = true, status = context.getString(R.string.status_gate))
      }
      modelScope.launch {
        try {
          val backend = SoproEngine.Backend.parse(intent.getStringExtra("accel") ?: "cpu")
          val precision = SoproEngine.Precision.parse(intent.getStringExtra("precision") ?: "fp32")
          val placementJson = JSONObject(intent.getStringExtra("placement") ?: "{}")
          val placement =
            placementJson.keys().asSequence().associateWith {
              SoproEngine.Backend.parse(placementJson.getString(it))
            }
          val report =
            SoproGateEntry.run(
              context,
              backend,
              precision,
              placement,
              intent.getStringExtra("gate_id"),
              intent.getStringExtra("fixture_manifest") ?: "gate/index.json",
              intent.getStringExtra("gate_mode") ?: "teacher",
              intent.getLongExtra("seed", 5000L),
              intent.getStringExtra("gate_config") ?: "{}",
            )
          mutable.update {
            it.copy(
              busy = false,
              status = context.getString(R.string.gate_result, report.status, report.path),
              error = report.error,
            )
          }
        } catch (failure: Exception) {
          fail(failure)
        } catch (failure: LinkageError) {
          fail(failure)
        }
      }
    } else {
      try {
        val placement = JSONObject(intent.getStringExtra("placement") ?: "{}")
        val hasExplicitMap =
          intent.hasExtra("placement") || intent.hasExtra("accel") || intent.hasExtra("precision")
        val selectedMode =
          if (hasExplicitMap) PlacementConfig.Mode.CUSTOM
          else
            runCatching {
                PlacementConfig.Mode.valueOf(preferences.getString("placement_mode", "AUTOMATIC")!!)
              }
              .getOrDefault(PlacementConfig.Mode.AUTOMATIC)
              .let {
                if (it == PlacementConfig.Mode.CUSTOM) PlacementConfig.Mode.AUTOMATIC else it
              }
        launchPlacement =
          if (hasExplicitMap)
            placement.keys().asSequence().associateWith {
              require(it in SoproEngine.requiredGraphNames)
              SoproEngine.Backend.parse(placement.getString(it))
            }
          else PlacementConfig.placement(selectedMode, hardware, socModel)
        launchPrecision =
          intent.getStringExtra("precision")?.let { SoproEngine.Precision.parse(it) }
            ?: if (hasExplicitMap) null else PlacementConfig.precision(selectedMode)
        launchContractSet = intent.getStringExtra("contract_set") ?: "r9"
        require(launchContractSet in listOf("r6", "r9"))
        styleVariant =
          PlacementConfig.StyleVariant.parse(
            intent.getStringExtra("style_variant")
              ?: preferences.getString("style_variant", "FP32")!!
          )
        val backend = SoproEngine.Backend.parse(intent.getStringExtra("accel") ?: "cpu")
        mutable.update {
          it.copy(
            backend = backend,
            placementMode = selectedMode,
            styleVariant = styleVariant,
            deviceHybridAvailable = PlacementConfig.isGalaxyS26(hardware, socModel),
            graphPlacements = placementRows(backend),
          )
        }
        intent.getStringExtra("measurement_tag")?.let { tag ->
          measurement =
            UiMeasurementReport(
              context,
              tag,
              intent.getStringExtra("apk_sha256") ?: "unknown",
              activityStartedNanos,
              activityStartedUnixMs,
            )
          measurementSeed = intent.getLongExtra("seed", 9000L)
        }
        compileForReady(activityStartedNanos)
      } catch (failure: Exception) {
        fail(failure)
      }
    }
  }

  private fun precisionFor(backend: SoproEngine.Backend) =
    launchPrecision
      ?: if (backend == SoproEngine.Backend.CPU) SoproEngine.Precision.SHIP
      else SoproEngine.Precision.WFP16

  private fun engineFor(backend: SoproEngine.Backend): SoproEngine {
    val precision = precisionFor(backend)
    val configuration =
      "$backend/$precision/$launchContractSet/$styleVariant/${launchPlacement.toSortedMap()}"
    if (engine == null || engineConfiguration != configuration) {
      engine?.close()
      engine = null
      engineConfiguration = null
      engine =
        SoproEngine(
          context.filesDir,
          launchPlacement,
          backend,
          precision,
          launchContractSet,
          styleVariant,
        )
      engineConfiguration = configuration
    }
    return requireNotNull(engine)
  }

  private fun placementRows(backend: SoproEngine.Backend) =
    SoproEngine.requiredGraphNames.map {
      GraphPlacement(it, launchPlacement[it] ?: backend)
    }

  private fun compileForReady(startedNanos: Long = System.nanoTime()) {
    cancelled.set(false)
    mutable.update {
      it.copy(
        busy = true,
        modelsReady = false,
        readyMs = null,
        error = null,
        status = context.getString(R.string.status_loading_models),
      )
    }
    modelScope.launch {
      try {
        val backend = mutable.value.backend
        val loaded = engineFor(backend)
        // Creation only: no reference DSP, inference, or synthesis runs before the tap.
        SoproEngine.requiredGraphNames.forEach {
          if (cancelled.get()) throw CancellationException()
          loaded.compileGraph(it)
        }
        val readyMs = (System.nanoTime() - startedNanos) / 1e6
        measurement?.ready(loaded, precisionFor(backend).name.lowercase(), launchContractSet)
        mutable.update {
          it.copy(
            busy = false,
            modelsReady = true,
            readyMs = readyMs,
            placement = loaded.placementDescription,
            graphPlacements = placementRows(backend),
            status = context.getString(R.string.status_ready_ms, readyMs),
          )
        }
      } catch (failure: CancellationException) {
        stopped()
      } catch (failure: Exception) {
        fail(failure)
      } catch (failure: LinkageError) {
        fail(failure)
      }
    }
  }

  private fun editable() = !mutable.value.busy && !mutable.value.gateMode

  fun setText(text: String) {
    if (editable()) mutable.update { it.copy(text = text) }
  }

  fun selectLanguage(language: String) {
    if (!editable()) return
    val example =
      when (language) {
        "pt" -> R.string.example_pt
        "fr" -> R.string.example_fr
        "de" -> R.string.example_de
        else -> R.string.example_en
      }
    mutable.update { it.copy(language = language, text = context.getString(example)) }
  }

  fun selectPlacement(mode: PlacementConfig.Mode) {
    if (!editable() || mode == PlacementConfig.Mode.CUSTOM) return
    launchPlacement = PlacementConfig.placement(mode, hardware, socModel)
    launchPrecision = PlacementConfig.precision(mode)
    preferences.edit().putString("placement_mode", mode.name).apply()
    mutable.update {
      it.copy(
        backend = SoproEngine.Backend.CPU,
        placementMode = mode,
        graphPlacements = placementRows(SoproEngine.Backend.CPU),
      )
    }
    compileForReady()
  }

  fun selectStyleVariant(variant: PlacementConfig.StyleVariant) {
    if (!editable()) return
    styleVariant = variant
    preferences.edit().putString("style_variant", variant.name).apply()
    mutable.update { it.copy(styleVariant = variant) }
    compileForReady()
  }

  fun selectDemo() {
    if (!editable()) return
    reference = null
    referenceId = "cc0_voice_0a67"
    mutable.update {
      it.copy(
        referenceKind = "demo",
        referenceLabel = context.getString(R.string.voice_default),
        referenceSamples = 240000,
        error = null,
        status = context.getString(R.string.status_ready),
      )
    }
  }

  fun pickReference(uri: Uri) {
    if (!editable()) return
    cancelled.set(false)
    mutable.update {
      it.copy(
        busy = true,
        error = null,
        status = context.getString(R.string.status_loading_reference),
      )
    }
    modelScope.launch {
      try {
        try {
          context.contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION,
          )
        } catch (_: SecurityException) {}
        val decoded = ReferenceAudioDecoder.decode(context, uri, cancelled::get)
        reference = decoded.wave
        referenceId = "file_" + UUID.randomUUID()
        mutable.update {
          it.copy(
            busy = false,
            referenceKind = "file",
            referenceLabel = context.getString(R.string.voice_file),
            referenceSamples = decoded.wave.size,
            status =
              context.getString(R.string.status_reference_loaded, decoded.wave.size / 24000.0),
          )
        }
      } catch (failure: CancellationException) {
        stopped()
      } catch (failure: Exception) {
        fail(failure)
      }
    }
  }

  fun permissionDenied() {
    mutable.update { it.copy(error = context.getString(R.string.microphone_permission_denied)) }
  }

  fun recordReference() {
    if (!editable()) return
    if (
      ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) !=
        PackageManager.PERMISSION_GRANTED
    ) {
      permissionDenied()
      return
    }
    cancelled.set(false)
    mutable.update {
      it.copy(
        busy = true,
        recording = true,
        error = null,
        status = context.getString(R.string.status_recording, 0.0),
      )
    }
    modelScope.launch {
      try {
        val wav =
          ReferenceRecorder.recordTenSeconds(cancelled::get) { seconds ->
            mutable.update {
              it.copy(status = context.getString(R.string.status_recording, seconds))
            }
          }
        reference = wav
        referenceId = "record_" + UUID.randomUUID()
        mutable.update {
          it.copy(
            busy = false,
            recording = false,
            referenceKind = "record",
            referenceSamples = wav.size,
            referenceLabel = context.getString(R.string.voice_recorded),
            status = context.getString(R.string.status_reference_loaded, wav.size / 24000.0),
          )
        }
      } catch (failure: CancellationException) {
        stopped()
      } catch (failure: Exception) {
        fail(failure)
      }
    }
  }

  fun synthesize() {
    if (!editable() || !mutable.value.modelsReady) return
    val tapNanos = System.nanoTime()
    val state = mutable.value
    val seed = measurementSeed?.let { it + measurementTapCount++ } ?: System.nanoTime()
    cancelled.set(false)
    mutable.update {
      it.copy(
        busy = true,
        error = null,
        generatedSamples = 0,
        ttfaMs = null,
        ttfaToOnsetMs = null,
        totalWallMs = null,
        rtf = null,
        seed = seed,
        status = context.getString(R.string.status_synthesizing),
      )
    }
    measurement?.tapped(seed)
    modelScope.launch {
      try {
        val activeEngine = engineFor(state.backend)
        val wav =
          reference ?: ReferenceAudioDecoder.decodeBundled(context).wave.also { reference = it }
        val audio = StreamingAudioPlayer { failure ->
          mutable.update { it.copy(error = failure.message) }
        }
        player = audio
        var decodedSamples = 0
        val result =
          activeEngine.synthesizeStreaming(
            state.text,
            state.language,
            wav,
            { chunk ->
              audio.enqueue(chunk)
              decodedSamples += chunk.size
              mutable.update {
                it.copy(
                  generatedSamples = decodedSamples,
                  status = context.getString(R.string.status_decoding, decodedSamples / 24000.0),
                )
              }
            },
            seed,
            tapNanos,
            referenceId,
            cancelled::get,
          )
        if (cancelled.get()) throw CancellationException()
        val stats = requireNotNull(result.stats)
        val saved =
          WavFiles.write(
            File(context.filesDir, "outputs"),
            state.language,
            referenceId,
            result.wav,
            stats
              .toJson()
              .put("trim", JSONObject(result.trim.asMap()))
              .put("contract_set", launchContractSet)
              .put("style_variant", styleVariant.name.lowercase()),
          )
        mutable.update {
          it.copy(
            lastWavPath = saved.wav.absolutePath,
            lastSidecarPath = saved.sidecar.absolutePath,
            generatedSamples = result.wav.size,
            ttfaMs = stats.ttfaMs,
            ttfaToOnsetMs = stats.ttfaToOnsetMs,
            totalWallMs = stats.totalWallMs,
            rtf = stats.rtf,
            placement = stats.placement,
            status = context.getString(R.string.status_draining),
          )
        }
        measurement?.generated(state, result, saved)
        audio.finishAndDrain()
        if (cancelled.get()) throw CancellationException()
        mutable.update {
          it.copy(
            busy = false,
            status =
              context.getString(
                R.string.status_complete,
                result.wav.size / 24000.0,
                result.tokens.size,
              ),
          )
        }
        measurement?.completed()
      } catch (failure: CancellationException) {
        stopped()
      } catch (failure: Exception) {
        fail(failure)
      } catch (failure: LinkageError) {
        fail(failure)
      } finally {
        player?.close()
        player = null
      }
    }
  }

  fun stop() {
    cancelled.set(true)
    player?.stop()
    mutable.update { it.copy(status = context.getString(R.string.status_stopping)) }
  }

  fun playAgain() {
    if (!editable()) return
    val path = mutable.value.lastWavPath ?: return
    cancelled.set(false)
    mutable.update {
      it.copy(busy = true, error = null, status = context.getString(R.string.status_playing))
    }
    modelScope.launch {
      try {
        val wav = WavFiles.readPcm16(File(path))
        val audio = StreamingAudioPlayer()
        player = audio
        var offset = 0
        while (offset < wav.size && !cancelled.get()) {
          val end = minOf(offset + 4096, wav.size)
          audio.enqueue(wav.copyOfRange(offset, end))
          offset = end
        }
        audio.finishAndDrain()
        if (cancelled.get()) stopped()
        else
          mutable.update {
            it.copy(busy = false, status = context.getString(R.string.status_ready))
          }
      } catch (failure: CancellationException) {
        stopped()
      } catch (failure: Exception) {
        fail(failure)
      } finally {
        player?.close()
        player = null
      }
    }
  }

  fun exportLastWav(uri: Uri) {
    val path = mutable.value.lastWavPath ?: return
    modelScope.launch {
      try {
        context.contentResolver.openOutputStream(uri)?.use { output ->
          File(path).inputStream().use { it.copyTo(output) }
        } ?: error(context.getString(R.string.error_save_destination))
        mutable.update { it.copy(status = context.getString(R.string.status_exported)) }
      } catch (failure: Exception) {
        fail(failure)
      }
    }
  }

  private fun stopped() {
    mutable.update {
      it.copy(busy = false, recording = false, status = context.getString(R.string.status_stopped))
    }
  }

  private fun fail(failure: Throwable) {
    measurement?.failure(failure)
    val error =
      if (failure is MissingModelFile) context.getString(R.string.status_missing, failure.filename)
      else failure.message ?: failure.javaClass.simpleName
    mutable.update {
      it.copy(
        busy = false,
        recording = false,
        error = error,
        status =
          context.getString(
            if (failure is MissingModelFile) R.string.status_unavailable else R.string.status_failed
          ),
      )
    }
  }

  override fun onCleared() {
    cancelled.set(true)
    player?.stop()
    modelScope.cancel()
    // The same serial dispatcher closes resources after any in-flight blocking model call.
    CoroutineScope(modelDispatcher).launch {
      try {
        engine?.close()
      } finally {
        engine = null
        player?.close()
        player = null
      }
    }
    super.onCleared()
  }

  companion object {
    fun getFactory(context: Context): ViewModelProvider.Factory =
      object : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
          require(modelClass.isAssignableFrom(MainViewModel::class.java))
          @Suppress("UNCHECKED_CAST")
          return MainViewModel(context.applicationContext) as T
        }
      }
  }
}

/** AudioTrack is created, written, drained and released on its own thread. */
class StreamingAudioPlayer(private val onError: (Throwable) -> Unit = {}) : Closeable {
  private sealed interface Message {
    data class Pcm(val samples: FloatArray) : Message

    data object End : Message
  }

  private val queue = LinkedBlockingQueue<Message>(32)
  private val cancelled = AtomicBoolean(false)
  private val ended = AtomicBoolean(false)
  @Volatile private var failure: Throwable? = null
  private val worker = Thread({ consume() }, "SoproPlayback").apply { start() }

  fun enqueue(samples: FloatArray) {
    check(!ended.get()) { "Playback stream has ended" }
    if (samples.isEmpty()) return
    while (!cancelled.get()) {
      failure?.let { throw IllegalStateException("Audio playback failed", it) }
      if (queue.offer(Message.Pcm(samples.copyOf()), 50, TimeUnit.MILLISECONDS)) return
    }
    throw java.util.concurrent.CancellationException("Playback stopped")
  }

  /** Waits until every queued frame has reached playback, unless Stop was requested. */
  fun finishAndDrain() {
    if (ended.compareAndSet(false, true)) {
      while (!cancelled.get() && !queue.offer(Message.End, 50, TimeUnit.MILLISECONDS)) {
        failure?.let { throw IllegalStateException("Audio playback failed", it) }
      }
    }
    worker.join()
    failure?.let { throw IllegalStateException("Audio playback failed", it) }
  }

  fun stop() {
    cancelled.set(true)
    queue.clear()
    queue.offer(Message.End)
  }

  override fun close() {
    stop()
    if (Thread.currentThread() !== worker) worker.join(3000)
  }

  private fun consume() {
    var track: AudioTrack? = null
    try {
      var framesWritten = 0L
      var first = true
      while (!cancelled.get()) {
        when (val message = queue.poll(100, TimeUnit.MILLISECONDS) ?: continue) {
          Message.End -> break
          is Message.Pcm -> {
            if (track == null) {
              val minimum =
                AudioTrack.getMinBufferSize(
                  24000,
                  AudioFormat.CHANNEL_OUT_MONO,
                  AudioFormat.ENCODING_PCM_FLOAT,
                )
              check(minimum > 0) { "24 kHz float PCM playback is unavailable ($minimum)" }
              track =
                AudioTrack.Builder()
                  .setAudioAttributes(
                    AudioAttributes.Builder()
                      .setUsage(AudioAttributes.USAGE_MEDIA)
                      .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                      .build()
                  )
                  .setAudioFormat(
                    AudioFormat.Builder()
                      .setSampleRate(24000)
                      .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                      .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                      .build()
                  )
                  .setTransferMode(AudioTrack.MODE_STREAM)
                  .setBufferSizeInBytes(maxOf(minimum, 24000 * 4 / 4))
                  .build()
              check(track.state == AudioTrack.STATE_INITIALIZED) {
                "AudioTrack initialization failed"
              }
            }
            if (first) {
              track.play()
              first = false
            }
            var offset = 0
            while (offset < message.samples.size && !cancelled.get()) {
              val count =
                track.write(
                  message.samples,
                  offset,
                  minOf(1024, message.samples.size - offset),
                  AudioTrack.WRITE_BLOCKING,
                )
              check(count > 0) { "AudioTrack write failed ($count)" }
              offset += count
              framesWritten += count
            }
          }
        }
      }
      if (!cancelled.get() && track != null) {
        val deadline = System.nanoTime() + 5_000_000_000L
        while (
          !cancelled.get() && (track.playbackHeadPosition.toLong() and 0xffffffffL) < framesWritten
        ) {
          check(System.nanoTime() < deadline) { "AudioTrack did not drain its queued frames" }
          Thread.sleep(10)
        }
      }
    } catch (error: Throwable) {
      failure = error
      cancelled.set(true)
      onError(error)
    } finally {
      track?.let {
        try {
          it.pause()
          if (cancelled.get()) it.flush()
          it.stop()
        } catch (_: IllegalStateException) {}
        it.release()
      }
    }
  }
}

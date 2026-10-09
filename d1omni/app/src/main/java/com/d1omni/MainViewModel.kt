package com.d1omni

import android.content.Context
import android.graphics.BitmapFactory
import android.os.SystemClock
import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.d1omni.view.InboxItemUi
import com.d1omni.view.InboxUi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * The screen's state: the status and engine lines, the inbox screen once the engine is ready, the
 * presentation of a running (or finished) inbox run.
 */
@Immutable
data class UiState(
  val status: String = "",
  val engine: String = "",
  val error: Boolean = false,
  val ready: Boolean = false,
  val inbox: InboxUi? = null,
  val presentation: D1PresentationUi? = null,
)

/**
 * Owns the engine and runs every graph call on [D1Runtime.dispatcher]. A normal launch loads the
 * tokenizer and the contract, compiles every graph the inbox demo needs (L256, L128, the audio
 * graph T1001, the vision tower and the projector) and makes one untimed pass over the bundled
 * inbox, then shows the inbox screen (`ENGINE_READY`); an autoplay intent (or the Decide button)
 * runs the inbox on the presentation screen ([D1AutoplayRunner]); a gate or timing launch (debug
 * build) runs that protocol into its report instead.
 */
class MainViewModel(private val context: Context) : ViewModel() {
  private val _uiState = MutableStateFlow(UiState())
  val uiState: StateFlow<UiState> = _uiState.asStateFlow()

  private var inbox: D1InboxEngine? = null
  private var info: D1EngineInfo? = null
  private var started = false
  private var running = false
  private var demoLayout: D1DemoLayout? = null
  private val player = D1AudioPlayer()

  /** Held while the engine loads and while an inbox run runs: an autoplay waits for the load. */
  private val engineLock = Mutex()

  /** Starts what [launch] asks for, once per ViewModel; a later launch intent goes to [newIntent]. */
  fun start(launch: D1Launch) {
    if (started) return
    started = true
    val received = SystemClock.elapsedRealtimeNanos()
    when (launch) {
      is D1Launch.Autoplay -> {
        load(launch.backend, launch.precisions)
        autoplay(launch, received)
      }
      is D1Launch.Normal -> load(launch.backend, launch.precisions)
      else -> run(launch)
    }
  }

  /** A launch intent that reached the running activity (`singleTop`). */
  fun newIntent(launch: D1Launch) {
    val received = SystemClock.elapsedRealtimeNanos()
    when (launch) {
      is D1Launch.Autoplay -> {
        if (inbox == null && !engineLock.isLocked) {
          D1Demo.autoplayStart(launch.fixture)
          D1Demo.failed("the inbox engine is not loaded (a gate or timing run, or a failed load): force-stop and launch again")
          return
        }
        autoplay(launch, received)
      }
      is D1Launch.Normal -> Unit
      is D1Launch.Invalid -> {
        D1Demo.failed(launch.reason)
        show(launch.reason, error = true)
      }
      else -> D1Demo.failed("the app is running; force-stop it and launch again")
    }
  }

  private fun run(launch: D1Launch) {
    when (launch) {
      is D1Launch.Invalid -> {
        D1Demo.failed(launch.reason)
        show(launch.reason, error = true)
      }
      is D1Launch.Gate -> diagnostics("Fixture gate") { progress -> D1GateRunner(context).run(launch, progress) }
      is D1Launch.Timing -> diagnostics("Timing") { progress -> D1TimingRunner(context).run(launch, progress) }
      // vision (round 3): the picture gate and timing (debug build)
      is D1Launch.VGate -> diagnostics("Picture gate") { progress -> D1VisionGate(context).gate(launch, progress) }
      is D1Launch.VTiming -> diagnostics("Picture timing") { progress -> D1VisionGate(context).timing(launch, progress) }
      // end vision (round 3)
      is D1Launch.Normal,
      is D1Launch.Autoplay -> Unit
    }
  }

  private fun load(backend: D1Backend, precisions: D1Precisions) {
    show("Loading the tokenizer…")
    viewModelScope.launch {
      engineLock.withLock {
        try {
          val (loaded, warmupMs) =
            withContext(D1Runtime.dispatcher) {
              inbox?.close()
              inbox = null
              val engine = D1InboxEngine.load(context, backend, precisions) { line -> show(line) }
              inbox = engine
              show("Warming up (one untimed pass over the inbox)…")
              val fixture = D1InboxFixture.parse(requireNotNull(D1Bundled.read(context, D1Bundled.FIXTURE)).bytes)
              engine to D1AutoplayRunner.warmup(context, engine, fixture)
            }
          val loadMs = Math.round(loaded.loadMs)
          info = D1EngineInfo(loadMs, warmupMs)
          D1Demo.engineReady(loadMs, warmupMs)
          _uiState.update {
            it.copy(
              status = "Ready",
              engine =
                "${D1InboxText.accelerator(loaded.backends())} · " +
                  D1InboxText.graphsLine(loaded.decide.resident, loaded.decide.audio.resident, vision = true) +
                  " · loaded in %.1f s".format(loadMs / 1000.0),
              error = false,
              ready = true,
              inbox = inboxUi(),
            )
          }
        } catch (failure: Exception) {
          loadFailed(D1Decider.describe(failure))
        } catch (failure: LinkageError) {
          loadFailed("Native runtime: ${D1Decider.describe(failure)}")
        } catch (failure: OutOfMemoryError) {
          loadFailed(D1Decider.describe(failure))
        }
      }
    }
  }

  private fun loadFailed(reason: String) {
    D1Demo.failed(reason)
    show(reason, error = true)
  }

  /** The Decide button: the bundled inbox on the presentation screen, no delay, 1.5 s between items. */
  fun decide() = autoplay(D1Launch.Autoplay(D1Bundled.FIXTURE, 0, D1Launch.DEFAULT_GAP_MS.toLong()), SystemClock.elapsedRealtimeNanos())

  private fun autoplay(launch: D1Launch.Autoplay, receivedNanos: Long) {
    D1Demo.autoplayStart(launch.fixture)
    if (running) {
      D1Demo.failed("busy with an inbox run")
      return
    }
    running = true
    viewModelScope.launch {
      engineLock.withLock {
        try {
          val engine = inbox ?: throw D1AutoplayRunner.Failure("engine not ready: ${uiState.value.status}")
          val engineInfo = requireNotNull(info)
          demoLayout = null
          val runner =
            D1AutoplayRunner(
              context,
              engine,
              engineInfo,
              player,
              show = { ui -> _uiState.update { it.copy(presentation = ui) } },
              update = { change -> _uiState.update { s -> s.copy(presentation = s.presentation?.let(change)) } },
              layout = { demoLayout },
            )
          val file = runner.run(launch.fixture, launch.delayMs, launch.gapMs, receivedNanos)
          D1Demo.autoplayDone(file.path)
        } catch (failure: D1AutoplayRunner.Failure) {
          autoplayFailed(failure.message.orEmpty())
        } catch (failure: Exception) {
          autoplayFailed(D1Decider.describe(failure))
        } catch (failure: LinkageError) {
          autoplayFailed("native runtime ${D1Decider.describe(failure)}")
        } finally {
          running = false
        }
      }
    }
  }

  private fun autoplayFailed(reason: String) {
    D1Demo.failed(reason)
    _uiState.update { state ->
      state.copy(presentation = state.presentation?.copy(failure = reason, running = false), status = reason, error = true)
    }
  }

  /** The presentation screen reports where it drew the pill and the cards. */
  fun onPresentationLayout(layout: D1DemoLayout) {
    demoLayout = layout
  }

  /** Back from the presentation to the inbox screen (not while a run is running). */
  fun leavePresentation() {
    if (running) return
    _uiState.update { it.copy(presentation = null) }
  }

  /** The inbox screen's items, from the bundled fixture. */
  private fun inboxUi(): InboxUi? =
    runCatching {
        val fixture = D1InboxFixture.parse(requireNotNull(D1Bundled.read(context, D1Bundled.FIXTURE)).bytes)
        InboxUi(
          fixture.title,
          fixture.items.map { item ->
            val source = item.mediaFile?.let { D1Bundled.read(context, it) }
            val samples = if (item.kind == D1Kind.AUDIO) source?.let { D1Wav.parse(it.bytes).size } else null
            val count = item.questions.size
            InboxItemUi(
              D1InboxText.icon(item.kind),
              D1InboxText.header(item, samples),
              "$count ${if (count == 1) "question" else "questions"}: ${item.questions.keys.joinToString(", ")}",
              thumbnail =
                if (item.kind == D1Kind.IMAGE && source != null) {
                  BitmapFactory.decodeByteArray(source.bytes, 0, source.bytes.size)?.asImageBitmap()
                } else {
                  null
                },
              message = if (item.kind == D1Kind.TEXT) D1Prompt.serialize(item.state) else null,
            )
          },
        )
      }
      .getOrNull()

  private fun diagnostics(title: String, body: ((String) -> Unit) -> D1GateRunner.Summary) {
    show("$title…")
    viewModelScope.launch {
      val summary = withContext(D1Runtime.dispatcher) { body { line -> show("$title: $line") } }
      show("$title: ${summary.status} · ${summary.path}", error = summary.error != null)
    }
  }

  private fun show(status: String, error: Boolean = false) {
    _uiState.update { it.copy(status = status, error = error) }
  }

  override fun onCleared() {
    player.close()
    val current = inbox
    inbox = null
    if (current != null) {
      viewModelScope.launch(D1Runtime.dispatcher) { current.close() }
    }
  }

  companion object {
    fun getFactory(context: Context): ViewModelProvider.Factory = viewModelFactory {
      initializer { MainViewModel(context.applicationContext) }
    }
  }
}

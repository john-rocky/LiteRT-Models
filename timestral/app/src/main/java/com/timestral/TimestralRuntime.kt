package com.timestral

import android.content.Context
import android.os.Process
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher

/**
 * One LiteRT Environment and one thread for the whole process: every CompiledModel call (create,
 * run, close) runs on [dispatcher], because the graph reuses its native input and output buffers.
 */
object TimestralRuntime {
  private val executor = Executors.newSingleThreadExecutor { runnable ->
    Thread(runnable, "Timestral-LiteRT").apply { isDaemon = true }
  }

  /** The single worker thread as a coroutine dispatcher. */
  val dispatcher: CoroutineDispatcher = executor.asCoroutineDispatcher()

  private var environment: Environment? = null

  /**
   * The process Environment. The Context overload gives LiteRT the app's cache directory for its
   * NPU JIT compilation cache; with the NPU libraries in the APK it also names their directory for
   * the dispatch library and the compiler plugin (without the plugin, a graph asked for on the NPU
   * is not compiled for it).
   */
  fun environment(context: Context): Environment =
    environment
      ?: context.applicationContext
        .let { app ->
          val options =
            if (NpuSupport.librariesInstalled(app)) {
              val directory = app.applicationInfo.nativeLibraryDir
              mapOf(
                Environment.Option.DispatchLibraryDir to directory,
                Environment.Option.CompilerPluginLibraryDir to directory,
              )
            } else {
              emptyMap()
            }
          Environment.create(app, options)
        }
        .also { environment = it }
}

/** The Qualcomm NPU (HTP) libraries in the APK (`scripts/fetch_npu_libs.sh` copies them). */
object NpuSupport {
  /**
   * The dispatch library, the JIT compiler plugin and the QNN HTP backend: without all three a
   * graph asked for on the NPU never reaches it.
   */
  val REQUIRED_LIBRARIES =
    listOf("libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so", "libQnnHtp.so")

  fun librariesInstalled(context: Context): Boolean {
    val directory = File(context.applicationInfo.nativeLibraryDir)
    return REQUIRED_LIBRARIES.all { File(directory, it).isFile }
  }

  /**
   * The Qualcomm options every graph of the process gets when the libraries are present: LiteRT
   * starts the HTP runtime once per process, from the options of the first graph it compiles on any
   * accelerator, so a CPU or GPU graph compiled first without BURST would leave the NPU graphs
   * after it several times slower.
   */
  fun options(): CompiledModel.QualcommOptions =
    CompiledModel.QualcommOptions(
      htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
    )
}

/**
 * Where an NPU compile put the graph, from this process's log (tags `tflite` and `litert`): LiteRT
 * runs a graph the compiler plugin did not take on the CPU without an error, so the app reads
 * LiteRT's "Replacing N out of M node(s) with delegate (DispatchDelegate)" line itself.
 */
class NpuEvidence(
  /** The matching lines, or null when the log could not be read. */
  val lines: List<String>?,
  /** Why the log could not be read. */
  val error: String? = null,
) {
  /** N and M of the DispatchDelegate line, or null when there is none. */
  val dispatched: Pair<Int, Int>?
    get() =
      lines
        ?.firstNotNullOfOrNull { DISPATCH.find(it) }
        ?.let { it.groupValues[1].toInt() to it.groupValues[2].toInt() }

  /** Whether a line shows the CPU took the graph (XNNPACK replaced its nodes, or the plugin failed). */
  val cpuTookOver: Boolean
    get() = lines.orEmpty().any { line -> FALLBACK.any { it in line } }

  /** Whether the JIT compilation came from LiteRT's cache. */
  val fromCache: Boolean
    get() = lines.orEmpty().any { CACHED in it }

  companion object {
    private val DISPATCH =
      Regex("""Replacing (\d+) out of (\d+) node\(s\) with delegate \(DispatchDelegate\)""")
    private const val CACHED = "Flatbuffer model initialized from cached model"
    private val FALLBACK =
      listOf(
        "with delegate (TfLiteXNNPackDelegate)",
        "Failed to apply compiler plugins",
        "Failed to load compiler plugins",
      )
    private val KEEP =
      listOf(
        "NPU accelerator",
        "Replacing ",
        "Partitioned subgraph",
        CACHED,
        "compiler plugins",
        "is not supported in Qualcomm Compiler",
      )

    /** The lines of [log] that tell where a graph went. */
    fun of(log: List<String>): NpuEvidence = NpuEvidence(log.filter { line -> KEEP.any { it in line } })
  }
}

/** Reads this process's own log lines (no permission needed for them) after an NPU compile. */
object NpuLog {
  private val LOGCAT_TIME = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

  /** The tagged lines of this process since [sinceMs] (wall clock). */
  fun since(sinceMs: Long): NpuEvidence =
    try {
      val command =
        listOf(
          "logcat",
          "-d",
          "-v",
          "threadtime",
          "-T",
          synchronized(LOGCAT_TIME) { LOGCAT_TIME.format(Date(sinceMs)) },
          "--pid=${Process.myPid()}",
          "-s",
          "tflite:V",
          "litert:V",
        )
      val process = ProcessBuilder(command).redirectErrorStream(true).start()
      val lines = process.inputStream.bufferedReader().use { it.readLines() }
      process.waitFor()
      NpuEvidence.of(lines)
    } catch (failure: Exception) {
      NpuEvidence(null, "${failure.javaClass.simpleName}: ${failure.message.orEmpty()}")
    }
}

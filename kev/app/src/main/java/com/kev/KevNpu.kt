package com.kev

import android.content.Context
import android.os.Build
import android.os.Process
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The Qualcomm NPU (HTP) runtime as this app uses it: whether the APK carries the libraries, where
 * LiteRT keeps a graph's JIT compilation, and the marks the app keeps of the graphs it compiled.
 */
object KevNpu {
  /**
   * The dispatch library, the JIT compiler plugin and the QNN HTP backend: without all three a
   * graph asked for on the NPU never reaches it. `scripts/fetch_npu_libs.sh` copies them (and the
   * other QAIRT libraries) into the APK.
   */
  val REQUIRED_LIBRARIES =
    listOf("libLiteRtDispatch_Qualcomm.so", "libLiteRtCompilerPlugin_Qualcomm.so", "libQnnHtp.so")

  /** The directory of the marks in `files/`, one file per graph file. */
  const val MARKS_DIR = "npu_marks"

  fun librariesInstalled(context: Context): Boolean =
    librariesIn(File(context.applicationInfo.nativeLibraryDir))

  fun librariesIn(directory: File): Boolean = REQUIRED_LIBRARIES.all { File(directory, it).isFile }

  /**
   * Where LiteRT 2.2.0 keeps the JIT compilation of [graphFile] under the compiler cache directory
   * (the app's `cacheDir`): `<file name without .tflite>/<hash of the file's bytes>/<hash of the
   * plugin, the Android build and the accelerators>.tflite`
   * (`litert/core/cache/compilation_cache.cc`).
   */
  fun cacheDirectory(cacheDir: File, graphFile: String): File =
    File(cacheDir, graphFile.removeSuffix(".tflite"))
}

/**
 * The Qualcomm options of a process, present only when the APK carries the NPU libraries. Every
 * graph gets [performance]: LiteRT 2.2.0 starts the HTP runtime once per process, from the options
 * of the first graph it compiles on any accelerator, so a GPU graph compiled first would otherwise
 * set the mode of the NPU graphs after it. The graphs compiled on the NPU also get [optimization].
 */
data class KevNpuOptions(
  /** null: no mode in the options (the debug `npu_perf none`). */
  val performance: CompiledModel.QualcommOptions.HtpPerformanceMode? =
    CompiledModel.QualcommOptions.HtpPerformanceMode.BURST,
  /** null: LiteRT's default level. */
  val optimization: CompiledModel.QualcommOptions.OptimizationLevel? = null,
) {
  /** The options of a graph that compiles on the NPU ([npuGraph]) or not; null for none. */
  fun qualcommOptions(npuGraph: Boolean): CompiledModel.QualcommOptions? {
    val level = if (npuGraph) optimization else null
    if (performance == null && level == null) return null
    return CompiledModel.QualcommOptions(
      htpPerformanceMode = performance,
      optimizationLevel = level,
    )
  }

  /** The `npu_perf` value of these options. */
  val performanceName: String
    get() = performance?.name?.lowercase(Locale.ROOT) ?: NONE

  /** The `npu_opt` value of these options (the marks record it). */
  val optimizationName: String
    get() = OPTIMIZATIONS.entries.firstOrNull { it.value == optimization }?.key ?: DEFAULT

  companion object {
    const val NONE = "none"
    const val DEFAULT = "default"

    private val OPTIMIZATIONS =
      linkedMapOf(
        DEFAULT to null,
        "inference" to CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_INFERENCE,
        "o3" to CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_INFERENCE_O3,
        "prepare" to CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_PREPARE,
      )

    /**
     * The debug extras `npu_perf` (a mode name such as `burst`, or `none`; default burst) and
     * `npu_opt` (`default`, `inference`, `o3` or `prepare`); null when either is not one of them.
     */
    fun parse(performance: String?, optimization: String?): KevNpuOptions? {
      val mode =
        when (val name = performance?.trim()?.lowercase(Locale.ROOT)) {
          null -> CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
          NONE -> null
          else ->
            CompiledModel.QualcommOptions.HtpPerformanceMode.entries.firstOrNull {
              it.name.lowercase(Locale.ROOT) == name
            } ?: return null
        }
      val levelName = optimization?.trim()?.lowercase(Locale.ROOT) ?: DEFAULT
      if (levelName !in OPTIMIZATIONS) return null
      return KevNpuOptions(mode, OPTIMIZATIONS[levelName])
    }
  }
}

/**
 * What one NPU compile of a graph file was: the file (name, bytes, last modified), the runtime
 * (LiteRT version, Android build fingerprint) and the optimization level. The app writes it after a
 * compile the HTP took and compares it before the next one ([KevNpuMarks.state]).
 */
data class KevNpuMark(
  val graphFile: String,
  val bytes: Long,
  val lastModified: Long,
  val litert: String,
  val build: String,
  val optimization: String,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "graph_file" to graphFile,
      "bytes" to bytes,
      "last_modified_ms" to lastModified,
      "litert" to litert,
      "build_fingerprint" to build,
      "optimization" to optimization,
    )

  companion object {
    /** The mark of [file] compiled now with [optimization]. */
    fun of(file: File, build: String, optimization: String): KevNpuMark =
      KevNpuMark(
        file.name,
        file.length(),
        file.lastModified(),
        KevDecider.LITERT_VERSION,
        build,
        optimization,
      )

    /** A mark file's content, or null when it is not one. */
    fun parse(text: String): KevNpuMark? = runCatching {
      val map = KevJson.parse(text) as Map<*, *>
      fun long(key: String) = (map[key] as JsonNumber).literal.toLong()
      KevNpuMark(
        map["graph_file"] as String,
        long("bytes"),
        long("last_modified_ms"),
        map["litert"] as String,
        map["build_fingerprint"] as String,
        map["optimization"] as String,
      )
    }
      .getOrNull()
  }
}

/** How the JIT cache of a graph stands before an NPU compile, as the marks tell it. */
enum class KevNpuCacheState(val wireName: String) {
  /** No mark: the app never compiled this graph file on the NPU. */
  FIRST("first"),

  /** The mark names other bytes or another modification time: the file was replaced. */
  CHANGED_FILE("changed_file"),

  /** Another LiteRT version or Android build compiled it: LiteRT's key has changed. */
  CHANGED_RUNTIME("changed_runtime"),

  /** The mark is of another optimization level, which LiteRT keys as another compilation. */
  CHANGED_OPTIONS("changed_options"),

  /** The mark matches, but the graph's cache directory is gone (cleared by the user or Android). */
  CACHE_MISSING("cache_missing"),

  /** The mark matches and the cache directory is there: LiteRT loads the compilation. */
  CACHED("cached");

  /** The compile takes minutes and runs alone (everything else closed first). */
  val firstCompile: Boolean
    get() = this != CACHED
}

/** The marks in `files/npu_marks/` and what they say before a compile. Android-free. */
object KevNpuMarks {
  fun file(marksDir: File, graphFile: String): File = File(marksDir, "$graphFile.json")

  fun read(marksDir: File, graphFile: String): KevNpuMark? =
    file(marksDir, graphFile).takeIf { it.isFile }?.let { KevNpuMark.parse(it.readText()) }

  fun write(marksDir: File, mark: KevNpuMark) {
    marksDir.mkdirs()
    val target = file(marksDir, mark.graphFile)
    val temporary = File(marksDir, "${target.name}.tmp")
    temporary.writeText(KevJson.writeIndented(mark.toJson(), 1) + "\n")
    check(temporary.renameTo(target)) { "Could not save ${target.absolutePath}" }
  }

  fun delete(marksDir: File, graphFile: String) {
    file(marksDir, graphFile).delete()
  }

  /**
   * The state of the graph whose compile now would be [current], given its [mark] and whether its
   * cache directory exists ([cached]).
   */
  fun state(mark: KevNpuMark?, current: KevNpuMark, cached: Boolean): KevNpuCacheState =
    when {
      mark == null -> KevNpuCacheState.FIRST
      mark.graphFile != current.graphFile ||
        mark.bytes != current.bytes ||
        mark.lastModified != current.lastModified -> KevNpuCacheState.CHANGED_FILE
      mark.litert != current.litert || mark.build != current.build ->
        KevNpuCacheState.CHANGED_RUNTIME
      mark.optimization != current.optimization -> KevNpuCacheState.CHANGED_OPTIONS
      !cached -> KevNpuCacheState.CACHE_MISSING
      else -> KevNpuCacheState.CACHED
    }

  /**
   * Whether the app deletes the graph's cache directory before compiling. LiteRT 2.2.0 keys a
   * compilation by the file's bytes, the compiler plugin, the Android build, the accelerators and
   * the Qualcomm options, so another level compiles anew; the app drops the old level's entry.
   */
  fun clearsCache(state: KevNpuCacheState): Boolean = state == KevNpuCacheState.CHANGED_OPTIONS
}

/**
 * The log lines of one NPU compile, from this process's log (tags `tflite` and `litert`): whether
 * the HTP took the graph ([applied]: LiteRT's "Replacing … with delegate (DispatchDelegate)" line;
 * LiteRT runs a graph the plugin fails on the CPU without an error) and whether it came from the
 * JIT cache ([fromCache]). Android-free.
 */
class KevNpuEvidence(
  /** The matching lines, or null when the log could not be read. */
  val lines: List<String>?,
  /** Why the log could not be read. */
  val error: String? = null,
) {
  /**
   * true: the DispatchDelegate line; false: a line that shows the CPU took the graph instead (the
   * XNNPACK delegate replacing its nodes, or the plugin failing); null: neither, or no log.
   */
  val applied: Boolean?
    get() {
      val found = lines ?: return null
      if (found.any { DISPATCH.containsMatchIn(it) }) return true
      if (found.any { line -> FALLBACK.any { it in line } }) return false
      return null
    }

  val fromCache: Boolean?
    get() = lines?.let { found -> found.any { CACHED in it } }

  /** "2 of 3": the nodes the DispatchDelegate replaced, from its line. */
  val dispatchNodes: String?
    get() =
      lines
        ?.firstNotNullOfOrNull { DISPATCH.find(it) }
        ?.let { "${it.groupValues[1]} of ${it.groupValues[2]}" }

  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "applied" to applied,
      "from_cache" to fromCache,
      "dispatch_nodes" to dispatchNodes,
      "lines" to lines,
      "error" to error,
    )

  companion object {
    private val DISPATCH =
      Regex("""Replacing (\d+) out of (\d+) node\(s\) with delegate \(DispatchDelegate\)""")
    private const val CACHED = "Flatbuffer model initialized from cached model"

    /** Lines that show the CPU took a graph asked for on the NPU. */
    private val FALLBACK =
      listOf(
        "with delegate (TfLiteXNNPackDelegate)",
        "Failed to apply compiler plugins",
        "Failed to load compiler plugins",
      )

    /** The lines that tell where a graph went and how it compiled. */
    private val KEEP =
      listOf(
        "NPU accelerator",
        "Replacing ",
        "Partitioned subgraph",
        CACHED,
        "NPU JIT compilation caching",
        "compiler plugins were applied",
        "Failed to apply compiler plugins",
        "Failed to load compiler plugins",
        "HtpPerformanceMode",
        "is not supported in Qualcomm Compiler",
        "Failed to load model from cache",
        "JIT compilation changed model",
      )

    /** The lines of [log] that [KEEP] names. */
    fun of(log: List<String>): KevNpuEvidence =
      KevNpuEvidence(log.filter { line -> KEEP.any { it in line } })
  }
}

/** Reads this process's own log (no permission needed for its own lines) after an NPU compile. */
object KevNpuLog {
  /** The time format of `logcat -T`, in the phone's time zone. */
  private val LOGCAT_TIME = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

  /**
   * The tagged lines of this process since [sinceMs] (wall clock), filtered by [KevNpuEvidence].
   */
  fun since(sinceMs: Long): KevNpuEvidence =
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
      KevNpuEvidence.of(lines)
    } catch (failure: Exception) {
      KevNpuEvidence(null, KevDecider.describe(failure))
    }
}

/**
 * One NPU compile of a graph file as the reports describe it: the cache state the marks gave before
 * it ([state]), whether the app deleted the graph's cache first ([cacheCleared]), the log lines
 * ([evidence]), the files of the app's cache directory before and after (relative path to bytes),
 * and the Qualcomm options.
 */
class KevNpuCompile(
  val state: KevNpuCacheState,
  val cacheCleared: Boolean,
  val evidence: KevNpuEvidence,
  val cacheBefore: Map<String, Long>,
  val cacheAfter: Map<String, Long>,
  val performance: String,
  val optimization: String,
  val compileMs: Double,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "cache_state" to state.wireName,
      "first_compile" to state.firstCompile,
      "cache_cleared" to cacheCleared,
      "performance" to performance,
      "optimization" to optimization,
      "compile_ms" to compileMs,
      "evidence" to evidence.toJson(),
      "cache_files_before" to cacheBefore,
      "cache_files_after" to cacheAfter,
    )
}

/** The steps around an NPU compile: the marks and the cache before, the log and the mark after. */
object KevNpuCompiler {
  /** What [before] saw and did, for [after]. */
  class Before(
    val file: File,
    val current: KevNpuMark,
    val state: KevNpuCacheState,
    val cacheCleared: Boolean,
    val sinceMs: Long,
    val cacheBefore: Map<String, Long>,
    val options: KevNpuOptions,
  )

  /** The cache state of [file] compiled now on the NPU with [options] (the default when null). */
  fun state(context: Context, file: File, options: KevNpuOptions?): KevNpuCacheState {
    val current =
      KevNpuMark.of(file, Build.FINGERPRINT, (options ?: KevNpuOptions()).optimizationName)
    val marks = File(context.filesDir, KevNpu.MARKS_DIR)
    return KevNpuMarks.state(
      KevNpuMarks.read(marks, file.name),
      current,
      hasFiles(KevNpu.cacheDirectory(context.cacheDir, file.name)),
    )
  }

  /**
   * Reads the mark of [file] and, when the options changed ([KevNpuMarks.clearsCache]), deletes its
   * cache directory and mark; records the cache files and the time the compile starts.
   */
  fun before(context: Context, file: File, options: KevNpuOptions?): Before {
    val npu = options ?: KevNpuOptions()
    val current = KevNpuMark.of(file, Build.FINGERPRINT, npu.optimizationName)
    val marks = File(context.filesDir, KevNpu.MARKS_DIR)
    val directory = KevNpu.cacheDirectory(context.cacheDir, file.name)
    val state = KevNpuMarks.state(KevNpuMarks.read(marks, file.name), current, hasFiles(directory))
    val clear = KevNpuMarks.clearsCache(state)
    if (clear) {
      directory.deleteRecursively()
      KevNpuMarks.delete(marks, file.name)
    }
    return Before(
      file,
      current,
      state,
      clear,
      System.currentTimeMillis(),
      cacheFiles(context.cacheDir),
      npu,
    )
  }

  /**
   * The log lines since the compile started, the cache files after it, and the mark: written when
   * the HTP took the graph (or the log could not be read), deleted when it did not.
   */
  fun after(context: Context, before: Before, compileMs: Double): KevNpuCompile {
    val evidence = KevNpuLog.since(before.sinceMs)
    val marks = File(context.filesDir, KevNpu.MARKS_DIR)
    if (evidence.applied == false) {
      KevNpuMarks.delete(marks, before.file.name)
    } else {
      KevNpuMarks.write(marks, before.current)
    }
    return KevNpuCompile(
      before.state,
      before.cacheCleared,
      evidence,
      before.cacheBefore,
      cacheFiles(context.cacheDir),
      before.options.performanceName,
      before.options.optimizationName,
      compileMs,
    )
  }

  /** Every file under [directory]: its path relative to it, and its bytes (at most 64 files). */
  fun cacheFiles(directory: File): Map<String, Long> =
    directory
      .walkTopDown()
      .filter { it.isFile }
      .sortedBy { it.path }
      .take(MAX_LISTED)
      .associate { it.relativeTo(directory).path to it.length() }

  private fun hasFiles(directory: File): Boolean = directory.walkTopDown().any { it.isFile }

  private const val MAX_LISTED = 64
}

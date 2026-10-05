package com.kev

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import kotlin.io.path.createTempDirectory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The NPU backend without a phone: which graphs it takes, the options of every graph with and
 * without the NPU libraries, the Environment options, the marks of compiled graphs and the log
 * lines that tell where a graph ran. Only option objects are built; no graph is compiled.
 */
class KevNpuTest {
  @Test
  fun theNpuTakesTheWindowsUpToL256AndTheGpuTheOtherGraphs() {
    val npu = KevDecider.Backend.NPU
    assertEquals(listOf(64, 128, 256), KevFiles.NPU_WINDOWS)
    for (window in KevFiles.WINDOWS) {
      val expected = if (window <= 256) npu else KevDecider.Backend.GPU
      assertEquals(
        "L$window",
        expected,
        KevDecider.Backend.forGraph(npu, KevGraphKey.Window(window)),
      )
    }
    for (shape in KevFiles.PAIRS) {
      assertEquals(
        KevDecider.Backend.GPU,
        KevDecider.Backend.forGraph(npu, KevGraphKey.Pair(shape)),
      )
    }
    // The other choices keep every graph where they are.
    for (backend in listOf(KevDecider.Backend.GPU, KevDecider.Backend.CPU)) {
      assertEquals(backend, KevDecider.Backend.forGraph(backend, KevGraphKey.Window(128)))
      assertEquals(
        backend,
        KevDecider.Backend.forGraph(backend, KevGraphKey.Pair(KevFiles.PAIRS[0])),
      )
    }
    // NPU with the CPU (LiteRT 2.2.0 does not compile these graphs for the NPU alone).
    assertEquals(listOf(Accelerator.NPU, Accelerator.CPU), npu.accelerators)
    assertEquals(listOf(Accelerator.GPU), KevDecider.Backend.GPU.accelerators)
    assertEquals(listOf(Accelerator.CPU), KevDecider.Backend.CPU.accelerators)
    assertEquals(npu, KevDecider.Backend.of(" NPU "))
    assertNull(KevDecider.Backend.of("htp"))
  }

  @Test
  fun everyGraphGetsBurstWhenTheApkCarriesTheNpuLibraries() {
    val burst = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
    val npu = KevNpuOptions()
    val onNpu = KevDecider.options(KevDecider.Backend.NPU, KevPrecision.FP16_FP32_ACCUM, npu = npu)
    assertEquals(burst, onNpu.qualcommOptions?.htpPerformanceMode)
    assertNull(onNpu.qualcommOptions?.optimizationLevel)
    assertNull(onNpu.gpuOptions)
    assertNull(onNpu.cpuOptions)
    // A GPU or CPU graph compiled first sets the HTP mode of the process, so they carry it too.
    val gpu = KevDecider.options(KevDecider.Backend.GPU, KevPrecision.FP16_FP32_ACCUM, false, npu)
    assertEquals(burst, gpu.qualcommOptions?.htpPerformanceMode)
    assertEquals(false, gpu.gpuOptions?.constantTensorSharing)
    assertEquals(
      CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM,
      gpu.gpuOptions?.precision,
    )
    val cpu = KevDecider.options(KevDecider.Backend.CPU, KevPrecision.FP32, npu = npu)
    assertEquals(burst, cpu.qualcommOptions?.htpPerformanceMode)
    assertEquals(KevDecider.CPU_THREADS, cpu.cpuOptions?.numThreads)
    // Without the libraries the options are those of the app before the NPU backend.
    val plain = KevDecider.options(KevDecider.Backend.GPU, KevPrecision.FP32)
    assertNull(plain.qualcommOptions)
    assertEquals(CompiledModel.GpuOptions.Precision.FP32, plain.gpuOptions?.precision)
    assertNull(KevDecider.options(KevDecider.Backend.CPU, KevPrecision.FP32).qualcommOptions)
    // The optimization level goes to the NPU graphs only.
    val o3 = KevNpuOptions.parse(null, "o3")!!
    assertEquals(
      CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_INFERENCE_O3,
      KevDecider.options(KevDecider.Backend.NPU, KevPrecision.FP32, npu = o3)
        .qualcommOptions
        ?.optimizationLevel,
    )
    assertNull(
      KevDecider.options(KevDecider.Backend.GPU, KevPrecision.FP32, npu = o3)
        .qualcommOptions
        ?.optimizationLevel
    )
    // npu_perf none: no Qualcomm options at all where nothing else is set.
    val none = KevNpuOptions.parse("none", null)!!
    assertNull(none.performance)
    assertNull(
      KevDecider.options(KevDecider.Backend.NPU, KevPrecision.FP32, npu = none).qualcommOptions
    )
  }

  @Test
  fun theDebugExtrasNameAModeAndALevel() {
    assertEquals(KevNpuOptions(), KevNpuOptions.parse(null, null))
    assertEquals("burst", KevNpuOptions().performanceName)
    assertEquals("default", KevNpuOptions().optimizationName)
    val prepare = KevNpuOptions.parse("burst", "prepare")!!
    assertEquals("prepare", prepare.optimizationName)
    assertEquals(
      CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_PREPARE,
      prepare.optimization,
    )
    assertEquals(
      CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE,
      KevNpuOptions.parse("high_performance", null)!!.performance,
    )
    assertEquals("none", KevNpuOptions.parse("none", "default")!!.performanceName)
    assertNull(KevNpuOptions.parse("fast", null))
    assertNull(KevNpuOptions.parse(null, "o2"))
  }

  @Test
  fun theEnvironmentNamesTheLibraryDirectoriesOnlyWithTheLibraries() {
    val dir = "/data/app/com.kev/lib/arm64"
    assertEquals(
      mapOf(
        Environment.Option.DispatchLibraryDir to dir,
        Environment.Option.CompilerPluginLibraryDir to dir,
      ),
      KevRuntime.environmentOptions(dir, npuLibraries = true),
    )
    assertTrue(KevRuntime.environmentOptions(dir, npuLibraries = false).isEmpty())
    val lib = createTempDirectory("kev-lib").toFile()
    assertFalse(KevNpu.librariesIn(lib))
    KevNpu.REQUIRED_LIBRARIES.dropLast(1).forEach { File(lib, it).writeText("x") }
    assertFalse(KevNpu.librariesIn(lib))
    File(lib, KevNpu.REQUIRED_LIBRARIES.last()).writeText("x")
    assertTrue(KevNpu.librariesIn(lib))
    lib.deleteRecursively()
  }

  @Test
  fun theMarksTellAFirstCompileFromACachedOne() {
    val marks = createTempDirectory("kev-marks").toFile()
    val name = KevFiles.graph(128)
    val mark = KevNpuMark(name, 1_258_444_912L, 1_759_600_000_000L, "2.2.0", "build-a", "default")
    assertNull(KevNpuMarks.read(marks, name))
    KevNpuMarks.write(marks, mark)
    assertEquals(mark, KevNpuMarks.read(marks, name))
    assertEquals(
      File(File("/cache"), "kev-0.8b_rowprefill_L128_fp16fc_i8emb"),
      KevNpu.cacheDirectory(File("/cache"), name),
    )
    val states =
      listOf(
        KevNpuCacheState.FIRST to KevNpuMarks.state(null, mark, cached = false),
        KevNpuCacheState.CACHED to KevNpuMarks.state(mark, mark, cached = true),
        KevNpuCacheState.CACHE_MISSING to KevNpuMarks.state(mark, mark, cached = false),
        KevNpuCacheState.CHANGED_FILE to
          KevNpuMarks.state(mark, mark.copy(lastModified = mark.lastModified + 1), true),
        KevNpuCacheState.CHANGED_FILE to KevNpuMarks.state(mark, mark.copy(bytes = 1L), true),
        KevNpuCacheState.CHANGED_RUNTIME to KevNpuMarks.state(mark, mark.copy(build = "b"), true),
        KevNpuCacheState.CHANGED_RUNTIME to
          KevNpuMarks.state(mark, mark.copy(litert = "2.3.0"), true),
        KevNpuCacheState.CHANGED_OPTIONS to
          KevNpuMarks.state(mark, mark.copy(optimization = "o3"), true),
      )
    for ((expected, actual) in states) assertEquals(expected, actual)
    // Only a cached graph loads in seconds; the rest compile alone, for minutes.
    assertFalse(KevNpuCacheState.CACHED.firstCompile)
    assertTrue(
      KevNpuCacheState.entries.filter { it != KevNpuCacheState.CACHED }.all { it.firstCompile }
    )
    // LiteRT's key holds the file's bytes and the Qualcomm options: the app deletes the cache of a
    // graph compiled at another optimization level, and only then.
    assertEquals(
      listOf(KevNpuCacheState.CHANGED_OPTIONS),
      KevNpuCacheState.entries.filter { KevNpuMarks.clearsCache(it) },
    )
    KevNpuMarks.delete(marks, name)
    assertNull(KevNpuMarks.read(marks, name))
    assertNull(KevNpuMark.parse("{\"graph_file\": 1}"))
    marks.deleteRecursively()
  }

  @Test
  fun theLogLinesTellWhetherTheHtpTookTheGraph() {
    // The conversion lane's S26 runner logs of the same file (LiteRT 2.2.0): a first compile, then
    // a load from the JIT cache.
    val first =
      KevNpuEvidence.of(
        listOf(
          "10-05 01:27:33.110 18209 18209 I litert  : [npu_registry.cc:30] NPU accelerator registered.",
          "10-05 01:27:33.437 18209 18209 I litert  : [qnn_manager.cc:175] Loading qnn system shared library from \"libQnnSystem.so\"",
          "10-05 01:27:33.429 18209 18209 I litert  :   HtpPerformanceMode       : Burst(2)",
          "10-05 01:27:33.918 18209 18209 I litert  : [compiler_plugin.cc:741] Partitioned subgraph<0>, selected 3911 ops, from a total of 3912 ops. resulted in 2 partitions.",
          "10-05 01:28:28.710 18209 18209 I tflite  : Replacing 2 out of 3 node(s) with delegate (DispatchDelegate) node, yielding 3 partitions for subgraph 0 (main).",
        )
      )
    assertEquals(true, first.applied)
    assertEquals(false, first.fromCache)
    assertEquals("2 of 3", first.dispatchNodes)
    assertEquals(4, first.lines!!.size)
    val cached =
      KevNpuEvidence.of(
        listOf(
          "10-05 01:41:22.349 23361 23361 I litert  : [compiled_model.cc:1233] Flatbuffer model initialized from cached model.",
          "10-05 01:41:22.459 23361 23361 I tflite  : Replacing 2 out of 3 node(s) with delegate (DispatchDelegate) node, yielding 3 partitions for subgraph 0 (main).",
        )
      )
    assertEquals(true, cached.applied)
    assertEquals(true, cached.fromCache)
    // A plugin that failed: LiteRT runs the graph on the CPU without an error.
    val cpu =
      KevNpuEvidence.of(
        listOf(
          "W litert  : Failed to apply compiler plugins: kLiteRtStatusErrorRuntimeFailure",
          "I tflite  : Replacing 3912 out of 3912 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for subgraph 0 (main).",
        )
      )
    assertEquals(false, cpu.applied)
    assertNull(cpu.dispatchNodes)
    // No log, or a log without either kind of line: unknown (the app then shows the NPU it asked
    // for and keeps the mark).
    assertNull(KevNpuEvidence(null, "IOException").applied)
    assertNull(KevNpuEvidence.of(emptyList()).applied)
    assertNull(
      KevNpuEvidence.of(listOf("I tflite  : Created TensorFlow Lite XNNPACK delegate for CPU."))
        .applied
    )
  }
}

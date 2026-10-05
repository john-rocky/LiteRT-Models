package com.kev

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The times of the engine line count only the graphs compiled now. */
class KevEngineTimesTest {
  @Test
  fun noGraphCompiledShowsNoTimes() {
    // A switch to another backend closed the graphs and is compiling the next ones.
    assertNull(
      KevEngineTimes.of(tokenizerMs = 912.4, headMs = 8.7, residentCompileMs = emptyList())
    )
  }

  @Test
  fun aSwitchCountsOnlyTheGraphsOfTheNewBackend() {
    // The GPU pair (18.1 s) closed; L128 compiled for the NPU the first time, L256 loaded from the
    // JIT cache.
    val times = requireNotNull(KevEngineTimes.of(912.4, 8.7, listOf(150_086.0, 2_065.0)))
    assertEquals(152_151L, times.compileMs)
    assertEquals(153_072L, times.loadMs)
  }

  @Test
  fun atStartupTheLoadIsTheEngineReadyFigure() {
    // Tokenizer + head + the startup compile, the figure `ENGINE_READY` logs.
    val times = requireNotNull(KevEngineTimes.of(912.4, 8.7, listOf(18_093.0)))
    assertEquals(19_014L, times.loadMs)
    assertEquals(18_093L, times.compileMs)
  }
}

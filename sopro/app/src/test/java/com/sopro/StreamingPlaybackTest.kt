package com.sopro

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamingPlaybackTest {
  @Test
  fun stopDuringIstftDoesNotHandAnotherChunkToSink() {
    val entry = DspGateSupport.metadata.getJSONArray("static_stream_replay").getJSONObject(0)
    val firstCall = entry.getJSONArray("calls").getJSONObject(0).getJSONArray("outputs")
    var stopped = false
    var clocks = 0
    var sinkCalls = 0
    val decoder =
      StreamingPcmDecoder(
        FixtureData.floats("dsp/constants/istft_window"),
        8192,
        100000,
        -23.0,
        0L,
        { sinkCalls++ },
        shouldStop = { stopped },
        clock = {
          clocks++
          if (clocks == 2) stopped = true
          clocks * 1_000_000L
        },
      )
    var cancelled = false
    try {
      decoder.run(FixtureData.floats(entry.getString("mel")), entry.getInt("frames")) { _, _ ->
        List(firstCall.length()) { FixtureData.floats(firstCall.getString(it)) }
      }
    } catch (_: java.util.concurrent.CancellationException) {
      cancelled = true
    }
    assertTrue(cancelled)
    assertEquals(0, sinkCalls)
    FixtureData.metrics(
      "streaming_cancel",
      mapOf("status" to "PASS", "sink_calls_after_cancel" to sinkCalls),
    )
  }

  @Test
  fun pointwisePlaybackExactlyMatchesKeptRegionBeforeFade() {
    val entries = DspGateSupport.metadata.getJSONArray("postprocess")
    val rows = mutableListOf<Map<String, Any?>>()
    var allPass = entries.length() == 24
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val raw = FixtureData.floats(entry.getString("raw"))
      val levelDb = entry.getDouble("reference_level_db")
      val offline = PostProcess.postprocessSegment(raw, levelDb)
      val played = FloatArray(raw.size)
      // Deliberately cross both trim points with arbitrary playback chunk boundaries.
      var start = 0
      while (start < raw.size) {
        val end = minOf(raw.size, start + 16381)
        StreamingPlayback.process(raw.copyOfRange(start, end), levelDb).copyInto(played, start)
        start = end
      }
      val fade = if (offline.wav.size > 2 * 1920) 1920 else 0
      val keptBeforeFade = offline.wav.size - fade
      val expected = offline.wav.copyOf(keptBeforeFade)
      val actual =
        played.copyOfRange(
          offline.trim.leadCutSamples,
          offline.trim.leadCutSamples + keptBeforeFade,
        )
      val difference = FixtureData.maxDiff(expected, actual)
      val bitExact = expected.contentEquals(actual)
      val pass = difference == 0.0 && bitExact
      allPass = allPass && pass
      rows +=
        mapOf(
          "id" to entry.getString("id"),
          "compared_samples" to actual.size,
          "lead_cut_samples" to offline.trim.leadCutSamples,
          "trail_end_after_lead_samples" to offline.trim.trailEndAfterLeadSamples,
          "excluded_fade_samples" to fade,
          "max_abs_diff" to difference,
          "bit_exact" to bitExact,
          "pass" to pass,
        )
    }
    FixtureData.metrics(
      "streaming_playback",
      mapOf(
        "status" to if (allPass) "PASS" else "FAIL",
        "runtime" to "Mac JVM",
        "utterances" to rows.size,
        "rows" to rows,
      ),
    )
    assertTrue("All 24 played streams must equal offline kept samples before fade exactly", allPass)
  }

  @Test
  fun fakeInvokerPreservesChunksAndTapToFirstSinkTiming() {
    val entries = DspGateSupport.metadata.getJSONArray("static_stream_replay")
    val rows = mutableListOf<Map<String, Any?>>()
    for (i in 0 until entries.length()) {
      val entry = entries.getJSONObject(i)
      val calls = entry.getJSONArray("calls")
      val retained = entry.getJSONArray("retained")
      val frames = entry.getInt("frames")
      val limit = (frames - 32) * 256
      var now = 500_000_000L
      val tap = 100_000_000L
      var callIndex = 0
      val sinkTimes = mutableListOf<Long>()
      val chunks = mutableListOf<FloatArray>()
      val decoder =
        StreamingPcmDecoder(
          FixtureData.floats("dsp/constants/istft_window"),
          8192,
          limit,
          -23.0,
          tap,
          {
            chunks += it
            sinkTimes += now
          },
          clock = { now },
        )
      val output =
        decoder.run(FixtureData.floats(entry.getString("mel")), frames) { mode, _ ->
          val call = calls.getJSONObject(callIndex++)
          assertEquals(call.getString("mode"), mode)
          now += 10_000_000L // deterministic ten milliseconds per recorded graph call
          val outputs = call.getJSONArray("outputs")
          List(outputs.length()) { FixtureData.floats(outputs.getString(it)) }
        }
      val expectedLengths = mutableListOf<Int>()
      var featureFrames = 0
      var istftEmitted = 0
      var context = 8192
      var emitted = 0
      for (j in 0 until retained.length()) {
        val retainedChunk = retained.getJSONObject(j)
        featureFrames += FixtureData.arrayShape(retainedChunk.getString("features"))[1]
        val target =
          if (retainedChunk.getBoolean("flush")) (featureFrames - 1) * 256
          else maxOf(0, featureFrames * 256 - 512)
        val countBeforeContext = target - istftEmitted
        istftEmitted = target
        val drop = minOf(context, countBeforeContext)
        context -= drop
        val count = minOf(countBeforeContext - drop, maxOf(0, limit - emitted))
        if (count > 0) {
          expectedLengths += count
          emitted += count
        }
      }
      assertEquals(expectedLengths, output.chunkLengths)
      assertEquals(output.chunkLengths, chunks.map { it.size })
      assertEquals(emitted, output.raw.size)
      assertEquals(minOf(limit, (frames - 1) * 256 - 8192), output.raw.size)
      assertEquals(calls.length(), callIndex)
      assertEquals(410.0, output.ttfaMs!!, 0.0)
      assertEquals((sinkTimes.first() - tap) / 1_000_000.0, output.ttfaMs!!, 0.0)
      assertEquals(510.0, SynthesisStats.onsetMs(output.ttfaMs, 2400)!!, 0.0)
      assertTrue(sinkTimes.last() > sinkTimes.first())
      val played = chunks.flatMap { it.asIterable() }.toFloatArray()
      assertTrue(StreamingPlayback.process(output.raw, -23.0).contentEquals(played))
      rows +=
        mapOf(
          "id" to entry.getString("id"),
          "frames" to frames,
          "graph_calls" to callIndex,
          "pcm_chunk_samples" to output.chunkLengths,
          "raw_samples" to output.raw.size,
          "ttfa_ms" to output.ttfaMs,
          "tap_before_pipeline_ms" to 400.0,
          "first_graph_ms" to 10.0,
          "ttfa_to_onset_ms_for_2400_lead_samples" to SynthesisStats.onsetMs(output.ttfaMs, 2400),
          "later_chunks_do_not_change_ttfa" to true,
          "pass" to true,
        )
    }
    FixtureData.metrics(
      "streaming_fake_invoker",
      mapOf(
        "status" to "PASS",
        "runtime" to "Mac JVM",
        "rows" to rows,
        "tests" to entries.length(),
      ),
    )
  }
}

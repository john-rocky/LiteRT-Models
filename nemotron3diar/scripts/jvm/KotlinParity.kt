// JVM harness for scripts/gate_kotlin_parity.py: runs the app's pure-Kotlin host (MelFrontend, SpeakerCache,
// Nemotron3Diarizer) on the Mac with a replay engine that returns the transformers reference instead of running
// the LiteRT graphs, and dumps what the host computed.
//
//   replay <assets dir> <wav> <ref dir> <out dir> [push samples]
//     graph A returns the reference chunk rows of the step (ref_rows.bin [n,13,512]), graph B the reference
//     chunk_logits (ref_logits.bin [n,4328,8]); dumps the mel each step fed to graph A (mel.bin [n,104,128]), the
//     packed input of graph B (packed.bin [n,T,512]), the emitted logits (out.bin [frames,8]) and steps.json.
//   replay_offline <assets dir> <wav> <ref dir> <out dir>
//     the offline file mode (Nemotron3Diarizer.runFile, StreamConfig.OFFLINE): graph A returns the reference
//     whole-recording embeddings block by block (embeds.bin [n,512]), graph B the reference chunk logits
//     (chunk_logits.bin [n,5472,8]); dumps mel.bin [blocks,104,128], packed.bin [chunks,684,512],
//     bias.bin [chunks,684], out.bin [frames,8] and steps.json.
//   rope <T> <out dir>
//     rope_cos_<T>.bin / rope_sin_<T>.bin [T,64].
package com.nemotron3diar

import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.BufferedOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

private class ReplayEngine(refDir: File, private val t: Int, private val out: File) : Engine {
  private val rows = StepLog.readFloats(File(refDir, "ref_rows.bin"))
  private val logits = StepLog.readFloats(File(refDir, "ref_logits.bin"))
  private val melOut = DataOutputStream(BufferedOutputStream(FileOutputStream(File(out, "mel.bin")), 1 shl 20))
  private val packedOut = DataOutputStream(BufferedOutputStream(FileOutputStream(File(out, "packed.bin")), 1 shl 20))
  private var frontendCalls = 0
  private var encoderCalls = 0
  private val rowsPerStep = 13 * 512
  private val logitsPerStep = t * 8 * 8

  private fun dump(s: DataOutputStream, a: FloatArray) {
    val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(a)
    s.write(bb.array())
  }

  override fun frontend(mel: FloatArray): FloatArray {
    dump(melOut, mel)
    val k = frontendCalls++
    return rows.copyOfRange(k * rowsPerStep, (k + 1) * rowsPerStep)
  }

  override fun encoder(packed: FloatArray, bias: FloatArray, cos: FloatArray, sin: FloatArray): FloatArray {
    dump(packedOut, packed)
    val k = encoderCalls++
    return logits.copyOfRange(k * logitsPerStep, (k + 1) * logitsPerStep)
  }

  override fun close() {
    melOut.close()
    packedOut.close()
  }
}

private class OfflineReplayEngine(refDir: File, private val t: Int, out: File) : Engine {
  private val embeds = StepLog.readFloats(File(refDir, "embeds.bin"))
  private val logits = StepLog.readFloats(File(refDir, "chunk_logits.bin"))
  private val melOut = DataOutputStream(BufferedOutputStream(FileOutputStream(File(out, "mel.bin")), 1 shl 20))
  private val packedOut = DataOutputStream(BufferedOutputStream(FileOutputStream(File(out, "packed.bin")), 1 shl 20))
  private val biasOut = DataOutputStream(BufferedOutputStream(FileOutputStream(File(out, "bias.bin")), 1 shl 16))
  private var blocks = 0
  private var chunks = 0

  private fun dump(s: DataOutputStream, a: FloatArray) {
    val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    bb.asFloatBuffer().put(a)
    s.write(bb.array())
  }

  override fun frontend(mel: FloatArray): FloatArray {
    dump(melOut, mel)
    val b = blocks++
    val out = FloatArray(13 * 512)
    val from = b * 13 * 512
    embeds.copyInto(out, 0, from, minOf(embeds.size, from + out.size))
    return out
  }

  override fun encoder(packed: FloatArray, bias: FloatArray, cos: FloatArray, sin: FloatArray): FloatArray {
    dump(packedOut, packed)
    dump(biasOut, bias)
    val k = chunks++
    val n = t * 8 * 8
    return logits.copyOfRange(k * n, (k + 1) * n)
  }

  override fun close() {
    melOut.close()
    packedOut.close()
    biasOut.close()
  }
}

private fun replayOffline(args: List<String>) {
  val (assetDir, wavPath, refDir, outDir) = args.take(4).map(::File)
  outDir.mkdirs()
  val (fb, hann, silence) = assets(assetDir)
  val wav = WavReader.read(wavPath)
  val config = StreamConfig.OFFLINE
  val engine = OfflineReplayEngine(refDir, config.maxRows, outDir)
  val diarizer = Nemotron3Diarizer(engine, MelFrontend(fb, hann), silence, config)
  val steps = diarizer.runFile(wav.samples)
  engine.close()
  val frames = steps.sumOf { it.numFrames }
  val all = FloatArray(frames * 8)
  var at = 0
  for (s in steps) {
    s.logits.copyInto(all, at, 0, s.numFrames * 8)
    at += s.numFrames * 8
  }
  StepLog.writeFloats(File(outDir, "out.bin"), all)
  val meta = "{\"wav\":\"${wavPath.name}\",\"samples\":${wav.samples.size},\"frames\":$frames," +
    "\"config\":\"${config.name}\",\"T\":${config.maxRows}}"
  File(outDir, "steps.json").writeText(
    StepLog.runJson(meta, steps.map { StepLog.stepJson(it, IntArray(0)) }, diarizer.cache.compressions))
  println("replay_offline ${wavPath.name}: ${steps.size} chunks, $frames frames")
}

private fun assets(dir: File): Triple<FloatArray, FloatArray, FloatArray> =
  Triple(
    StepLog.readFloats(File(dir, "frontend_mel128_257.bin")),
    StepLog.readFloats(File(dir, "hann400.bin")),
    StepLog.readFloats(File(dir, "silence_embeds.bin")),
  )

private fun replay(args: List<String>) {
  val (assetDir, wavPath, refDir, outDir) = args.take(4).map(::File)
  val push = args.getOrNull(4)?.toInt() ?: 1600
  outDir.mkdirs()
  val (fb, hann, silence) = assets(assetDir)
  val wav = WavReader.read(wavPath)
  check(wav.sampleRate == 16000) { "sample rate ${wav.sampleRate}" }
  val config = StreamConfig.LOW_LATENCY
  val engine = ReplayEngine(refDir, config.maxRows, outDir)
  val diarizer = Nemotron3Diarizer(engine, MelFrontend(fb, hann), silence, config)
  val stepJson = mutableListOf<String>()
  val logits = mutableListOf<FloatArray>()
  var frames = 0
  fun take(steps: List<Step>, ids: List<IntArray>) {
    for ((s, r) in steps.zip(ids)) {
      stepJson += StepLog.stepJson(s, r)
      logits += s.logits
      frames += s.numFrames
    }
  }
  // feed the audio like a microphone, [push] samples at a time; record the fed row ids before each step
  var pos = 0
  while (pos < wav.samples.size) {
    val n = minOf(push, wav.samples.size - pos)
    val before = diarizer.cache.rowIds()
    val steps = diarizer.push(wav.samples, pos, n)
    take(steps, rowIdsPerStep(before, steps))
    pos += n
  }
  val before = diarizer.cache.rowIds()
  val last = diarizer.finish()
  take(last, rowIdsPerStep(before, last))
  engine.close()
  val all = FloatArray(frames * 8)
  var at = 0
  for (l in logits) {
    l.copyInto(all, at)
    at += l.size
  }
  StepLog.writeFloats(File(outDir, "out.bin"), all)
  val meta = "{\"wav\":\"${wavPath.name}\",\"samples\":${wav.samples.size},\"push\":$push,\"frames\":$frames," +
    "\"config\":\"${config.name}\",\"T\":${config.maxRows}}"
  File(outDir, "steps.json").writeText(StepLog.runJson(meta, stepJson, diarizer.cache.compressions))
  println("replay ${wavPath.name}: ${stepJson.size} steps, $frames frames, ${diarizer.cache.compressions.size} compressions")
}

/**
 * The row ids fed to each of [steps]: the first step saw [before]; a push that runs several steps is rare (only
 * when a push spans two chunk boundaries), and those later steps' ids are reconstructed from the steps' own state.
 */
private fun rowIdsPerStep(before: IntArray, steps: List<Step>): List<IntArray> {
  check(steps.size <= 1) { "one push ran ${steps.size} steps: use a smaller push size" }
  return steps.map { before }
}

private fun rope(args: List<String>) {
  val t = args[0].toInt()
  val out = File(args[1])
  out.mkdirs()
  val (c, s) = Nemotron3Diarizer.ropeTables(t)
  StepLog.writeFloats(File(out, "rope_cos_$t.bin"), c)
  StepLog.writeFloats(File(out, "rope_sin_$t.bin"), s)
  println("rope T=$t")
}

fun main(args: Array<String>) {
  when (args.firstOrNull()) {
    "replay" -> replay(args.drop(1))
    "replay_offline" -> replayOffline(args.drop(1))
    "rope" -> rope(args.drop(1))
    else -> error("usage: replay <assets> <wav> <ref dir> <out dir> [push] | rope <T> <out dir>")
  }
}

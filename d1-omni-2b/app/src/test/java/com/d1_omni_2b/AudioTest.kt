package com.d1_omni_2b

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The WAV reader against the model's own code (Python's `wave` and the provider's `read_audio`, on the fixture's
 * test files), and the app's refusals before anything loads: a clip longer than the largest audio graph, an empty
 * question, an unreadable WAV file, an empty clip, and missing model files.
 */
class AudioTest {
  private val fixturesDir = System.getProperty("d1.fixtures").orEmpty()

  private fun refusal(block: () -> Unit): String {
    val failure = runCatching(block).exceptionOrNull()
    assertTrue("expected a refusal, got ${failure ?: "none"}", failure is IllegalArgumentException)
    return failure!!.message!!
  }

  @Test
  fun wavReaderMatchesThePythonReader() {
    assumeTrue("set -Pd1.fixtures", fixturesDir.isNotEmpty())
    val root = File(fixturesDir, "wav_cases")
    @Suppress("UNCHECKED_CAST") val cases = (D1Json.parse(File(root, "cases.json").readBytes()) as Map<String, Any?>)["cases"] as List<Map<String, Any?>>
    assertTrue("at least 10 cases", cases.size >= 10)
    for (case in cases) {
      val name = case["name"] as String
      val bytes = File(root, case["file"] as String).readBytes()
      if (case["ok"] == true) {
        val wav = Wav.read(bytes, name)
        val shape = (case["shape"] as List<*>).map { (it as JsonNumber).toInt() }
        assertEquals("$name rate", (case["rate"] as JsonNumber).toInt(), wav.rate)
        assertEquals("$name channels", shape[0], wav.channels)
        assertEquals("$name samples", shape[0] * shape[1], wav.pcm.size)
        val samples = HostParityTest.readNpy(File(root, case["samples_file"] as String)) // [channels, frames]
        for (f in 0 until shape[1]) {
          for (c in 0 until shape[0]) assertEquals("$name sample $f/$c", samples[c * shape[1] + f], wav.pcm[f * shape[0] + c] / 32768f, 0f)
        }
      } else {
        refusal { Wav.read(bytes, name) }
      }
    }
  }

  @Test
  fun clipsLongerThanTheLargestAudioGraphAreRefused() {
    val extractor = FeatureExtractor()
    fun clip(samples: Int) = AudioClip("clip.wav", FloatArray(samples), 16000)
    // 480,240 samples = 3,000 frames = 30.015 s: the longest clip the 3,000-frame graph takes
    Requests.checkClip(clip(480_240), extractor, 3000)
    assertEquals(3000, extractor.numFrames(480_240))
    assertEquals(3001, extractor.numFrames(480_241))
    val message = refusal { Requests.checkClip(clip(480_241), extractor, 3000) }
    assertTrue(message, "30.015 s" in message && "never cuts" in message)
    // the same limit at another rate: 30.1 s at 44.1 kHz
    refusal { Requests.checkClip(AudioClip("long.wav", FloatArray(1_327_410), 44100), extractor, 3000) }
    // a rate whose resampling filter the app does not build
    val odd = refusal { Requests.checkClip(AudioClip("odd.wav", FloatArray(44101), 44101), extractor, 3000) }
    assertTrue(odd, "44101 Hz" in odd)
  }

  @Test
  fun emptyQuestionsAreRefused() {
    val clip = AudioClip("clip.wav", FloatArray(16000), 16000)
    val draft = ClipDraft(1, File("clip.wav"), "clip.wav", clip, "")
    val empty = QuestionDraft(2, "q1", QType.NOUL, "   ")
    val message = refusal { Requests.build("", false, null, listOf(draft), listOf(empty)) }
    assertTrue(message, "'q1' is empty" in message)
    // a recorded question has no text and is not empty
    val spoken = empty.copy(spoken = draft)
    val (_, questions, media) = Requests.build("", false, null, listOf(draft), listOf(spoken))
    assertTrue((questions["q1"] as Map<*, *>)["instructions"] is AudioClip)
    assertEquals(1, media.size)
    // nothing to read
    refusal { Requests.build("  ", false, null, emptyList(), listOf(QuestionDraft(3, "q2", QType.NOUL, "Is it urgent?"))) }
    // the host's own check, for questions from a run file
    refusal { Questions.parse(mapOf("type" to "noul", "instructions" to "")) }
  }

  @Test
  fun unreadableWavFilesAreRefused() {
    val good = Wav.write(ShortArray(1600) { (it % 100).toShort() }, 16000)
    assertEquals(1600, Wav.clip(good, "good.wav").mono.size)
    val notWav = refusal { Wav.clip("ID3 this is an mp3".toByteArray(), "song.mp3") }
    assertTrue(notWav, "song.mp3" in notWav)
    refusal { Wav.clip(good.copyOf(good.size - 1), "cut.wav") } // ends inside a sample
    refusal { Wav.clip(good.copyOf(40), "header.wav") } // no data chunk
    val eight = good.copyOf().also { it[34] = 8 } // 8-bit samples
    assertTrue(refusal { Wav.clip(eight, "eight.wav") }.contains("16-bit PCM WAV only"))
    val float = good.copyOf().also { it[20] = 3 } // format 3, IEEE float
    assertTrue(refusal { Wav.clip(float, "float.wav") }.contains("16-bit PCM WAV only"))
    // a WAV file with no samples
    assertTrue(refusal { Wav.clip(Wav.write(ShortArray(0), 16000), "empty.wav") }.contains("no samples"))
  }

  @Test
  fun missingModelFilesAreNamed() {
    val dir = Files.createTempDirectory("d1model").toFile()
    try {
      File(dir, "provider").mkdirs()
      File(dir, "provider/tokenizer.json").writeText("{}")
      val message = refusal { ModelFiles.require(dir, ModelFiles.HOST, "the host") }
      for (name in ModelFiles.HOST.drop(1)) assertTrue("$name named", name in message)
      assertTrue("provider/tokenizer.json" !in message)
      assertTrue("nothing was loaded" in message)
    } finally {
      dir.deleteRecursively()
    }
  }
}

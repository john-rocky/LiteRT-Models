package com.d1omni

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * contract.json as the app reads it: the file table (sizes of the files in the repository
 * directory, the tokenizer's sha256), the decision buckets and their signatures, the token IDs
 * against the tokenizer, the temperatures, the request kinds' settings and the Android precision.
 */
class D1ContractTest {
  @Test
  fun fileTableAndBuckets() {
    val contract = ExternalTestData.contract()
    assertEquals(2, contract.version)
    assertEquals(listOf(128, 256, 512, 1024, 2048, 4096), contract.buckets)
    for ((bucket, entry) in contract.decisionFiles) {
      assertEquals("d1-omni-600M_decide_L${bucket}_fp16.tflite", entry.name)
      assertEquals("decide_$bucket", entry.signature)
      val file = java.io.File(ExternalTestData.repo(), entry.name)
      if (file.isFile) assertEquals(entry.name, entry.bytes, file.length())
    }
    assertEquals(
      "69720aa44d60a7feb0bd3056b55c7a038e67761b5d810cd5d5b0496ec61273f3",
      contract.decisionFiles.getValue(128).sha256,
    )
    assertEquals(896250176L, contract.decisionFiles.getValue(128).bytes)
    assertEquals(
      "eebf0dcc2bbefa713c71ce64c5b90600497ff5a26edc3a8ecd0116f99725089b",
      contract.decisionFiles.getValue(256).sha256,
    )
    assertEquals(896315712L, contract.decisionFiles.getValue(256).bytes)
    assertEquals("tokenizer.json", contract.tokenizerFile)
    assertEquals(contract.tokenizerSha256, D1Contract.sha256(ExternalTestData.repoFile("tokenizer.json")))
    assertTrue(contract.androidRecommended!!.startsWith("FP16_WITH_FP32_ACCUM"))
  }

  @Test
  fun tokenIdsTemperaturesAndModes() {
    val contract = ExternalTestData.contract()
    assertEquals(
      linkedMapOf(
        "<|pad|>" to 0,
        "<|startoftext|>" to 1,
        "<|im_end|>" to 7,
        "<|mask|>" to 16,
        "<|reserved_7|>" to 17,
        "<|reserved_8|>" to 18,
        "<|reserved_9|>" to 19,
        "<|reserved_10|>" to 20,
        "<|reserved_11|>" to 21,
      ),
      contract.tokenIds,
    )
    contract.checkTokenizer(ExternalTestData.tokenizer())
    assertEquals(10, contract.temperatures.size)
    assertEquals(1.7301132678985596, contract.temperatures.getValue("score:3-5"), 0.0)
    assertEquals(16384, contract.maxLength)
    val text = contract.modes.getValue("text")
    assertEquals(16384, text.maxLength)
    assertTrue(text.calibrate)
    assertEquals(null, text.noulDefault)
    assertTrue(text.stateNoneBecomes === D1Contract.NOT_SET)
    val audio = contract.modes.getValue("audio")
    assertEquals(15360, audio.maxLength)
    assertTrue(audio.audio)
    assertEquals(linkedMapOf("false" to "no", "true" to "yes"), audio.noulDefault)
    assertEquals(linkedMapOf<String, Any?>(), audio.stateNoneBecomes)
    assertEquals(896, contract.modes.getValue("image").maxLength)
  }

  @Test
  fun sha256OfText() {
    assertEquals(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      D1Contract.sha256(""),
    )
  }
}

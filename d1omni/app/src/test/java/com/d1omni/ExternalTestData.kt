package com.d1omni

import java.io.File
import org.junit.Assume.assumeTrue

/**
 * Reference data kept out of the source tree: the model repository directory (`d1omni.repo`:
 * contract.json, tokenizer.json, fixtures/) and the conversion run's test data directory
 * (`d1omni.demo`: fixtures/tokenizer_probes.json, fixtures/scores_probe.json, device/r1/ rows). A
 * missing property, directory or file is an assumption skip, and a skip is not parity evidence.
 */
internal object ExternalTestData {
  private const val SETUP =
    "Set -Pd1omni.repo=/path/to/d1-omni-600M-LiteRT and -Pd1omni.demo=/path/to/demo; " +
      "see scripts/TEST_DATA.md."

  const val TOKENIZER_CASES = "fixtures/tokenizer_probes.json"
  const val SCORES = "fixtures/scores_probe.json"
  const val ROWS_L128 = "device/r1/rows_L128.json"
  const val ROWS_L256 = "device/r1/rows_L256.json"
  const val TIMING_ROWS = "device/r1/timing_rows.json"

  private var cachedTokenizer: D1Tokenizer? = null
  private var cachedContract: D1Contract? = null
  private var tokenizerLoadMillis = 0.0

  /** The model repository directory. */
  fun repo(): File = directory("d1omni.repo")

  /** The conversion run's test data directory. */
  fun demo(): File = directory("d1omni.demo")

  /** [path] under the model repository; an assumption skip when it is missing. */
  fun repoFile(path: String): File = existing(repo(), path)

  /** [path] under the test data directory; an assumption skip when it is missing. */
  fun demoFile(path: String): File = existing(demo(), path)

  /** The repository's tokenizer, loaded once per test JVM. */
  @Synchronized
  fun tokenizer(): D1Tokenizer {
    cachedTokenizer?.let {
      return it
    }
    val file = repoFile("tokenizer.json")
    val started = System.nanoTime()
    val tokenizer = D1Tokenizer(file)
    tokenizerLoadMillis = (System.nanoTime() - started) / NANOS_PER_MILLI
    println("D1_TOKENIZER loaded ${file.length()} bytes in %.1f ms".format(tokenizerLoadMillis))
    cachedTokenizer = tokenizer
    return tokenizer
  }

  /** The repository's contract.json, read once per test JVM. */
  @Synchronized
  fun contract(): D1Contract =
    cachedContract ?: D1Contract.read(repoFile(D1Contract.FILE)).also { cachedContract = it }

  fun tokenizerLoadMillis(): Double = tokenizerLoadMillis

  /** A JSON file as parsed (Python semantics). */
  fun json(file: File): Map<*, *> = D1Json.parse(file.readBytes()) as Map<*, *>

  fun writeReport(name: String, report: Map<String, Any?>) {
    val build = File(requireNotNull(System.getProperty("d1omni.buildDir"))).canonicalFile
    val file = File(build, "reports/parity/$name").canonicalFile
    check(file.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    requireNotNull(file.parentFile).mkdirs()
    file.writeText(D1Json.writeIndented(report, 1) + "\n")
  }

  fun ints(value: Any?): IntArray = (value as List<*>).map { (it as JsonNumber).toInt() }.toIntArray()

  fun doubles(value: Any?): DoubleArray =
    (value as List<*>).map { (it as JsonNumber).toDouble() }.toDoubleArray()

  private fun directory(property: String): File {
    val path = System.getProperty(property)
    assumeTrue("External parity tests skipped ($property). $SETUP", !path.isNullOrBlank())
    val root = File(requireNotNull(path))
    assumeTrue("External data directory is missing: $root. $SETUP", root.isDirectory)
    return root
  }

  private fun existing(root: File, path: String): File {
    val file = File(root, path)
    assumeTrue("Missing external data: $file. $SETUP", file.isFile)
    return file
  }

  private const val NANOS_PER_MILLI = 1e6
}

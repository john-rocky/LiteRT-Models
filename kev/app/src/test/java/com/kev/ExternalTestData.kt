package com.kev

import java.io.File
import org.junit.Assume.assumeTrue

/**
 * Optional reference data from the conversion run (`kev_work`), kept out of the source tree. A
 * missing property, directory or file is an assumption skip, and a skip is not parity evidence.
 */
internal object ExternalTestData {
  private const val SETUP =
    "Set -Pkev.work=/path/to/kev_work (or -Dkev.work). It holds fixtures/requests.json, " +
      "oracle/oracle_0.8b.json, oracle/hidden_0.8b.npz, host/kev_0.8b_pointer_head.safetensors and " +
      "the Kev tokenizer.json; see scripts/TEST_DATA.md."

  const val REQUESTS = "fixtures/requests.json"
  const val ORACLE = "oracle/oracle_0.8b.json"
  const val HIDDEN = "oracle/hidden_0.8b.npz"
  const val HEAD = "host/kev_0.8b_pointer_head.safetensors"
  const val HEAD_JSON = "host/kev_0.8b_pointer_head.json"

  /** The tokenizer.json published with the Kev checkpoint (jaredpalmer/kev-0.8b @ 788ddbdd). */
  const val TOKENIZER =
    "hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d/tokenizer.json"

  private var cachedTokenizer: KevTokenizer? = null
  private var tokenizerLoadMillis = 0.0

  fun root(): File {
    val directory = System.getProperty("kev.work")
    assumeTrue("External parity tests skipped. $SETUP", !directory.isNullOrBlank())
    val root = File(requireNotNull(directory))
    assumeTrue("External data directory is missing: $root. $SETUP", root.isDirectory)
    return root
  }

  /** [path] under the external root; an assumption skip when it is missing. */
  fun file(path: String): File {
    val file = File(root(), path)
    assumeTrue("Missing external data: $path. $SETUP", file.isFile)
    return file
  }

  /** The Kev tokenizer, loaded once per test JVM. */
  @Synchronized
  fun tokenizer(): KevTokenizer {
    cachedTokenizer?.let {
      return it
    }
    val file = file(TOKENIZER)
    val started = System.nanoTime()
    val tokenizer = KevTokenizer(file)
    tokenizerLoadMillis = (System.nanoTime() - started) / NANOS_PER_MILLI
    println("KEV_TOKENIZER loaded ${file.length()} bytes in %.1f ms".format(tokenizerLoadMillis))
    cachedTokenizer = tokenizer
    return tokenizer
  }

  /** Milliseconds the first [tokenizer] call spent loading tokenizer.json (0 before it). */
  fun tokenizerLoadMillis(): Double = tokenizerLoadMillis

  fun reportFile(name: String): File {
    val build = File(requireNotNull(System.getProperty("kev.buildDir"))).canonicalFile
    val report = File(build, "reports/parity/$name").canonicalFile
    check(report.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    requireNotNull(report.parentFile).mkdirs()
    return report
  }

  fun writeReport(name: String, report: Map<String, Any?>) {
    reportFile(name).writeText(KevJson.write(report) + "\n")
  }

  fun moduleFile(path: String): File =
    File(File(requireNotNull(System.getProperty("kev.moduleRoot"))), path)

  /** A file of app/src/test/resources. */
  fun resource(name: String): File = moduleFile("app/src/test/resources/$name")

  private const val NANOS_PER_MILLI = 1e6
}

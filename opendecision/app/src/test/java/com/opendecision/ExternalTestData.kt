package com.opendecision

import java.io.File
import org.junit.Assume.assumeTrue

/** Optional reference data is separate from the source-only Android project (see scripts/TEST_DATA.md). */
internal object ExternalTestData {
  private const val SETUP =
    "Set -Popendecision.fixtures=/path/to/data (or -Dopendecision.fixtures). The directory holds the conversion run's " +
      "fixtures/requests.json and fixtures/oracle.json, plus tokenizer.json; see scripts/TEST_DATA.md."

  fun resolve(): File {
    val directory = System.getProperty("opendecision.fixtures")
    assumeTrue("External parity tests skipped. $SETUP", !directory.isNullOrBlank())
    val root = File(requireNotNull(directory))
    assumeTrue("External data directory is missing: $root. $SETUP", root.isDirectory)
    requireFiles(root, "fixtures/requests.json", "fixtures/oracle.json")
    return root
  }

  /** `tokenizer.json` next to the run (download layout) or inside the pinned source snapshot. */
  fun tokenizer(root: File): File =
    listOf("tokenizer.json", "hf_staging/tokenizer.json", "src/open-jev-deberta-v3-large/tokenizer.json")
      .map { File(root, it) }
      .firstOrNull { it.isFile }
      ?: run {
        assumeTrue("Missing external data: tokenizer.json. $SETUP", false)
        error("unreachable")
      }

  fun requireFiles(root: File, vararg names: String) {
    for (name in names) assumeTrue("Missing external data: $name. $SETUP", File(root, name).exists())
  }

  fun reportFile(name: String): File {
    val build = File(requireNotNull(System.getProperty("opendecision.buildDir"))).canonicalFile
    val report = File(build, "reports/parity/$name").canonicalFile
    check(report.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    requireNotNull(report.parentFile).mkdirs()
    return report
  }

  fun resource(name: String): String =
    requireNotNull(ExternalTestData::class.java.classLoader?.getResource(name)) { "Missing test resource $name" }.readText()
}

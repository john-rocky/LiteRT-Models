package com.gliner25decide

import java.io.File
import org.junit.Assume.assumeTrue

/** Optional reference data is separate from the source-only Android project. */
internal object ExternalTestData {
  private const val SETUP =
    "Set -Pgliner.fixtures=/path/to/data (or -Dgliner.fixtures). The directory holds " +
      "fixtures/oracle_fp32.json plus the tokenizer and float16 table, either as host_assets/ " +
      "or in the conversion run's layout; see scripts/TEST_DATA.md."
  private const val SNAPSHOT =
    "cache/hf/hub/models--fastino--GLiNER2.5-Decide/snapshots/" +
      "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"

  fun resolve(): File {
    val directory = System.getProperty("gliner.fixtures")
    assumeTrue("External parity tests skipped. $SETUP", !directory.isNullOrBlank())
    val root = File(requireNotNull(directory))
    assumeTrue("External data directory is missing: $root. $SETUP", root.isDirectory)
    requireFiles(root, "fixtures/oracle_fp32.json")
    return root
  }

  /** `host_assets/tokenizer.json` (download layout) or the run's pinned HF snapshot. */
  fun tokenizer(root: File): File =
    firstExisting(root, "host_assets/tokenizer.json", "$SNAPSHOT/tokenizer.json")

  /** `host_assets/word_embeddings_fp16.bin` (download layout) or the run's export. */
  fun embeddingTable(root: File): File =
    firstExisting(
      root,
      "host_assets/word_embeddings_fp16.bin",
      "exports/host_assets/word_embeddings_fp16.bin",
    )

  fun requireFiles(root: File, vararg names: String) {
    for (name in names) {
      assumeTrue("Missing external data: $name. $SETUP", File(root, name).exists())
    }
  }

  fun reportFile(name: String): File {
    val build = File(requireNotNull(System.getProperty("gliner.buildDir"))).canonicalFile
    val report = File(build, "reports/parity/$name").canonicalFile
    check(report.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    requireNotNull(report.parentFile).mkdirs()
    return report
  }

  fun moduleFile(path: String): File =
    File(File(requireNotNull(System.getProperty("gliner.moduleRoot"))), path)

  fun resource(name: String): String =
    requireNotNull(ExternalTestData::class.java.classLoader?.getResource(name)) {
        "Missing test resource $name"
      }
      .readText()

  private fun firstExisting(root: File, vararg names: String): File {
    val file = names.map { File(root, it) }.firstOrNull { it.isFile }
    assumeTrue("Missing external data: one of ${names.toList()}. $SETUP", file != null)
    return file!!
  }
}

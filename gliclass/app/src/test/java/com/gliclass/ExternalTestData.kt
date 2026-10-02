package com.gliclass

import java.io.File
import org.junit.Assume.assumeTrue

/** Optional reference data is separate from the source-only Android project. */
internal object ExternalTestData {
  private const val SETUP =
    "Set -Pgliclass.fixtures=/path/to/data (or -Dgliclass.fixtures). The directory holds " +
      "fixtures/oracle.json, fixtures/requests.json and fixtures/tokenizer_stress.json plus the " +
      "tokenizer and float16 table; see scripts/TEST_DATA.md."

  fun resolve(): File {
    val directory = System.getProperty("gliclass.fixtures")
    assumeTrue("External parity tests skipped. $SETUP", !directory.isNullOrBlank())
    val root = File(requireNotNull(directory))
    assumeTrue("External data directory is missing: $root. $SETUP", root.isDirectory)
    requireFiles(root, "fixtures/oracle.json", "fixtures/requests.json")
    return root
  }

  /** `tokenizer.json` at the top level, under `host_assets/`, or in the conversion run's source. */
  fun tokenizer(root: File): File =
    firstExisting(
      root,
      "tokenizer.json",
      "host_assets/tokenizer.json",
      "src/gliclass-edge-v3.0/tokenizer.json",
    )

  /** `tok_embeddings_fp16.bin` at the top level, under `host_assets/`, or in the run's exports. */
  fun embeddingTable(root: File): File =
    firstExisting(
      root,
      "tok_embeddings_fp16.bin",
      "host_assets/tok_embeddings_fp16.bin",
      "exports/tables/tok_embeddings_fp16.bin",
    )

  fun requireFiles(root: File, vararg names: String) {
    for (name in names) {
      assumeTrue("Missing external data: $name. $SETUP", File(root, name).exists())
    }
  }

  fun reportFile(name: String): File {
    val build = File(requireNotNull(System.getProperty("gliclass.buildDir"))).canonicalFile
    val report = File(build, "reports/parity/$name").canonicalFile
    check(report.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    requireNotNull(report.parentFile).mkdirs()
    return report
  }

  fun moduleFile(path: String): File =
    File(File(requireNotNull(System.getProperty("gliclass.moduleRoot"))), path)

  private fun firstExisting(root: File, vararg names: String): File {
    val file = names.map { File(root, it) }.firstOrNull { it.isFile }
    assumeTrue("Missing external data: one of ${names.toList()}. $SETUP", file != null)
    return file!!
  }
}

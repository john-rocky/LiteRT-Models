package com.gliner25

import java.io.File
import org.junit.Assume.assumeTrue

/** Optional reference data is separate from the source-only Android project. */
internal object ExternalTestData {
  private const val SETUP =
    "Set -Pgliner.fixtures=/path/to/data (or -Dgliner.fixtures). " +
      "Download litert-community/GLiNER2.5-Small-LiteRT with hf download into that directory " +
      "for host_assets/. Packed references are not distributed; regenerate them with the " +
      "HF Python HostRuntime and CPU CompiledModel as described in scripts/TEST_DATA.md."

  fun resolve(): File {
    val directory = System.getProperty("gliner.fixtures")
    assumeTrue("External parity tests skipped. $SETUP", !directory.isNullOrBlank())
    val root = File(requireNotNull(directory))
    assumeTrue("External data directory is missing: $root. $SETUP", root.isDirectory)
    requireFiles(root, "host_assets/tokenizer.json")
    return root
  }

  fun requireFiles(root: File, vararg names: String) {
    for (name in names) {
      assumeTrue("Missing external data: $name. $SETUP", File(root, name).exists())
    }
  }

  fun reportFile(name: String): File {
    val build = File(requireNotNull(System.getProperty("gliner.buildDir"))).canonicalFile
    val report = File(build, "reports/parity/$name").canonicalFile
    check(report.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    report.parentFile.mkdirs()
    return report
  }
}

package com.gliformer

import java.io.File
import org.junit.Assume.assumeTrue

/** Large parity fixtures are optional external data, never a dependency of assembling the app. */
internal object ExternalTestData {
  fun resolve(): File {
    val property = System.getProperty("gliformer.fixtures")
    assumeTrue(
      "External parity data is not configured. Set -Pgliformer.fixtures=DATA; see scripts/TEST_DATA.md. A skipped test is not parity evidence.",
      !property.isNullOrBlank(),
    )
    val directory = File(requireNotNull(property)).canonicalFile
    check(directory.isDirectory) { "Configured fixture directory does not exist: $directory" }
    return directory
  }

  fun reportFile(name: String): File {
    val build = File(requireNotNull(System.getProperty("gliformer.buildDir"))).canonicalFile
    val report = File(build, "reports/parity/$name").canonicalFile
    check(report.toPath().startsWith(build.toPath())) { "Reports must remain under build/" }
    requireNotNull(report.parentFile).mkdirs()
    return report
  }
}

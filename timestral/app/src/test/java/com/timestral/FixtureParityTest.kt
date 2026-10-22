package com.timestral

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The Kotlin host on the conversion run's parity fixture, read from outside the module (it is not
 * redistributed): `-Ptimestral.fixturePack=<dir>` (index.tsv and each window's context, float32),
 * `-Ptimestral.rows=<dir>` (the graph inputs the Python host built for each window and the desktop
 * CPU output of the shipped graph on them), `-Ptimestral.host=<dir>` (the host tables) and
 * `-Ptimestral.out=<dir>` (where this test writes each window's quantiles for the Python side to
 * score). Without them the test is skipped; a skip is not parity evidence.
 *
 * Bar: the five graph inputs of every window are bit-equal to the Python host's bytes.
 */
class FixtureParityTest {
  @Test
  fun everyWindowBuildsThePythonInputsAndFinishesTheForecast() {
    val pack = System.getProperty("timestral.fixturePack")
    val rows = System.getProperty("timestral.rows")
    val out = System.getProperty("timestral.out")
    val tables = TestData.hostTables()
    assumeTrue("external fixture not configured", pack != null && rows != null && out != null && tables != null)
    val target = File(out!!, "kotlin")
    target.mkdirs()
    val lines = File(pack!!, "index.tsv").readLines().drop(1).filter { it.isNotBlank() }
    val report = StringBuilder("index\tid\tbucket\tinputs_bit_equal\tmax_ulp\n")
    var windows = 0
    var allEqual = 0
    for (line in lines) {
      val fields = line.split('\t')
      val index = fields[0]
      val id = fields[1]
      val bucketDir = fields[2]
      val horizon = fields[3].toInt()
      val mask = fields[4].toDouble().toFloat()
      val reference = fields[5]
      val series = TestData.floats(File(pack, fields[6]).readBytes())
      val prepared = TimestralHost.prepare(series, horizon)
      val bucket =
        checkNotNull(
            GraphSpec.choose(
              TestData.TWO_ROW_H32,
              Backend.CPU,
              prepared.contextPatches,
              prepared.horizonPatches,
            )
          )
          .bucket
      check(bucketDir == "l${bucket.contextPatches}_h${bucket.horizonPatches}") { "$id: $bucketDir vs ${bucket.name}" }
      val inputs = TimestralHost.buildInputs(prepared, bucket, tables!!, mask).byName()
      var equal = 0
      var maxUlp = 0L
      for ((k, name) in NAMES.withIndex()) {
        val python = TestData.floats(File(rows!!, "$bucketDir/$id/in_0${k}_$name.f32").readBytes())
        val difference = TestData.difference(inputs.getValue(name), python)
        if (difference.notBitEqual == 0) equal++
        maxUlp = maxOf(maxUlp, difference.maxUlp)
      }
      val raw = TestData.floats(File(rows!!, reference).readBytes())
      val quantiles = TimestralHost.finish(prepared, raw, bucket)
      File(target, "$index.q.f32").writeBytes(TestData.bytes(quantiles))
      windows++
      if (equal == NAMES.size) allEqual++
      report.append("$index\t$id\t${bucket.name}\t$equal/${NAMES.size}\t$maxUlp\n")
    }
    File(out, "fixture_inputs.tsv").writeText(report.toString())
    println("fixture windows $windows, all five inputs bit-equal in $allEqual")
    assertTrue("no windows", windows > 0)
    assertTrue("inputs differ in ${windows - allEqual} windows (see fixture_inputs.tsv)", allEqual == windows)
  }

  private companion object {
    val NAMES = listOf("patches", "dec", "bias", "cos", "sin")
  }
}

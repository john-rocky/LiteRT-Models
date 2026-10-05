package com.kev

/**
 * What a timed call starts under besides the GPU ceiling: whether any CPU frequency policy is
 * capped ([cpuCapped]) and the thermal status (0 = none); null when it cannot be read.
 */
data class KevCaps(val cpuCapped: Boolean?, val thermalStatus: Int?)

/** The GPU's clock ceiling (MHz) and temperature (milli °C) as kgsl reports them. */
data class KevGpuState(val maxClockMhz: Int, val tempMilliC: Int) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf("readable" to true, "max_clock_mhz" to maxClockMhz, "temp_mc" to tempMilliC)
}

/**
 * The wait before each timed set and request path of a timing run with [coolMs] > 0. A graph
 * compile alone warms the GPU and lowers its clock ceiling for the calls right after it, so the run
 * reads the GPU state once before the first compile ([readBase]) and, before each set, waits until
 * the ceiling is back to at least that value and the temperature within [TEMP_MARGIN_MILLI_C] of
 * it, for at most [coolMs]. When the state cannot be read, it waits [coolMs] instead. Android-free:
 * [read], [clock] and [sleep] are passed in.
 */
class KevCooler(
  private val coolMs: Long,
  private val read: () -> KevGpuState?,
  private val clock: () -> Long = System::currentTimeMillis,
  private val sleep: (Long) -> Unit = Thread::sleep,
) {
  /** The state read before the first compile, or null (not read yet, or not readable). */
  var base: KevGpuState? = null
    private set

  private var baseRead = false

  /** Reads the base once and returns it; null when the state cannot be read. */
  fun readBase(): KevGpuState? {
    if (!baseRead) {
      base = read()
      baseRead = true
    }
    return base
  }

  /** Waits as described on the class; the record says how (`mode` none / fixed / kgsl). */
  fun waitForBase(): LinkedHashMap<String, Any?> {
    val record = linkedMapOf<String, Any?>("cool_ms" to coolMs)
    if (coolMs <= 0) return record.apply { put("mode", "none") }
    val start = clock()
    val base = base
    var now = read()
    if (base == null || now == null) {
      sleep(coolMs)
      return record.apply {
        put("mode", "fixed")
        put("waited_ms", clock() - start)
      }
    }
    while (now != null && !recovered(base, now) && clock() - start < coolMs) {
      sleep(POLL_MS)
      now = read()
    }
    return record.apply {
      put("mode", "kgsl")
      put("waited_ms", clock() - start)
      put("base", base.toJson())
      put("end", now?.toJson() ?: linkedMapOf("readable" to false))
      put("recovered", now != null && recovered(base, now))
    }
  }

  companion object {
    /** How far above the base temperature the GPU may be when a set starts. */
    const val TEMP_MARGIN_MILLI_C = 5_000

    /** How often the state is read while waiting. */
    const val POLL_MS = 500L

    /** The ceiling is back to the base and the temperature within the margin of it. */
    fun recovered(base: KevGpuState, now: KevGpuState): Boolean =
      now.maxClockMhz >= base.maxClockMhz && now.tempMilliC <= base.tempMilliC + TEMP_MARGIN_MILLI_C
  }
}

package com.d1omni

/** The extras of a launch intent, read by name (Android-free; `MainActivity` wraps the intent). */
interface D1Extras {
  fun has(name: String): Boolean

  fun string(name: String): String?

  fun int(name: String, default: Int): Int

  fun boolean(name: String, default: Boolean): Boolean
}

/** What a launch intent asks for (see `MainActivity` for the extras). */
sealed interface D1Launch {
  /** The app: [backend], the GPU [precision] of the decision graphs and of the audio graph. */
  data class Normal(
    val backend: D1Backend = D1Backend.GPU,
    val precision: D1Precision = D1Precision.DEFAULT,
    val audioPrecision: D1Precision = D1AudioEngine.DEFAULT_PRECISION,
  ) : D1Launch

  /**
   * Debug build: the rows of `files/<fixture>` (one bucket L, rows already encoded) through the L
   * graph, one call per row, into `files/<report>`; [resident] compiles that bucket first and keeps
   * it while L compiles (two graphs at once, the memory record); [limit] > 0 runs the first rows. A
   * rows file of kind `audio` holds clips instead: each clip's wav through the audio graph at
   * [audioPrecision], then its rows on the decision graphs (see [D1GateRunner]).
   */
  data class Gate(
    val fixture: String,
    val report: String,
    val limit: Int,
    val backend: D1Backend,
    val precision: D1Precision,
    val resident: Int?,
    val audioPrecision: D1Precision = D1AudioEngine.DEFAULT_PRECISION,
  ) : D1Launch

  /**
   * Debug build: the timing sets of `files/<rows>` (only [sets] when named): per set, a wait of at
   * most [coolMs] for the GPU clock ceiling and temperature, [warmup] calls, then [reps] rounds of
   * the set's rows; into `files/<report>`.
   */
  data class Timing(
    val rows: String,
    val report: String,
    val warmup: Int,
    val reps: Int,
    val coolMs: Long,
    val sets: List<String>?,
    val backend: D1Backend,
    val precision: D1Precision,
    val audioPrecision: D1Precision = D1AudioEngine.DEFAULT_PRECISION,
  ) : D1Launch

  /** Extras that cannot be followed. */
  data class Invalid(val reason: String) : D1Launch

  companion object {
    const val EXTRA_GATE = "gate"
    const val EXTRA_TIMING = "timing"
    const val EXTRA_FIXTURE = "fixture"
    const val EXTRA_ROWS = "rows"
    const val EXTRA_REPORT = "report"
    const val EXTRA_LIMIT = "limit"
    const val EXTRA_PRECISION = "precision"
    const val EXTRA_PRECISION_AUDIO = "precision_audio"
    const val EXTRA_BACKEND = "backend"
    const val EXTRA_RESIDENT = "resident"
    const val EXTRA_WARMUP = "warmup"
    const val EXTRA_REPS = "reps"
    const val EXTRA_COOL_MS = "cool_ms"
    const val EXTRA_SETS = "sets"
    const val DEFAULT_WARMUP = 5
    const val DEFAULT_REPS = 20

    /** File names stay inside `files/`: letters, digits, dot, underscore and hyphen. */
    private val FILE_NAME = Regex("[A-Za-z0-9._-]+")

    /** Buckets a gate may keep resident (the graphs of this round's install). */
    private val RESIDENT_BUCKETS = listOf(128, 256, 512, 1024, 2048)

    fun fileNameValid(name: String): Boolean =
      name.matches(FILE_NAME) && !name.endsWith(".partial") && name != "." && name != ".."

    /** The launch [extras] ask for; gate and timing runs only in a [debug] build. */
    fun parse(extras: D1Extras, debug: Boolean): D1Launch {
      val precisionName = extras.string(EXTRA_PRECISION)
      val precision =
        if (precisionName == null) D1Precision.DEFAULT
        else
          D1Precision.of(precisionName)
            ?: return Invalid("precision $precisionName is not fp16acc or fp32")
      val audioPrecisionName = extras.string(EXTRA_PRECISION_AUDIO)
      val audioPrecision =
        if (audioPrecisionName == null) D1AudioEngine.DEFAULT_PRECISION
        else
          D1Precision.of(audioPrecisionName)
            ?: return Invalid("precision_audio $audioPrecisionName is not fp16acc or fp32")
      val backendName = extras.string(EXTRA_BACKEND)
      val backend =
        if (backendName == null) D1Backend.GPU
        else D1Backend.of(backendName) ?: return Invalid("backend $backendName is not gpu or cpu")
      val gate = extras.boolean(EXTRA_GATE, false)
      val timing = extras.boolean(EXTRA_TIMING, false)
      if ((gate || timing) && !debug) return Invalid("gate and timing runs need the debug build")
      return when {
        gate && timing -> Invalid("gate and timing exclude each other")
        gate -> {
          val fixture = extras.string(EXTRA_FIXTURE)
          val report = extras.string(EXTRA_REPORT)
          val resident = if (extras.has(EXTRA_RESIDENT)) extras.int(EXTRA_RESIDENT, 0) else null
          when {
            fixture.isNullOrEmpty() -> Invalid("no fixture extra")
            !fileNameValid(fixture) -> Invalid("invalid fixture name $fixture")
            report.isNullOrEmpty() || !fileNameValid(report) -> Invalid("invalid report name $report")
            resident != null && resident !in RESIDENT_BUCKETS ->
              Invalid("resident $resident is not one of ${RESIDENT_BUCKETS.joinToString(", ")}")
            else ->
              Gate(fixture, report, extras.int(EXTRA_LIMIT, 0), backend, precision, resident, audioPrecision)
          }
        }
        timing -> {
          val rows = extras.string(EXTRA_ROWS)
          val report = extras.string(EXTRA_REPORT)
          val warmup = extras.int(EXTRA_WARMUP, DEFAULT_WARMUP)
          val reps = extras.int(EXTRA_REPS, DEFAULT_REPS)
          when {
            rows.isNullOrEmpty() || !fileNameValid(rows) -> Invalid("invalid rows name $rows")
            report.isNullOrEmpty() || !fileNameValid(report) -> Invalid("invalid report name $report")
            warmup < 0 || reps < 1 -> Invalid("warmup $warmup and reps $reps: warmup >= 0, reps >= 1")
            else ->
              Timing(
                rows,
                report,
                warmup,
                reps,
                extras.int(EXTRA_COOL_MS, 0).toLong(),
                extras.string(EXTRA_SETS)?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() },
                backend,
                precision,
                audioPrecision,
              )
          }
        }
        else -> Normal(backend, precision, audioPrecision)
      }
    }
  }
}

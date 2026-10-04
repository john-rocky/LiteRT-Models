package com.kev

/** One timed row: a key and its unpadded token IDs. */
class KevTimingRow(val key: String, val ids: IntArray)

/**
 * A set of `timing_rows.json`: a `request` set runs its rows back to back as one request, a
 * `single` set is one row. [window] is the graph window the set is meant for.
 */
class KevTimingSet(
  val name: String,
  val kind: String,
  val window: Int,
  val synthetic: Boolean,
  val rows: List<KevTimingRow>,
)

/**
 * A set left out of a timing run ([window] is null for a requested name the file does not have).
 */
class KevTimingSkip(val name: String, val window: Int?, val reason: String)

/** The sets one timing run times, in file order, and the ones it leaves out. */
class KevTimingSelection(val run: List<KevTimingSet>, val skipped: List<KevTimingSkip>)

/**
 * The conversion run's timing rows (`{pad_id, protocol, source, sets:
 * [{name, kind, L, rows: [{key, ids}]}]}`), so the app times the same rows as the model card.
 */
class KevTimingRows(
  val padId: Int,
  val protocol: String?,
  val source: String?,
  val sets: List<KevTimingSet>,
) {
  /**
   * The sets to time on a [window] graph: the [requested] names (every set when null, none when
   * empty) whose L is [window], so that each set can get a launch of its own.
   */
  fun select(requested: List<String>?, window: Int): KevTimingSelection {
    val run = ArrayList<KevTimingSet>()
    val skipped = ArrayList<KevTimingSkip>()
    for (set in sets) {
      when {
        requested != null && set.name !in requested ->
          skipped.add(KevTimingSkip(set.name, set.window, "not requested"))
        set.window != window ->
          skipped.add(KevTimingSkip(set.name, set.window, "the resident graph is L$window"))
        else -> run.add(set)
      }
    }
    requested
      .orEmpty()
      .filter { name -> sets.none { it.name == name } }
      .forEach { skipped.add(KevTimingSkip(it, null, "not in the rows file")) }
    return KevTimingSelection(run, skipped)
  }

  /**
   * The sets to time on the pair [shape] (their L is not used): the [requested] names (every set
   * when null, none when empty) whose rows split into requests the pair takes — each row's state
   * part (before its `[question]` token) at most Ls, its branch at most Lq, and one state for all
   * rows of a `request` set.
   */
  fun selectPair(requested: List<String>?, shape: KevPairShape): KevTimingSelection {
    val run = ArrayList<KevTimingSet>()
    val skipped = ArrayList<KevTimingSkip>()
    for (set in sets) {
      if (requested != null && set.name !in requested) {
        skipped.add(KevTimingSkip(set.name, set.window, "not requested"))
        continue
      }
      val reason = pairMiss(set, shape)
      if (reason == null) run.add(set) else skipped.add(KevTimingSkip(set.name, set.window, reason))
    }
    requested
      .orEmpty()
      .filter { name -> sets.none { it.name == name } }
      .forEach { skipped.add(KevTimingSkip(it, null, "not in the rows file")) }
    return KevTimingSelection(run, skipped)
  }

  /** Why the pair [shape] cannot run [set], or null. */
  private fun pairMiss(set: KevTimingSet, shape: KevPairShape): String? {
    val states = ArrayList<List<Int>>()
    for (row in set.rows) {
      val cut = row.ids.indexOf(KevEncoder.QUESTION_ID)
      if (cut <= 0) return "row ${row.key} has no question token"
      if (cut > shape.stateLength) return "row ${row.key}: state $cut > Ls ${shape.stateLength}"
      val branch = row.ids.size - cut
      if (branch > shape.questionLength) {
        return "row ${row.key}: branch $branch > Lq ${shape.questionLength}"
      }
      states.add(row.ids.copyOf(cut).toList())
    }
    if (set.kind == "request" && states.distinct().size > 1) return "the rows differ in their state"
    return null
  }

  companion object {
    fun parse(bytes: ByteArray): KevTimingRows {
      val json = KevJson.parse(bytes) as Map<*, *>
      val padId = (json["pad_id"] as JsonNumber).toInt()
      require(padId == KevEncoder.PAD_ID) {
        "timing rows pad_id $padId, the graph pads with ${KevEncoder.PAD_ID}"
      }
      val sets =
        (json["sets"] as List<*>).map { entry ->
          val set = entry as Map<*, *>
          val rows =
            (set["rows"] as List<*>).map { rowEntry ->
              val row = rowEntry as Map<*, *>
              KevTimingRow(
                row["key"] as String,
                (row["ids"] as List<*>).map { (it as JsonNumber).toInt() }.toIntArray(),
              )
            }
          val window = (set["L"] as JsonNumber).toInt()
          require(rows.isNotEmpty() && rows.all { it.ids.size <= window }) {
            "Set ${set["name"]}: a row is over L=$window"
          }
          KevTimingSet(
            set["name"] as String,
            set["kind"] as String,
            window,
            set["synthetic"] == true,
            rows,
          )
        }
      return KevTimingRows(padId, json["protocol"] as String?, json["source"] as String?, sets)
    }
  }
}

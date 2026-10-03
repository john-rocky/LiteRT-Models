package com.kev

/** One timed row: a key and its unpadded token IDs. */
class KevTimingRow(val key: String, val ids: IntArray)

/**
 * A set of `timing_rows.json`: a `request` set runs its rows back to back as one request, a
 * `single` set is one row. [window] is the graph window the set is meant for.
 */
class KevTimingSet(val name: String, val kind: String, val window: Int, val synthetic: Boolean, val rows: List<KevTimingRow>)

/**
 * The conversion run's timing rows (`{pad_id, protocol, source, sets: [{name, kind, L, rows: [{key,
 * ids}]}]}`), so the app times the same rows as the model card.
 */
class KevTimingRows(val padId: Int, val protocol: String?, val source: String?, val sets: List<KevTimingSet>) {
  companion object {
    fun parse(bytes: ByteArray): KevTimingRows {
      val json = KevJson.parse(bytes) as Map<*, *>
      val padId = (json["pad_id"] as JsonNumber).toInt()
      require(padId == KevEncoder.PAD_ID) { "timing rows pad_id $padId, the graph pads with ${KevEncoder.PAD_ID}" }
      val sets =
        (json["sets"] as List<*>).map { entry ->
          val set = entry as Map<*, *>
          val rows =
            (set["rows"] as List<*>).map { rowEntry ->
              val row = rowEntry as Map<*, *>
              KevTimingRow(row["key"] as String, (row["ids"] as List<*>).map { (it as JsonNumber).toInt() }.toIntArray())
            }
          val window = (set["L"] as JsonNumber).toInt()
          require(rows.isNotEmpty() && rows.all { it.ids.size <= window }) { "Set ${set["name"]}: a row is over L=$window" }
          KevTimingSet(set["name"] as String, set["kind"] as String, window, set["synthetic"] == true, rows)
        }
      return KevTimingRows(padId, json["protocol"] as String?, json["source"] as String?, sets)
    }
  }
}

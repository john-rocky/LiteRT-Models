// SPDX-License-Identifier: Apache-2.0
package com.sopro

import java.io.File
import org.json.JSONObject

class MissingModelFile(val filename: String) : IllegalStateException(filename)

/** Model metadata is overlaid newest first; unchanged models use the base contract. */
class ModelCatalog(
  private val directory: File,
  private val precision: SoproEngine.Precision,
  private val contractSet: String,
  private val styleVariant: PlacementConfig.StyleVariant,
) {
  private val contracts: List<JSONObject>

  init {
    require(contractSet == "r6" || contractSet == "r9")
    val names =
      if (contractSet == "r9") listOf("contract_r9.json", "contract_r6.json", "contract.json")
      else listOf("contract_r6.json", "contract.json")
    contracts = names.map { name ->
      val file = File(directory, name)
      if (!file.isFile) throw MissingModelFile(name)
      JSONObject(file.readText())
    }
  }

  fun spec(graph: String): JSONObject {
    val storage = PlacementConfig.storage(graph, precision, contractSet, styleVariant)
    for (contract in contracts) {
      val models = contract.getJSONArray("models")
      val matches =
        (0 until models.length())
          .map { models.getJSONObject(it) }
          .filter {
            it.getString("graph") == graph &&
              it.optString("storage", it.getString("path").substringBefore('/')) == storage
          }
      require(matches.size <= 1) { "Ambiguous model $graph/$storage" }
      if (matches.isNotEmpty()) return matches.single()
    }
    error("Model contract is missing $graph/$storage")
  }

  fun requireFiles(graphs: List<String>) {
    val paths = hostAssetPaths + graphs.map { spec(it).getString("path") }
    for (path in paths) if (!File(directory, path).isFile) throw MissingModelFile(File(path).name)
  }

  companion object {
    val hostAssetPaths =
      listOf(
        "host_assets/host_assets.json",
        "host_assets/dsp_constants_fp32.bin",
        "host_assets/tokenizer.model",
        "host_assets/ar_tables_fp32.json",
        "host_assets/ar_tables_fp32.bin",
      )
  }
}

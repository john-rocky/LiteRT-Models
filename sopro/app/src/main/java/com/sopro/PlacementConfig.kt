// SPDX-License-Identifier: Apache-2.0
package com.sopro

/** Explicit device map; unrecognized devices retain the CPU configuration. */
object PlacementConfig {
  enum class Mode {
    AUTOMATIC,
    CPU,
    GPU_AR,
    CUSTOM,
  }

  enum class StyleVariant {
    FP32,
    WFP16;

    companion object {
      fun parse(value: String): StyleVariant = valueOf(value.uppercase())
    }
  }

  fun isGalaxyS26(hardware: String, socModel: String): Boolean =
    socModel.trim().equals("SM8850", true) || hardware.trim().equals("SM8850", true)

  fun automatic(hardware: String, socModel: String): Map<String, SoproEngine.Backend> =
    if (isGalaxyS26(hardware, socModel)) hybrid() else emptyMap()

  fun hybrid(): Map<String, SoproEngine.Backend> =
    mapOf(
      "speaker_encoder" to SoproEngine.Backend.GPU32,
      "semantic_encoder" to SoproEngine.Backend.GPU32,
      "acoustic_condition" to SoproEngine.Backend.GPU,
      "acoustic_velocity" to SoproEngine.Backend.GPU,
      "acoustic_condition_t4096" to SoproEngine.Backend.GPU,
      "acoustic_velocity_t4096" to SoproEngine.Backend.GPU,
    )

  fun placement(mode: Mode, hardware: String, socModel: String): Map<String, SoproEngine.Backend> =
    when (mode) {
      Mode.AUTOMATIC -> automatic(hardware, socModel)
      Mode.CPU -> emptyMap()
      Mode.GPU_AR -> hybrid() + ("ar_merged" to SoproEngine.Backend.GPU32)
      Mode.CUSTOM -> error("A custom placement requires an explicit map")
    }

  fun precision(mode: Mode): SoproEngine.Precision =
    if (mode == Mode.GPU_AR) SoproEngine.Precision.WFP16 else SoproEngine.Precision.SHIP

  /** FP32 style is the current candidate; the smaller twin remains an explicit selection. */
  fun storage(
    graph: String,
    precision: SoproEngine.Precision,
    contractSet: String,
    styleVariant: StyleVariant,
  ): String =
    when {
      precision == SoproEngine.Precision.FP32 -> "fp32"
      graph == "style_prefix" && contractSet == "r9" -> styleVariant.name.lowercase()
      graph == "ar_merged" && precision == SoproEngine.Precision.SHIP -> "int8"
      else -> "wfp16"
    }
}

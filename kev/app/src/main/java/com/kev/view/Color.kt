package com.kev.view

import androidx.compose.ui.graphics.Color
import com.kev.CardState
import com.kev.KevDemoRun

/** Primary color of the sample's Material theme. */
val darkBlue = Color(0xFF174378)

/** Secondary color of the sample's Material theme. */
val teal = Color(0xFF00796B)

/** A failed card (not part of the recorded palette). */
val failedRed = Color(0xFFC62828)

/** The card state colors, read from the palette the demo run JSON records. */
fun stateColor(state: CardState): Color =
  when (state) {
    CardState.PENDING -> hex(KevDemoRun.PALETTE.getValue("pending"))
    CardState.RUNNING -> hex(KevDemoRun.PALETTE.getValue("running"))
    CardState.DONE -> hex(KevDemoRun.PALETTE.getValue("done"))
    CardState.FAILED -> failedRed
  }

private fun hex(value: String): Color = Color(0xFF000000 or value.removePrefix("#").toLong(16))

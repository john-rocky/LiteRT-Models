package com.gliformer.view

import androidx.compose.ui.graphics.Color

val darkBlue = Color(0xFF173B63)
val teal = Color(0xFF00796B)

internal fun entityColor(label: String): Color =
  when (label) {
    "person" -> Color(0xFFD6E8FF)
    "organization" -> Color(0xFFD4F1E8)
    "location" -> Color(0xFFFFE7BD)
    "product" -> Color(0xFFE8DDF8)
    else -> Color(0xFFFFDDE6)
  }

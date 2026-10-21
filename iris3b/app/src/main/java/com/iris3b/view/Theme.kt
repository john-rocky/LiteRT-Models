package com.iris3b.view

import androidx.compose.material.MaterialTheme
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Indigo = Color(0xFF283593)
private val DeepIndigo = Color(0xFF001064)
private val Amber = Color(0xFFFFB300)

/** The sample's Material theme. */
@Composable
fun IrisTheme(content: @Composable () -> Unit) {
    MaterialTheme(colors = lightColors(primary = Indigo, primaryVariant = DeepIndigo, secondary = Amber), content = content)
}

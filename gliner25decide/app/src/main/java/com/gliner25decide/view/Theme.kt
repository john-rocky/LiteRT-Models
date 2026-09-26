package com.gliner25decide.view

import androidx.compose.material.MaterialTheme
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable

/** Material 1 theme shared by the sample's composables. */
@Composable
fun ApplicationTheme(content: @Composable () -> Unit) {
  MaterialTheme(
    colors = lightColors(primary = darkBlue, secondary = teal),
    content = content,
  )
}

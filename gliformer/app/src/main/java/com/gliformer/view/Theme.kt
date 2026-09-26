package com.gliformer.view

import androidx.compose.material.MaterialTheme
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable

@Composable
fun ApplicationTheme(content: @Composable () -> Unit) {
  MaterialTheme(
    colors = lightColors(primary = darkBlue, secondary = teal),
    content = content,
  )
}

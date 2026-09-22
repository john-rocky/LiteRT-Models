// SPDX-License-Identifier: Apache-2.0
package com.sopro.view

import androidx.compose.material.MaterialTheme
import androidx.compose.material.lightColors
import androidx.compose.runtime.Composable

/** Shared Material 1 theme for the sample. */
@Composable
fun ApplicationTheme(content: @Composable () -> Unit) {
  MaterialTheme(colors = lightColors(primary = darkBlue, secondary = teal), content = content)
}

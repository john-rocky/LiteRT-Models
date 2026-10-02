package com.gliclass

/**
 * The label editor's format: one label per line, or a single line of comma-separated labels. Labels
 * are trimmed and empty entries dropped; the request keeps the remaining order.
 */
object LabelEditor {
  /** Labels of the editor text [text]. */
  fun parse(text: String): List<String> {
    val parts = if (text.contains('\n')) text.split('\n') else text.split(',')
    return parts.map { it.trim() }.filter { it.isNotEmpty() }
  }
}

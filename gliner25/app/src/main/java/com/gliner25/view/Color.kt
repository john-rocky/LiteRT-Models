package com.gliner25.view

import androidx.annotation.StringRes
import androidx.compose.ui.graphics.Color
import com.gliner25.R

val darkBlue = Color(0xFF174378)
val teal = Color(0xFF00796B)

enum class EntityLabel(val key: String, @param:StringRes val title: Int, val color: Color) {
  PERSON("person", R.string.label_person, Color(0xFF145DA0)),
  ORGANIZATION("organization", R.string.label_organization, Color(0xFF6B3FA0)),
  LOCATION("location", R.string.label_location, Color(0xFF16724C)),
  PRODUCT("product", R.string.label_product, Color(0xFF9A5100)),
  DATE("date", R.string.label_date, Color(0xFFAB3253));

  companion object {
    fun fromKey(key: String): EntityLabel = entries.first { it.key == key }
  }
}

// Copyright 2026 Daisuke Majima. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Tabular digits in any font: every digit is drawn centred in a cell as wide
// as the font's widest digit, so a running stopwatch never shifts sideways and
// a column of seconds lines up — whether or not the device font has tabular
// figures (the phone's system font is not the one this was written against).

package com.bonsai.imagegen

import android.graphics.Canvas
import android.graphics.Paint
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ReplacementSpan
import kotlin.math.ceil

class TabularSpan : ReplacementSpan() {

    override fun getSize(paint: Paint, text: CharSequence, start: Int, end: Int, fm: Paint.FontMetricsInt?): Int {
        if (fm != null) paint.getFontMetricsInt(fm)
        val cell = cell(paint)
        var w = 0f
        for (i in start until end) w += if (text[i].isDigit()) cell else paint.measureText(text, i, i + 1)
        return ceil(w).toInt()
    }

    override fun draw(
        canvas: Canvas, text: CharSequence, start: Int, end: Int, x: Float, top: Int, y: Int, bottom: Int,
        paint: Paint,
    ) {
        val cell = cell(paint)
        var cx = x
        for (i in start until end) {
            val w = paint.measureText(text, i, i + 1)
            if (text[i].isDigit()) {
                canvas.drawText(text, i, i + 1, cx + (cell - w) / 2, y.toFloat(), paint)
                cx += cell
            } else {
                canvas.drawText(text, i, i + 1, cx, y.toFloat(), paint)
                cx += w
            }
        }
    }

    private fun cell(paint: Paint): Float = DIGITS.maxOf { paint.measureText(it) }

    companion object {
        private val DIGITS = (0..9).map { it.toString() }

        /** [s] with every digit in a fixed-width cell. */
        fun of(s: String): CharSequence =
            SpannableString(s).apply { setSpan(TabularSpan(), 0, s.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE) }
    }
}

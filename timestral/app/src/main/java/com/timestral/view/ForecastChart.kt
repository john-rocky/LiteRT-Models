package com.timestral.view

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.timestral.ForecastChartData
import com.timestral.MainViewModel

/**
 * The series' last values (gray, gaps where values are missing), then the forecast: the median and
 * the 10–90 % band start at the last observed value (the band at zero width), and the dashed line
 * marks the series' last step. The y axis covers every value drawn; three labeled grid lines.
 */
@Composable
fun ForecastChart(data: ForecastChartData, modifier: Modifier = Modifier) {
  Canvas(modifier) {
    val history = data.history
    val total = history.size + data.median.size
    var lo = Float.POSITIVE_INFINITY
    var hi = Float.NEGATIVE_INFINITY
    for (values in listOf(history, data.median, data.low, data.high)) {
      for (v in values) {
        if (v.isFinite()) {
          lo = minOf(lo, v)
          hi = maxOf(hi, v)
        }
      }
    }
    if (!lo.isFinite() || total < 2) return@Canvas
    if (hi - lo < 1e-6f * maxOf(1f, Math.abs(hi))) {
      val pad = maxOf(Math.abs(hi) * 0.01f, 1e-3f)
      lo -= pad
      hi += pad
    }
    val margin = (hi - lo) * 0.08f
    lo -= margin
    hi += margin
    val left = 64.dp.toPx()
    val right = size.width - 8.dp.toPx()
    val top = 8.dp.toPx()
    val bottom = size.height - 8.dp.toPx()
    fun x(index: Int) = left + (right - left) * index / (total - 1).toFloat()
    fun y(value: Float) = bottom - (bottom - top) * (value - lo) / (hi - lo)

    drawGrid(lo, hi, left, right, ::y)
    val start = history.size
    if (start > 0) {
      val now = x(start - 1)
      drawLine(
        Color.LightGray,
        Offset(now, top),
        Offset(now, bottom),
        strokeWidth = 1.dp.toPx(),
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 6f)),
      )
    }
    // The last observed value: the band and the median line start there, across any missing steps.
    val last = history.indexOfLast { it.isFinite() }
    val anchor = if (last >= 0) Offset(x(last), y(history[last])) else null
    val band = Path()
    anchor?.let { band.moveTo(it.x, it.y) }
    for (i in data.high.indices) {
      if (i == 0 && anchor == null) band.moveTo(x(start), y(data.high[0]))
      else band.lineTo(x(start + i), y(data.high[i]))
    }
    for (i in data.low.indices.reversed()) band.lineTo(x(start + i), y(data.low[i]))
    band.close()
    drawPath(band, bandBlue)
    if (data.median.size == 1 && anchor == null) {
      drawLine(medianBlue, Offset(x(start), y(data.low[0])), Offset(x(start), y(data.high[0])), 2.dp.toPx())
    }
    drawSeries(history, 0, historyGray, ::x, ::y)
    drawSeries(data.median, start, medianBlue, ::x, ::y, from = anchor)
  }
}

private fun DrawScope.drawSeries(
  values: FloatArray,
  offset: Int,
  color: Color,
  x: (Int) -> Float,
  y: (Float) -> Float,
  from: Offset? = null,
) {
  val path = Path()
  from?.let { path.moveTo(it.x, it.y) }
  var open = from != null
  var points = 0
  for (i in values.indices) {
    val v = values[i]
    if (!v.isFinite()) {
      open = false
      continue
    }
    if (open) path.lineTo(x(offset + i), y(v)) else path.moveTo(x(offset + i), y(v))
    open = true
    points++
  }
  drawPath(path, color, style = Stroke(width = 1.5.dp.toPx()))
  if (points == 1 && from == null) {
    val i = values.indexOfFirst { it.isFinite() }
    drawCircle(color, 3.dp.toPx(), Offset(x(offset + i), y(values[i])))
  }
}

private fun DrawScope.drawGrid(lo: Float, hi: Float, left: Float, right: Float, y: (Float) -> Float) {
  val paint =
    Paint().apply {
      color = Color.DarkGray.toArgb()
      textSize = 11.sp.toPx()
      isAntiAlias = true
      textAlign = Paint.Align.RIGHT
    }
  for (step in 0..2) {
    val value = lo + (hi - lo) * (0.1f + 0.4f * step)
    val py = y(value)
    drawLine(Color(0x22000000), Offset(left, py), Offset(right, py), strokeWidth = 1f)
    drawIntoCanvas {
      it.nativeCanvas.drawText(MainViewModel.value(value), left - 6.dp.toPx(), py + 4.dp.toPx(), paint)
    }
  }
}

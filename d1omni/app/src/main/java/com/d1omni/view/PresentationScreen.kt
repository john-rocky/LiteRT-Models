package com.d1omni.view

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.LocalTextStyle
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.sp
import com.d1omni.D1CardState
import com.d1omni.D1CardUi
import com.d1omni.D1DemoLayout
import com.d1omni.D1DemoRun
import com.d1omni.D1InboxLayout as L
import com.d1omni.D1InboxPlan
import com.d1omni.D1InboxText
import com.d1omni.D1Kind
import com.d1omni.D1Pill
import com.d1omni.D1PresentationUi
import com.d1omni.D1RowUi
import com.d1omni.D1TextMetrics

/**
 * The inbox demo's presentation, inside the centred 9:16 band of the screen (y 210 to 2130 on a
 * 1080 × 2340 screen): the title and the state pill, one card per item (a state indicator, the
 * header with the media graph's ms, the voice note's playback bar / the photo / the message, one
 * row per question that fills in when its call returns, the item's total), the footer. Every size
 * is fixed before the first frame by [D1InboxLayout.plan] with the device's own fonts, so nothing
 * moves or wraps while the run fills the rows in. Where the pill and the indicators were drawn goes
 * to [onLayout] for the run JSON.
 */
@Composable
fun PresentationScreen(ui: D1PresentationUi, onLayout: (D1DemoLayout) -> Unit, onLeave: () -> Unit) {
  BackHandler(enabled = !ui.running, onBack = onLeave)
  val view = LocalView.current
  val density = LocalDensity.current
  val measurer = rememberTextMeasurer()
  val base = LocalTextStyle.current
  BoxWithConstraints(Modifier.fillMaxSize().background(Background)) {
    val width = constraints.maxWidth
    val height = constraints.maxHeight
    val plan =
      remember(ui.runId, width, height, density) {
        L.plan(width, height, density.density, density.fontScale, ui.layoutInput, ComposeMetrics(measurer, base))
      }
    val store = remember(ui.runId, plan) { LayoutStore(ui.cards.map { it.item }) }
    fun report() {
      val location = IntArray(2).also { view.rootView.getLocationOnScreen(it) }
      store.toLayout(view.rootView.width, view.rootView.height, location, plan, density)?.let(onLayout)
    }
    val left = (width - plan.bandWidth) / 2
    Column(
      Modifier.offset { IntOffset(left, plan.bandTop) }
        .width(plan.dp(plan.bandWidth))
        .height(plan.dp(plan.bandHeight))
        .padding(horizontal = plan.dp(plan.px(L.PAGE_PAD_H_DP)), vertical = plan.dp(plan.px(L.BAND_PAD_V_DP))),
      verticalArrangement = Arrangement.spacedBy(plan.dp(plan.px(L.GAP_DP))),
    ) {
      TitleRow(ui, plan, store, ::report)
      for (card in ui.cards) CardView(card, plan, store, ::report)
      Footer(ui.footer, plan, store, ::report)
    }
    ui.failure?.let { reason ->
      Text(
        reason,
        color = FailureRed,
        fontSize = 14.sp,
        modifier = Modifier.offset { IntOffset(left + plan.px(L.PAGE_PAD_H_DP), plan.bandTop + plan.bandHeight - plan.footerPx) },
      )
    }
  }
}

@Composable
private fun TitleRow(ui: D1PresentationUi, plan: D1InboxPlan, store: LayoutStore, report: () -> Unit) {
  Row(Modifier.fillMaxWidth().height(plan.dp(plan.titleRowPx)), verticalAlignment = Alignment.CenterVertically) {
    Text(
      ui.title,
      style = style(plan.titleSp, true, plan),
      color = Ink,
      maxLines = 1,
      softWrap = false,
      modifier =
        Modifier.weight(1f).onGloballyPositioned {
          store.title = it.boundsTopBottom()
          report()
        },
    )
    Box(
      Modifier.width(plan.dp(plan.pillWidthPx))
        .height(plan.dp(plan.px(L.PILL_HEIGHT_DP)))
        .background(pillColor(ui.pill), RoundedCornerShape(percent = 50))
        .onGloballyPositioned {
          store.pill = it.bounds()
          report()
        },
      contentAlignment = Alignment.CenterStart,
    ) {
      Text(
        D1InboxText.PILL.getValue(ui.pill.wireName),
        style = style(plan.pillSp, true, plan),
        color = Color.White,
        maxLines = 1,
        softWrap = false,
        modifier = Modifier.padding(start = plan.dp(plan.px(L.PILL_PAD_H_DP))),
      )
    }
  }
}

@Composable
private fun CardView(card: D1CardUi, plan: D1InboxPlan, store: LayoutStore, report: () -> Unit) {
  val gap = plan.dp(plan.px(L.CARD_GAP_DP))
  Column(
    Modifier.fillMaxWidth()
      .background(CardBackground, RoundedCornerShape(plan.dp(plan.px(L.CORNER_DP))))
      .padding(horizontal = plan.dp(plan.px(L.CARD_PAD_H_DP)), vertical = plan.dp(plan.px(L.CARD_PAD_V_DP)))
  ) {
    // Header: the state indicator, the item's icon and header, the media graph's ms at the right end.
    val headerPx = maxOf(plan.linePx(plan.headerSp), plan.px(L.INDICATOR_DP))
    Row(Modifier.fillMaxWidth().height(plan.dp(headerPx)), verticalAlignment = Alignment.CenterVertically) {
      Box(
        Modifier.size(plan.dp(plan.px(L.INDICATOR_DP)))
          .background(cardColor(card.state), CircleShape)
          .onGloballyPositioned {
            store.indicators[card.item] = it.bounds()
            report()
          }
      )
      Spacer(Modifier.width(plan.dp(plan.px(L.INDICATOR_GAP_DP))))
      Text(
        "${D1InboxText.icon(card.kind)} ${card.header}",
        style = style(plan.headerSp, true, plan),
        color = Ink,
        maxLines = 1,
        softWrap = false,
        modifier = Modifier.weight(1f),
      )
      Text(
        card.mediaMs.orEmpty(),
        style = style(plan.mediaMsSp, false, plan),
        color = Muted,
        maxLines = 1,
        softWrap = false,
      )
    }
    Spacer(Modifier.height(gap))
    when (card.kind) {
      D1Kind.AUDIO -> PlayBar(card, plan)
      D1Kind.IMAGE ->
        card.thumbnail?.let { bitmap ->
          val heightPx = plan.thumbnailPx
          val widthPx = heightPx * bitmap.width / maxOf(1, bitmap.height)
          Image(
            bitmap,
            contentDescription = card.header,
            contentScale = ContentScale.Fit,
            modifier =
              Modifier.width(plan.dp(widthPx)).height(plan.dp(heightPx)).clip(RoundedCornerShape(plan.dp(plan.px(6f)))),
          )
        } ?: Spacer(Modifier.height(plan.dp(plan.thumbnailPx)))
      D1Kind.TEXT -> {
        val lines = plan.messageLines[card.item] ?: 1
        Box(Modifier.fillMaxWidth().height(plan.dp(lines * plan.linePx(plan.messageSp)))) {
          Text(
            card.message.orEmpty(),
            style = style(plan.messageSp, false, plan),
            color = Ink,
            maxLines = lines,
            overflow = TextOverflow.Clip,
          )
        }
      }
    }
    Spacer(Modifier.height(gap))
    for ((index, row) in card.rows.withIndex()) {
      if (index > 0) Spacer(Modifier.height(plan.dp(plan.px(L.ROW_GAP_DP))))
      AnswerRow(card.item, row, plan)
    }
    Spacer(Modifier.height(gap))
    Box(Modifier.fillMaxWidth().height(plan.dp(plan.linePx(plan.totalSp))), contentAlignment = Alignment.CenterEnd) {
      Text(card.total.orEmpty(), style = style(plan.totalSp, true, plan), color = Muted, maxLines = 1, softWrap = false)
    }
  }
}

/**
 * One question: its name, then (once its call returned) the answer's word, its probability, a bar
 * of that probability and the call's ms. A row whose widest word does not fit beside the numbers
 * takes two lines from the start (the plan decides), so no row ever changes height.
 */
@Composable
private fun AnswerRow(item: String, row: D1RowUi, plan: D1InboxPlan) {
  val gap = plan.dp(plan.px(L.COL_GAP_DP))
  val lines = plan.rowLines["$item/${row.qid}"] ?: 1
  val answerPx = plan.linePx(plan.answerSp)
  val shown = row.shown
  @Composable
  fun Numbers() {
    Text(
      shown?.prob.orEmpty(),
      style = style(plan.probSp, false, plan),
      color = Ink,
      textAlign = TextAlign.End,
      maxLines = 1,
      softWrap = false,
      modifier = Modifier.width(plan.dp(plan.probWidthPx)),
    )
    Spacer(Modifier.width(gap))
    ProbabilityBar(shown?.fraction?.toFloat(), plan)
    Spacer(Modifier.width(gap))
    Text(
      row.ms?.let { D1InboxText.rowMs(it) }.orEmpty(),
      style = style(plan.msSp, false, plan),
      color = Muted,
      textAlign = TextAlign.End,
      maxLines = 1,
      softWrap = false,
      modifier = Modifier.width(plan.dp(plan.msWidthPx)),
    )
  }
  @Composable
  fun NameAndWord(modifier: Modifier) {
    Text(
      row.qid,
      style = style(plan.qnameSp, false, plan),
      color = if (shown == null) Pending else Muted,
      maxLines = 1,
      softWrap = false,
      modifier = Modifier.width(plan.dp(plan.qnameWidthPx)),
    )
    Spacer(Modifier.width(gap))
    Text(
      shown?.answer.orEmpty(),
      style = style(plan.answerSp, true, plan),
      color = Ink,
      maxLines = 1,
      softWrap = false,
      overflow = TextOverflow.Clip,
      modifier = modifier,
    )
  }
  if (lines == 1) {
    Row(Modifier.fillMaxWidth().height(plan.dp(answerPx)), verticalAlignment = Alignment.CenterVertically) {
      NameAndWord(Modifier.weight(1f))
      Spacer(Modifier.width(gap))
      Numbers()
    }
  } else {
    Column(Modifier.fillMaxWidth()) {
      Row(Modifier.fillMaxWidth().height(plan.dp(answerPx)), verticalAlignment = Alignment.CenterVertically) {
        NameAndWord(Modifier.weight(1f))
      }
      Row(
        Modifier.fillMaxWidth().height(plan.dp(plan.linePx(plan.probSp))),
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Spacer(Modifier.weight(1f))
        Numbers()
      }
    }
  }
}

/** A horizontal bar of [fraction] (empty until the row's answer arrives). */
@Composable
private fun ProbabilityBar(fraction: Float?, plan: D1InboxPlan) {
  Canvas(Modifier.width(plan.dp(plan.px(L.BAR_WIDTH_DP))).height(plan.dp(plan.px(L.BAR_HEIGHT_DP)))) {
    val radius = CornerRadius(size.height / 2, size.height / 2)
    drawRoundRect(Track, cornerRadius = radius)
    if (fraction != null && fraction > 0f) {
      drawRoundRect(BarFill, size = Size(maxOf(size.height, size.width * fraction.coerceIn(0f, 1f)), size.height), cornerRadius = radius)
    }
  }
}

/** The voice note's bar: it follows the sound from its start (red), and stays full (grey) after it. */
@Composable
private fun PlayBar(card: D1CardUi, plan: D1InboxPlan) {
  var fraction by remember(card.item) { mutableFloatStateOf(0f) }
  val start = card.playStartNanos
  LaunchedEffect(start, card.playEnded) {
    if (card.playEnded && start != null) {
      fraction = 1f
      return@LaunchedEffect
    }
    if (start == null || card.playDurationNanos <= 0) return@LaunchedEffect
    while (fraction < 1f) {
      withFrameNanos {}
      fraction = ((System.nanoTime() - start).toFloat() / card.playDurationNanos).coerceIn(0f, 1f)
    }
  }
  val fill = if (card.playEnded) PillReady else PillPlaying
  Canvas(Modifier.fillMaxWidth().height(plan.dp(plan.px(L.PROGRESS_DP)))) {
    val radius = CornerRadius(size.height / 2, size.height / 2)
    drawRoundRect(Track, cornerRadius = radius)
    val f = if (card.playEnded && start != null) 1f else fraction
    if (f > 0f) drawRoundRect(fill, size = Size(maxOf(size.height, size.width * f), size.height), cornerRadius = radius)
  }
}

@Composable
private fun Footer(lines: List<String>, plan: D1InboxPlan, store: LayoutStore, report: () -> Unit) {
  val base = plan.linePx(L.FOOTER_SP * plan.scale)
  Column(
    Modifier.fillMaxWidth().onGloballyPositioned {
      store.footer = it.boundsTopBottom()
      report()
    }
  ) {
    for ((index, text) in lines.withIndex()) {
      Box(Modifier.fillMaxWidth().height(plan.dp(base)), contentAlignment = Alignment.CenterStart) {
        Text(
          text,
          style = style(plan.footerSp.getOrElse(index) { L.FOOTER_SP * plan.scale }, false, plan),
          color = Muted,
          maxLines = 1,
          softWrap = false,
        )
      }
    }
  }
}

/** The device's fonts as [D1TextMetrics] for the plan. */
private class ComposeMetrics(private val measurer: TextMeasurer, private val base: TextStyle) : D1TextMetrics {
  private fun styleOf(sp: Float, bold: Boolean) =
    base.merge(TextStyle(fontSize = sp.sp, fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal, lineHeight = (sp * L.LINE).sp))

  override fun width(text: String, sp: Float, bold: Boolean): Float =
    measurer.measure(text, styleOf(sp, bold), maxLines = 1, softWrap = false).size.width.toFloat()

  override fun lines(text: String, sp: Float, bold: Boolean, widthPx: Float): Int =
    measurer.measure(text, styleOf(sp, bold), constraints = Constraints(maxWidth = widthPx.toInt())).lineCount
}

@Composable
private fun style(sp: Float, bold: Boolean, plan: D1InboxPlan): TextStyle =
  LocalTextStyle.current.merge(
    TextStyle(fontSize = sp.sp, fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal, lineHeight = (sp * L.LINE).sp)
  )

/** Window-pixel bounds collected while the presentation lays out. */
private class LayoutStore(private val items: List<String>) {
  var title: IntArray? = null
  var footer: IntArray? = null
  var pill: IntArray? = null
  val indicators = LinkedHashMap<String, IntArray>()

  /** The layout in screen pixels once the title, the pill, the footer and every indicator are placed. */
  fun toLayout(width: Int, height: Int, windowOnScreen: IntArray, plan: D1InboxPlan, density: Density): D1DemoLayout? {
    val title = title ?: return null
    val footer = footer ?: return null
    val pill = pill ?: return null
    if (items.any { it !in indicators }) return null
    val (dx, dy) = windowOnScreen[0] to windowOnScreen[1]
    fun shift(box: IntArray) = intArrayOf(box[0] + dx, box[1] + dy, box[2], box[3])
    return D1DemoLayout(
      width,
      height,
      title[0] + dy,
      footer[1] + dy,
      shift(pill),
      plan.px(L.PILL_PAD_H_DP),
      LinkedHashMap(items.associateWith { shift(indicators.getValue(it)) }),
      with(density) { plan.answerSp.sp.toPx() },
      density.density,
      density.fontScale,
      plan,
    )
  }
}

/** Top and bottom in window pixels. */
private fun LayoutCoordinates.boundsTopBottom(): IntArray =
  boundsInWindow().let { intArrayOf(it.top.toInt(), it.bottom.toInt()) }

/** Left, top, width and height in window pixels. */
private fun LayoutCoordinates.bounds(): IntArray =
  boundsInWindow().let { intArrayOf(it.left.toInt(), it.top.toInt(), it.width.toInt(), it.height.toInt()) }

/** [px] as dp at the plan's density. */
private fun D1InboxPlan.dp(px: Int): Dp = Dp(px / density)

private fun hex(value: String): Color = Color(android.graphics.Color.parseColor(value))

private fun pillColor(pill: D1Pill): Color = hex(D1DemoRun.PILL_PALETTE.getValue(pill.wireName))

private fun cardColor(state: D1CardState): Color =
  when (state) {
    D1CardState.FAILED -> FailureRed
    else -> hex(D1DemoRun.CARD_PALETTE.getValue(state.wireName))
  }

private val Background = Color.White
private val CardBackground = Color(0xFFF1F3F4)
private val Ink = Color(0xFF202124)
private val Muted = Color(0xFF5F6368)
private val Pending = Color(0xFF9AA0A6)
private val Track = Color(0xFFDADCE0)
private val BarFill = Color(0xFF1565C0)
private val PillPlaying = Color(0xFFE53935)
private val PillReady = Color(0xFF5F6368)
private val FailureRed = Color(0xFFB00020)

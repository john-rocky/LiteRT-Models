package com.kev.view

import androidx.activity.compose.BackHandler
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
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.layout.boundsInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.kev.AnswerCardUi
import com.kev.CardState
import com.kev.KevDemoLayout
import com.kev.PresentationUi
import com.kev.QuestionType
import com.kev.R

/**
 * The read-only demo layout of an autoplay run, inside the centred 9:16 band of the screen (y from
 * 210 to 2130 on a 1080 × 2340 screen): title, the ticket (state), one card per question that turns
 * grey → blue → green, and the footer. Card heights do not change between states, and the ticket
 * shrinks from [TICKET_SP] until everything fits, so nothing moves while the questions run. Where
 * things were drawn goes to [onLayout] for the run JSON.
 */
@Composable
fun PresentationScreen(ui: PresentationUi, onLayout: (KevDemoLayout) -> Unit, onLeave: () -> Unit) {
  BackHandler(onBack = onLeave)
  val view = LocalView.current
  val density = LocalDensity.current
  val qids = ui.cards.map { it.qid }
  val store = remember(qids) { LayoutStore(qids) }
  var ticketSp by remember(ui.ticket) { mutableFloatStateOf(TICKET_SP) }
  fun report() {
    val location = IntArray(2).also { view.rootView.getLocationOnScreen(it) }
    store.toLayout(
      view.rootView.width,
      view.rootView.height,
      location,
      with(density) { ticketSp.sp.toPx() },
      with(density) { ANSWER_SP.sp.toPx() },
      density.density,
      density.fontScale,
    )?.let(onLayout)
  }
  BoxWithConstraints(Modifier.fillMaxSize().background(Color.White)) {
    val band = minOf(constraints.maxHeight, constraints.maxWidth * BAND_HEIGHT / BAND_WIDTH)
    val top = (constraints.maxHeight - band) / 2
    Column(
      Modifier.offset { IntOffset(0, top) }
        .height(with(density) { band.toDp() })
        .fillMaxWidth()
        .padding(horizontal = 24.dp, vertical = 8.dp),
      verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
      Text(
        ui.title,
        fontSize = TITLE_SP.sp,
        fontWeight = FontWeight.Bold,
        color = MaterialTheme.colors.primary,
        modifier = Modifier.onGloballyPositioned { store.title = it.boundsTopBottom(); report() },
      )
      Text(
        ui.ticket,
        fontSize = ticketSp.sp,
        lineHeight = (ticketSp * LINE_HEIGHT).sp,
        overflow = TextOverflow.Clip,
        modifier = Modifier.weight(1f).fillMaxWidth(),
        onTextLayout = { result ->
          if (result.hasVisualOverflow && ticketSp > MIN_TICKET_SP) ticketSp -= 1f else report()
        },
      )
      ui.failure?.let { Text(it, fontSize = QUESTION_SP.sp, color = MaterialTheme.colors.error) }
      ui.cards.forEach { card ->
        PresentationCard(card) { indicator -> store.indicators[card.qid] = indicator; report() }
      }
      // Room for the footer lines even when they wrap, so the total arriving moves nothing.
      Box(Modifier.fillMaxWidth().height(with(density) { (FOOTER_SP * LINE_HEIGHT * FOOTER_LINES).sp.toDp() })) {
        Column(Modifier.onGloballyPositioned { store.footer = it.boundsTopBottom(); report() }) {
          ui.footerLines.forEach {
            Text(it, fontSize = FOOTER_SP.sp, lineHeight = (FOOTER_SP * LINE_HEIGHT).sp, color = Color.DarkGray)
          }
        }
      }
    }
  }
}

/** A card of fixed height: the question, then one answer line and its bar in every state. */
@Composable
private fun PresentationCard(card: AnswerCardUi, onIndicator: (IntArray) -> Unit) {
  val view = card.view?.takeIf { card.state == CardState.DONE }
  Row(verticalAlignment = Alignment.Top) {
    Box(
      Modifier.padding(top = 4.dp)
        .size(INDICATOR_DP.dp)
        .background(stateColor(card.state), CircleShape)
        .onGloballyPositioned { onIndicator(it.bounds()) }
    )
    Spacer(Modifier.width(14.dp))
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
      Text(card.question, fontSize = QUESTION_SP.sp, maxLines = 2, overflow = TextOverflow.Ellipsis, color = Color.DarkGray)
      Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
          when {
            card.state == CardState.FAILED -> card.error.orEmpty()
            view?.type == QuestionType.NOUL -> stringResource(R.string.presentation_noul)
            view != null -> view.headline.label
            else -> ELLIPSIS
          },
          fontSize = ANSWER_SP.sp,
          fontWeight = FontWeight.Bold,
          maxLines = 1,
          overflow = TextOverflow.Ellipsis,
          modifier = Modifier.weight(1f),
        )
        Text(view?.headline?.value.orEmpty(), fontSize = ANSWER_SP.sp, fontFamily = FontFamily.Monospace)
        Spacer(Modifier.width(16.dp))
        Text(card.msText.orEmpty(), fontSize = QUESTION_SP.sp, color = Color.DarkGray)
      }
      if (card.state == CardState.RUNNING) {
        LinearProgressIndicator(Modifier.fillMaxWidth().height(BAR_DP.dp))
      } else {
        LinearProgressIndicator(progress = view?.headline?.fraction ?: 0f, modifier = Modifier.fillMaxWidth().height(BAR_DP.dp))
      }
    }
  }
}

/** Window-pixel bounds collected while the presentation of the cards [qids] lays out. */
private class LayoutStore(private val qids: List<String>) {
  var title: IntArray? = null
  var footer: IntArray? = null
  val indicators = LinkedHashMap<String, IntArray>()

  /** The layout in screen pixels, once the title and the footer have been placed. */
  fun toLayout(
    width: Int,
    height: Int,
    windowOnScreen: IntArray,
    ticketPx: Float,
    answerPx: Float,
    density: Float,
    fontScale: Float,
  ): KevDemoLayout? {
    val title = title ?: return null
    val footer = footer ?: return null
    val (dx, dy) = windowOnScreen[0] to windowOnScreen[1]
    return KevDemoLayout(
      width,
      height,
      title[0] + dy,
      footer[1] + dy,
      qids.mapNotNull { qid ->
        indicators[qid]?.let { box -> KevDemoLayout.Card(qid, box[0] + dx, box[1] + dy, box[2], box[3], answerPx) }
      },
      ticketPx,
      density,
      fontScale,
    )
  }
}

/** Top and bottom in window pixels. */
private fun LayoutCoordinates.boundsTopBottom(): IntArray =
  boundsInWindow().let { intArrayOf(it.top.toInt(), it.bottom.toInt()) }

/** Left, top, width and height in window pixels. */
private fun LayoutCoordinates.bounds(): IntArray =
  boundsInWindow().let { intArrayOf(it.left.toInt(), it.top.toInt(), it.width.toInt(), it.height.toInt()) }

private const val TITLE_SP = 30f
private const val TICKET_SP = 26f
private const val MIN_TICKET_SP = 16f
private const val QUESTION_SP = 20f
private const val ANSWER_SP = 26f
private const val FOOTER_SP = 15f
private const val FOOTER_LINES = 4
private const val LINE_HEIGHT = 1.25f
private const val INDICATOR_DP = 28
private const val BAR_DP = 6
private const val BAND_WIDTH = 9
private const val BAND_HEIGHT = 16
private const val ELLIPSIS = "…"

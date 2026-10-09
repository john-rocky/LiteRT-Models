package com.d1omni

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Text widths and line counts in pixels: the device's fonts (Compose `TextMeasurer`) or a JVM stand-in. */
interface D1TextMetrics {
  /** Width in px of [text] on one line at [sp] (bold when [bold]); the screen's density and font scale applied. */
  fun width(text: String, sp: Float, bold: Boolean): Float

  /** Lines [text] takes at [sp] wrapped to [widthPx]. */
  fun lines(text: String, sp: Float, bold: Boolean, widthPx: Float): Int
}

/** What the presentation lays out: the title, one card per item, the footer's lines. */
class D1InboxLayoutInput(val title: String, val cards: List<Card>, val footer: List<String>) {
  /**
   * One item's card: its [header] ("Voice note · 8.7 s"), the widest text its header's right end can
   * show ([mediaMsWidest]), a [message] (text items), its question rows and the widest total line.
   */
  class Card(
    val item: String,
    val kind: D1Kind,
    val header: String,
    val mediaMsWidest: String?,
    val message: String?,
    val rows: List<Row>,
    val totalWidest: String,
  )

  /** One question row: its name and every answer word it can show. */
  class Row(val qid: String, val words: List<String>)

  companion object {
    /** The input of a fixture: [headers] by item (the audio clip's length known), the footer lines. */
    fun of(fixture: D1InboxFixture, headers: Map<String, String>, footer: List<String>): D1InboxLayoutInput =
      D1InboxLayoutInput(
        fixture.title,
        fixture.items.map { item ->
          Card(
            item.item,
            item.kind,
            headers[item.item] ?: item.header,
            D1InboxText.mediaMsWidest(item.kind),
            if (item.kind == D1Kind.TEXT) D1Prompt.serialize(item.state) else null,
            item.questions.map { (name, question) -> Row(name, D1InboxAnswer.words(question)) },
            D1InboxText.itemTotalWidest(item.questions.size),
          )
        },
        footer,
      )
  }
}

/** One text's size as drawn and whether it fits its room at that size (the run JSON's `layout.texts`). */
data class D1TextFit(val id: String, val sp: Float, val fits: Boolean)

/**
 * The presentation's sizes, fixed before anything is drawn: one [scale] for every text (the largest
 * of [D1InboxLayout.SCALES] at which everything fits the 9:16 band with a photo of at least
 * [D1InboxLayout.THUMB_MIN_DP]), each footer line's own size (shrunk alone when it is wider than the
 * band), the line count of every answer row (two when its widest word does not fit beside the
 * probability, the bar and the ms) and of each message, the photo's height (the room left, at most
 * [D1InboxLayout.THUMB_MAX_DP]), the columns' widths and every block's height in px.
 */
class D1InboxPlan(
  val scale: Float,
  val bandTop: Int,
  val bandHeight: Int,
  val bandWidth: Int,
  val density: Float,
  val fontScale: Float,
  val titleSp: Float,
  val pillSp: Float,
  val headerSp: Float,
  val mediaMsSp: Float,
  val qnameSp: Float,
  val answerSp: Float,
  val probSp: Float,
  val msSp: Float,
  val totalSp: Float,
  val messageSp: Float,
  val footerSp: List<Float>,
  /** Lines of each question row by "item/qid": 1 or 2. */
  val rowLines: Map<String, Int>,
  /** Lines of each message by item. */
  val messageLines: Map<String, Int>,
  val thumbnailPx: Int,
  val pillWidthPx: Int,
  val qnameWidthPx: Int,
  val probWidthPx: Int,
  val msWidthPx: Int,
  /** Each card's height in px by item, and the title row's, the footer's and the whole column's. */
  val cardHeightPx: Map<String, Int>,
  val titleRowPx: Int,
  val footerPx: Int,
  val contentPx: Int,
  val fits: Boolean,
  val texts: List<D1TextFit>,
) {
  fun px(dp: Float): Int = (dp * density).roundToInt()

  /** Line height in px of a text of [sp]. */
  fun linePx(sp: Float): Int = (sp * fontScale * density * D1InboxLayout.LINE).roundToInt()

  fun rowPx(item: String, qid: String): Int =
    if ((rowLines["$item/$qid"] ?: 1) == 1) linePx(answerSp) else linePx(answerSp) + linePx(probSp)

  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "scale" to scale,
      "band_px" to linkedMapOf("top" to bandTop, "height" to bandHeight, "width" to bandWidth),
      "content_height_px" to contentPx,
      "fits" to fits,
      "sp" to
        linkedMapOf(
          "title" to titleSp,
          "pill" to pillSp,
          "header" to headerSp,
          "media_ms" to mediaMsSp,
          "question" to qnameSp,
          "answer" to answerSp,
          "prob" to probSp,
          "ms" to msSp,
          "total" to totalSp,
          "message" to messageSp,
          "footer" to footerSp,
        ),
      "row_lines" to rowLines,
      "message_lines" to messageLines,
      "thumbnail_px" to thumbnailPx,
      "texts" to texts.map { linkedMapOf("id" to it.id, "sp" to it.sp, "fits" to it.fits) },
    )
}

/**
 * The presentation's layout, Android-free (the JVM tests run it with a stand-in for the fonts): the
 * centred 9:16 band of the screen (y 210 to 2130 on a 1080 × 2340 screen) holds, from the top, the
 * title and the state pill, one card per item and the three footer lines. Sizes are in dp and sp at
 * scale 1; [plan] picks the scale.
 */
object D1InboxLayout {
  const val PAGE_PAD_H_DP = 12f
  const val BAND_PAD_V_DP = 4f
  const val GAP_DP = 6f
  const val CARD_PAD_H_DP = 8f
  const val CARD_PAD_V_DP = 6f
  const val CARD_GAP_DP = 4f
  const val ROW_GAP_DP = 2f
  const val PROGRESS_DP = 6f
  const val INDICATOR_DP = 16f
  const val INDICATOR_GAP_DP = 8f
  const val PILL_HEIGHT_DP = 24f
  const val PILL_PAD_H_DP = 10f
  const val BAR_WIDTH_DP = 44f
  const val BAR_HEIGHT_DP = 6f
  const val COL_GAP_DP = 6f
  const val THUMB_MIN_DP = 72f
  const val THUMB_MAX_DP = 120f
  const val CORNER_DP = 10f

  /** Line height as a multiple of the text size. */
  const val LINE = 1.25f

  const val TITLE_SP = 20f
  const val PILL_SP = 12f
  const val HEADER_SP = 15f
  const val MEDIA_MS_SP = 13f
  const val QNAME_SP = 13f
  const val ANSWER_SP = 16f
  const val PROB_SP = 15f
  const val MS_SP = 13f
  const val TOTAL_SP = 13f
  const val MESSAGE_SP = 15f
  const val FOOTER_SP = 13f
  const val FOOTER_MIN_SP = 10f
  const val FOOTER_STEP_SP = 0.5f

  /** The scales tried, largest first. */
  val SCALES: List<Float> = (0..12).map { 1.30f - 0.05f * it }

  const val BAND_WIDTH = 9
  const val BAND_HEIGHT = 16

  /** The centred 9:16 band of a [width] × [height] px screen: its top and its height. */
  fun band(width: Int, height: Int): Pair<Int, Int> {
    val h = min(height, width * BAND_HEIGHT / BAND_WIDTH)
    return (height - h) / 2 to h
  }

  /** A column's width in whole px: the text's width rounded up, plus one px. */
  private fun column(width: Float): Int = kotlin.math.ceil(width).toInt() + 1

  /** The plan at the largest scale that fits; the smallest scale's plan (fits false) when none does. */
  fun plan(
    width: Int,
    height: Int,
    density: Float,
    fontScale: Float,
    input: D1InboxLayoutInput,
    metrics: D1TextMetrics,
  ): D1InboxPlan {
    for (scale in SCALES) {
      val plan = attempt(scale, width, height, density, fontScale, input, metrics)
      if (plan.fits) return plan
    }
    return attempt(SCALES.last(), width, height, density, fontScale, input, metrics)
  }

  private fun attempt(
    scale: Float,
    width: Int,
    height: Int,
    density: Float,
    fontScale: Float,
    input: D1InboxLayoutInput,
    metrics: D1TextMetrics,
  ): D1InboxPlan {
    fun dp(value: Float) = value * density
    fun line(sp: Float) = (sp * fontScale * density * LINE).roundToInt()
    val (top, band) = band(width, height)
    val bandWidth = min(width, height * BAND_WIDTH / BAND_HEIGHT)
    // Widths in whole px as the screen lays them out: the column boxes are these sizes exactly.
    val content = (bandWidth - 2 * dp(PAGE_PAD_H_DP).roundToInt()).toFloat()
    val inner = content - 2 * dp(CARD_PAD_H_DP).roundToInt()
    val gap = dp(COL_GAP_DP).roundToInt()
    val title = TITLE_SP * scale
    val pill = PILL_SP * scale
    val header = HEADER_SP * scale
    val mediaMs = MEDIA_MS_SP * scale
    val qname = QNAME_SP * scale
    val answer = ANSWER_SP * scale
    val prob = PROB_SP * scale
    val ms = MS_SP * scale
    val total = TOTAL_SP * scale
    val message = MESSAGE_SP * scale
    val footerBase = FOOTER_SP * scale
    val texts = ArrayList<D1TextFit>()
    var fits = true
    fun record(id: String, sp: Float, ok: Boolean) {
      texts.add(D1TextFit(id, sp, ok))
      if (!ok) fits = false
    }
    // The title row: the title, then the pill (as wide as its widest label, so it never changes size).
    val pillWidth = D1InboxText.PILL.values.maxOf { metrics.width(it, pill, true) } + 2 * dp(PILL_PAD_H_DP)
    val titleWidth = metrics.width(input.title, title, true)
    record("title", title, titleWidth + gap + pillWidth <= content)
    record("pill", pill, pillWidth <= content)
    // The answer rows' columns: question name, answer word (the rest), probability, bar, ms.
    val qnameCol = column(input.cards.flatMap { it.rows }.maxOfOrNull { metrics.width(it.qid, qname, false) } ?: 0f)
    val probCol = column(metrics.width(D1InboxText.PROB_WIDEST, prob, false))
    val msCol = column(metrics.width(D1InboxText.ROW_MS_WIDEST, ms, false))
    val oneLine = inner - qnameCol - probCol - dp(BAR_WIDTH_DP).roundToInt() - msCol - 4 * gap
    val twoLines = inner - qnameCol - gap
    val rowLines = LinkedHashMap<String, Int>()
    val messageLines = LinkedHashMap<String, Int>()
    val cardHeights = LinkedHashMap<String, Int>()
    var fixed = line(title).coerceAtLeast(dp(PILL_HEIGHT_DP).roundToInt()) + 2 * dp(BAND_PAD_V_DP).roundToInt()
    val titleRow = line(title).coerceAtLeast(dp(PILL_HEIGHT_DP).roundToInt())
    var images = 0
    for (card in input.cards) {
      val headerText = "${D1InboxText.icon(card.kind)} ${card.header}"
      val headerWidth =
        dp(INDICATOR_DP) + dp(INDICATOR_GAP_DP) + metrics.width(headerText, header, true) +
          (card.mediaMsWidest?.let { gap + metrics.width(it, mediaMs, false) } ?: 0f)
      record("${card.item}/header", header, headerWidth <= inner)
      record("${card.item}/total", total, metrics.width(card.totalWidest, total, false) <= inner)
      var rows = 0
      for ((index, row) in card.rows.withIndex()) {
        val widest = row.words.maxOf { metrics.width(it, answer, true) }
        val lines =
          when {
            widest <= oneLine -> 1
            widest <= twoLines -> 2
            else -> 2
          }
        record("${card.item}/${row.qid}/answer", answer, widest <= twoLines)
        rowLines["${card.item}/${row.qid}"] = lines
        rows += if (lines == 1) line(answer) else line(answer) + line(prob)
        if (index > 0) rows += dp(ROW_GAP_DP).roundToInt()
      }
      var body = 2 * dp(CARD_PAD_V_DP).roundToInt() + line(header).coerceAtLeast(dp(INDICATOR_DP).roundToInt()) +
        3 * dp(CARD_GAP_DP).roundToInt() + rows + line(total)
      when (card.kind) {
        D1Kind.AUDIO -> body += dp(PROGRESS_DP).roundToInt()
        D1Kind.IMAGE -> images++
        D1Kind.TEXT -> {
          val text = card.message.orEmpty()
          val lines = metrics.lines(text, message, false, inner).coerceAtLeast(1)
          messageLines[card.item] = lines
          record("${card.item}/message", message, true)
          body += lines * line(message)
        }
      }
      cardHeights[card.item] = body
      fixed += body
    }
    fixed += (input.cards.size + 1) * dp(GAP_DP).roundToInt()
    // The footer keeps three lines at its base size; each line shrinks alone, in steps, until it fits.
    val footerSp = ArrayList<Float>()
    for ((index, text) in input.footer.withIndex()) {
      var sp = footerBase
      while (metrics.width(text, sp, false) > content && sp - FOOTER_STEP_SP >= FOOTER_MIN_SP - 1e-4f) sp -= FOOTER_STEP_SP
      record("footer/${index + 1}", sp, metrics.width(text, sp, false) <= content)
      footerSp.add(sp)
    }
    val footer = input.footer.size * line(footerBase)
    fixed += footer
    val spare = band - fixed
    val thumbnail =
      if (images == 0) 0 else min(floor(spare.toFloat() / images).toInt(), dp(THUMB_MAX_DP).roundToInt())
    if (images > 0 && thumbnail < dp(THUMB_MIN_DP)) fits = false
    if (images == 0 && spare < 0) fits = false
    for (card in input.cards) {
      if (card.kind == D1Kind.IMAGE) cardHeights[card.item] = (cardHeights[card.item] ?: 0) + max(thumbnail, 0)
    }
    val contentHeight = fixed + max(thumbnail, 0) * images
    return D1InboxPlan(
      scale,
      top,
      band,
      bandWidth,
      density,
      fontScale,
      title,
      pill,
      header,
      mediaMs,
      qname,
      answer,
      prob,
      ms,
      total,
      message,
      footerSp,
      rowLines,
      messageLines,
      max(thumbnail, 0),
      kotlin.math.ceil(pillWidth).toInt(),
      qnameCol,
      probCol,
      msCol,
      cardHeights,
      titleRow,
      footer,
      contentHeight,
      fits && contentHeight <= band,
      texts,
    )
  }
}

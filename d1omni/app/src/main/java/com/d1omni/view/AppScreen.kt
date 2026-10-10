package com.d1omni.view

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import com.d1omni.MainViewModel
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Tab
import androidx.compose.material.TabRow
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
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
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.graphics.toColorInt
import com.d1omni.AnswerUi
import com.d1omni.D1AnswerLayout
import com.d1omni.D1Input
import com.d1omni.D1Source
import com.d1omni.D1Text
import com.d1omni.RecentPhoto
import com.d1omni.ResultUi
import com.d1omni.SummaryUi
import com.d1omni.UiState

/** What the screen can ask for. */
class AppActions(
  val onTab: (Int) -> Unit,
  val onLoadSample: () -> Unit,
  val onRecord: () -> Unit,
  val onStop: () -> Unit,
  val onPickAudio: () -> Unit,
  val onPlay: () -> Unit,
  val onPickPhoto: () -> Unit,
  val onRecentPhoto: (RecentPhoto) -> Unit,
  val onBrowsePhotos: () -> Unit,
  val onCloseRecent: () -> Unit,
  val onMessage: (String) -> Unit,
  val onDecide: (D1Input) -> Unit,
  val editor: EditorActions,
  val onPill: (IntArray, Int) -> Unit,
  val onAnswerLayout: (D1Input, List<D1AnswerLayout>) -> Unit,
)

/** The answer's word and its probability are drawn this large (the screen must read on a phone video). */
const val ANSWER_SP = 34f
const val PROB_SP = 28f
private const val PILL_PAD_DP = 12

/**
 * The app's one screen: the title and the state pill, four tabs (Voice, Photo, Message, Summary), and on each input tab
 * the input, its questions (folded; Edit opens the editor), Decide and the answers in large type with the ms they took.
 */
@Composable
fun AppScreen(state: UiState, actions: AppActions) {
  Column(Modifier.fillMaxSize().background(Background).statusBarsPadding()) {
    TitleRow(state, actions)
    TabRow(selectedTabIndex = state.tab, backgroundColor = Background, contentColor = Accent) {
      val titles = D1Input.entries.map { it.label } + "Summary"
      titles.forEachIndexed { index, title ->
        // 13 sp without the button style's letter spacing, and the Tab's content slot (no 16 dp text padding): the text
        // slot at 15 sp cut "Message" and "Summary" to "Messa" and "Summa" on the Galaxy S26 (round 7 smoke).
        Tab(
          selected = state.tab == index,
          onClick = { actions.onTab(index) },
          enabled = !state.recording && state.deciding == null,
          modifier = Modifier.height(48.dp),
        ) {
          Text(
            title,
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.sp,
            maxLines = 1,
            softWrap = false,
            modifier = Modifier.padding(horizontal = 4.dp),
          )
        }
      }
    }
    val scroll = rememberScrollState()
    // A new tab starts at its top. Each tab's input, questions, Decide and answers fit the Galaxy S26's screen (360 x
    // 780 dp) without scrolling, so the answers never push the input out of view; a longer screen (your own questions)
    // scrolls by hand.
    LaunchedEffect(state.tab) { scroll.scrollTo(0) }
    Column(
      Modifier.fillMaxWidth().weight(1f).imePadding().verticalScroll(scroll).padding(horizontal = 16.dp, vertical = 10.dp),
      verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
      if (!state.ready || state.error) {
        Text(state.status, fontSize = 15.sp, color = if (state.error) FailureRed else Muted)
      }
      when (state.tab) {
        0 -> InputTab(D1Input.VOICE, state, actions) { VoiceInput(state, actions) }
        1 -> InputTab(D1Input.PHOTO, state, actions) { PhotoInput(state, actions) }
        2 -> InputTab(D1Input.MESSAGE, state, actions) { MessageInput(state, actions) }
        else -> SummaryTab(state.summary, state.engineLine)
      }
    }
  }
}

@Composable
private fun TitleRow(state: UiState, actions: AppActions) {
  val view = LocalView.current
  val density = LocalDensity.current
  Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
    Text("d1-omni", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Ink)
    Spacer(Modifier.weight(1f))
    val pill = state.pill
    Box(
      Modifier.heightIn(min = 30.dp)
        .background(hex(D1Text.PILL_PALETTE.getValue(pill)), RoundedCornerShape(percent = 50))
        .onGloballyPositioned { actions.onPill(screenBox(it, view), with(density) { PILL_PAD_DP.dp.roundToPx() }) }
        .padding(horizontal = PILL_PAD_DP.dp, vertical = 4.dp),
      contentAlignment = Alignment.Center,
    ) {
      Text(D1Text.PILL.getValue(pill), fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Color.White, maxLines = 1)
    }
  }
}

@Composable
private fun InputTab(input: D1Input, state: UiState, actions: AppActions, content: @Composable () -> Unit) {
  val ui = state.input(input)
  val focus = LocalFocusManager.current
  content()
  QuestionsBlock(input, ui, enabled = state.ready && !state.recording && state.deciding == null, actions.editor)
  Button(
    onClick = {
      focus.clearFocus()
      actions.onDecide(input)
    },
    enabled = state.ready && !state.recording && state.deciding == null,
    modifier = Modifier.fillMaxWidth().height(52.dp),
    colors = ButtonDefaults.buttonColors(backgroundColor = Accent, contentColor = Color.White),
  ) {
    Text("Decide", fontSize = 18.sp, fontWeight = FontWeight.Bold)
  }
  ui.error?.let { Text(it, fontSize = 15.sp, color = FailureRed) }
  ui.result?.let { AnswerBlock(input, it, actions) }
}

@Composable
private fun VoiceInput(state: UiState, actions: AppActions) {
  val enabled = state.ready && state.deciding == null
  Card {
    Row(verticalAlignment = Alignment.CenterVertically) {
      if (state.recording) {
        Button(
          onClick = actions.onStop,
          colors = ButtonDefaults.buttonColors(backgroundColor = Recording, contentColor = Color.White),
          modifier = Modifier.height(48.dp),
        ) { Text("■ Stop", fontSize = 17.sp, fontWeight = FontWeight.Bold) }
      } else {
        Button(
          onClick = actions.onRecord,
          enabled = enabled,
          colors = ButtonDefaults.buttonColors(backgroundColor = Recording, contentColor = Color.White),
          modifier = Modifier.height(48.dp),
        ) { Text("● Record", fontSize = 17.sp, fontWeight = FontWeight.Bold) }
      }
      Spacer(Modifier.width(12.dp))
      val seconds =
        if (state.recording) D1Text.seconds(state.recordedSamples)
        else state.voice?.let { D1Text.seconds(it.samples.size) } ?: "0.0 s"
      Text(seconds, fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Ink)
      Spacer(Modifier.width(12.dp))
      LevelBar(if (state.recording) state.levelDb else -90f, Modifier.weight(1f))
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      SmallButton("Pick WAV", actions.onPickAudio, enabled && !state.recording)
      SmallButton("Play", actions.onPlay, enabled && !state.recording && state.voice != null)
      SmallButton("Load sample", actions.onLoadSample, enabled && !state.recording)
    }
    val clip = state.voice
    Text(
      when {
        state.recording -> "Recording… stops at ${"%.0f".format(java.util.Locale.ROOT, state.maxRecordSeconds)} s"
        clip == null -> "Record a voice note (16 kHz mono), pick a 16 kHz mono WAV, or load the sample."
        clip.source == D1Source.PICKED -> "${sourceWord(clip.source)} · ${clip.name} · ${D1Text.seconds(clip.samples.size)}"
        // a recording's or the sample's file name stays in the run JSON
        else -> "${sourceWord(clip.source)} · ${D1Text.seconds(clip.samples.size)}"
      },
      fontSize = 14.sp,
      color = Muted,
    )
  }
}

@Composable
private fun PhotoInput(state: UiState, actions: AppActions) {
  val enabled = state.ready && state.deciding == null
  Card {
    val recent = state.recentPhotos
    if (recent != null) RecentPhotosSheet(recent, actions) else PhotoPreview(state, actions, enabled)
  }
}

@Composable
private fun PhotoPreview(state: UiState, actions: AppActions, enabled: Boolean) {
  val photo = state.photo
  val thumbnail = photo?.thumbnail
  if (thumbnail != null) {
    // At most 160 dp high: the photo, its question, Decide and the answer fit one screen of the S26.
    Image(
      thumbnail,
      contentDescription = photo.name,
      contentScale = ContentScale.Fit,
      modifier = Modifier.fillMaxWidth().height(160.dp).clip(RoundedCornerShape(8.dp)),
    )
  } else {
    Box(Modifier.fillMaxWidth().height(96.dp).background(Track, RoundedCornerShape(8.dp)), contentAlignment = Alignment.Center) {
      Text("No photo yet", fontSize = 15.sp, color = Muted)
    }
  }
  Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    SmallButton("Pick photo", actions.onPickPhoto, enabled)
    SmallButton("Load sample", actions.onLoadSample, enabled)
  }
  if (photo != null) {
    Text("${sourceWord(photo.source)} · ${photo.name} · ${photo.width} × ${photo.height}", fontSize = 14.sp, color = Muted)
  }
}

/** The newest pictures on the phone (up to four in a row; tap one to use it), Browse… for the system photo picker. */
@Composable
private fun RecentPhotosSheet(photos: List<RecentPhoto>, actions: AppActions) {
  Text("Recent photos", fontSize = 17.sp, fontWeight = FontWeight.Bold, color = Ink)
  if (photos.isEmpty()) {
    Text("No photos on this phone yet.", fontSize = 15.sp, color = Muted)
  } else {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      for (photo in photos) {
        val thumbnail = photo.thumbnail
        Box(
          Modifier.weight(1f)
            .aspectRatio(1f)
            .clip(RoundedCornerShape(6.dp))
            .background(Track)
            .clickable { actions.onRecentPhoto(photo) }
            .semantics { contentDescription = photo.name },
          contentAlignment = Alignment.Center,
        ) {
          if (thumbnail != null) {
            Image(thumbnail, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
          } else {
            Text(photo.name, fontSize = 11.sp, color = Muted, maxLines = 2)
          }
        }
      }
      repeat(MainViewModel.RECENT_PHOTOS - photos.size) { Spacer(Modifier.weight(1f)) }
    }
  }
  Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    SmallButton("Browse…", actions.onBrowsePhotos, true)
    SmallButton("Cancel", actions.onCloseRecent, true)
  }
}

@Composable
private fun MessageInput(state: UiState, actions: AppActions) {
  val enabled = state.ready && state.deciding == null
  Card {
    OutlinedTextField(
      value = state.message,
      onValueChange = actions.onMessage,
      modifier = Modifier.fillMaxWidth(),
      enabled = enabled,
      minLines = 3,
      maxLines = 8,
      textStyle = androidx.compose.ui.text.TextStyle(fontSize = 17.sp, color = Ink),
      label = { Text("Message") },
    )
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      SmallButton("Load sample", actions.onLoadSample, enabled)
    }
  }
}

/**
 * The answers in large type: per question the question as asked, then the answer's word and its probability; under
 * them the ms of the whole input and the graph it ran on. Each word's and probability's size, box, lines and overflow
 * go to [AppActions.onAnswerLayout] once every answer is drawn (the run JSON's `layout.answers`).
 */
@Composable
private fun AnswerBlock(input: D1Input, result: ResultUi, actions: AppActions) {
  val view = LocalView.current
  val density = LocalDensity.current
  val key = result.answers.map { it.qid to it.shown }
  // Each text's lines and overflow (onTextLayout) and its box on the screen (onGloballyPositioned) arrive separately.
  val texts = remember(key) { HashMap<String, Pair<Int, Boolean>>() }
  val boxes = remember(key) { HashMap<String, IntArray>() }
  val store = remember(key) { LinkedHashMap<String, D1AnswerLayout>() }
  fun report(answer: AnswerUi, part: String, text: String, sp: Float) {
    val k = "${answer.qid}/$part"
    val (lines, overflow) = texts[k] ?: return
    val box = boxes[k] ?: return
    store[k] = D1AnswerLayout(answer.qid, part, text, sp, with(density) { sp.sp.toPx() }, box, lines, overflow)
    if (result.answers.all { it.shown != null } && store.size == result.answers.size * 2) {
      actions.onAnswerLayout(input, store.values.toList())
    }
  }
  Column(
    Modifier.fillMaxWidth().background(CardBackground, RoundedCornerShape(12.dp)).padding(12.dp),
    verticalArrangement = Arrangement.spacedBy(8.dp),
  ) {
    for (answer in result.answers) {
      Column {
        Text(answer.instructions, fontSize = 15.sp, color = Muted)
        val shown = answer.shown
        if (shown == null) {
          Text("…", fontSize = ANSWER_SP.sp, fontWeight = FontWeight.Bold, color = Pending)
        } else {
          Row(verticalAlignment = Alignment.Bottom) {
            for ((part, text, sp) in listOf(Triple("answer", shown.answer, ANSWER_SP), Triple("prob", shown.prob, PROB_SP))) {
              if (part == "prob") Spacer(Modifier.width(14.dp))
              val k = "${answer.qid}/$part"
              AnswerText(
                text,
                sp,
                bold = part == "answer",
                modifier = if (part == "answer") Modifier.weight(1f, fill = false) else Modifier,
                onLayout = { l ->
                  texts[k] = l.lineCount to l.hasVisualOverflow
                  report(answer, part, text, sp)
                },
                onPlaced = { c ->
                  boxes[k] = screenBox(c, view)
                  report(answer, part, text, sp)
                },
              )
            }
          }
        }
      }
    }
    Text(result.msLine ?: " ", fontSize = 15.sp, color = Muted)
  }
}

@Composable
private fun AnswerText(
  text: String,
  sp: Float,
  bold: Boolean,
  modifier: Modifier,
  onLayout: (TextLayoutResult) -> Unit,
  onPlaced: (LayoutCoordinates) -> Unit,
) {
  Text(
    text,
    fontSize = sp.sp,
    lineHeight = (sp * 1.15f).sp,
    fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal,
    color = Ink,
    onTextLayout = onLayout,
    modifier = modifier.onGloballyPositioned(onPlaced),
  )
}

@Composable
private fun SummaryTab(summary: SummaryUi?, engineLine: String) {
  if (summary == null) {
    Text("Decide on the voice, photo and message tabs; their answers add up here.", fontSize = 17.sp, color = Muted)
  } else {
    Column(
      Modifier.fillMaxWidth().background(CardBackground, RoundedCornerShape(12.dp)).padding(16.dp),
      verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
      // D1Text.summary's line ("3 inputs · 4 answers · 464 ms · airplane mode on") as three lines, the ms largest.
      Text(D1Text.summaryCounts(summary.inputs, summary.answers), fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Ink)
      Text("${summary.totalMs} ms", fontSize = 40.sp, fontWeight = FontWeight.Bold, color = Ink)
      Text(D1Text.airplane(summary.airplane), fontSize = 18.sp, color = Ink)
      for ((input, shown) in summary.lines) {
        Spacer(Modifier.height(6.dp))
        Text(input.label, fontSize = 14.sp, color = Muted)
        if (shown == null) {
          Text("not decided", fontSize = 22.sp, color = Pending)
        } else {
          for (answer in shown) {
            Row(verticalAlignment = Alignment.Bottom) {
              Text(answer.answer, fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Ink)
              Spacer(Modifier.width(10.dp))
              Text(answer.prob, fontSize = 22.sp, color = Ink)
            }
          }
        }
      }
    }
    Text(summary.deviceLine, fontSize = 14.sp, color = Muted)
  }
  if (engineLine.isNotEmpty()) Text(engineLine, fontSize = 13.sp, color = Muted)
}

/** An outlined button with a one-line label and a tighter padding (three fit one row at 360 dp). */
@Composable
private fun SmallButton(label: String, onClick: () -> Unit, enabled: Boolean) {
  OutlinedButton(onClick = onClick, enabled = enabled, contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp)) {
    Text(label, fontSize = 14.sp, letterSpacing = 0.sp, maxLines = 1, softWrap = false)
  }
}

@Composable
private fun Card(content: @Composable ColumnScope.() -> Unit) {
  Column(
    Modifier.fillMaxWidth().background(CardBackground, RoundedCornerShape(12.dp)).padding(10.dp),
    verticalArrangement = Arrangement.spacedBy(8.dp),
  ) { content() }
}

/** The microphone's level: a bar from -60 dBFS to 0 (red while recording). */
@Composable
private fun LevelBar(db: Float, modifier: Modifier) {
  Canvas(modifier.height(10.dp)) {
    val radius = CornerRadius(size.height / 2, size.height / 2)
    drawRoundRect(Track, cornerRadius = radius)
    val fraction = ((db + 60f) / 60f).coerceIn(0f, 1f)
    if (fraction > 0f) drawRoundRect(Recording, size = Size(maxOf(size.height, size.width * fraction), size.height), cornerRadius = radius)
  }
}

private fun sourceWord(source: D1Source): String =
  when (source) {
    D1Source.RECORDED -> "Recorded"
    D1Source.PICKED -> "Picked"
    D1Source.TYPED -> "Typed"
    D1Source.SAMPLE -> "Sample"
  }

/** Left, top, width and height on the screen, in px. */
private fun screenBox(coordinates: LayoutCoordinates, view: android.view.View): IntArray {
  val location = IntArray(2).also { view.rootView.getLocationOnScreen(it) }
  val bounds = coordinates.boundsInWindow()
  return intArrayOf(bounds.left.toInt() + location[0], bounds.top.toInt() + location[1], bounds.width.toInt(), bounds.height.toInt())
}

internal fun hex(value: String): Color = Color(value.toColorInt())

internal val Background = Color.White
internal val CardBackground = Color(0xFFF1F3F4)
internal val Ink = Color(0xFF202124)
internal val Muted = Color(0xFF5F6368)
internal val Pending = Color(0xFF9AA0A6)
internal val Track = Color(0xFFDADCE0)
internal val Accent = Color(0xFF1565C0)
internal val Recording = Color(0xFFE53935)
internal val FailureRed = Color(0xFFB00020)

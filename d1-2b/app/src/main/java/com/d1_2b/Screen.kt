package com.d1_2b

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.Card
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.material.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.OffsetMapping
import androidx.compose.ui.text.input.TransformedText
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** One color per question card; the same color marks its span in the state and its box on the photo. */
private val MARKS =
  listOf(
    Color(0xFF1E88E5),
    Color(0xFFE53935),
    Color(0xFF43A047),
    Color(0xFFFB8C00),
    Color(0xFF8E24AA),
    Color(0xFF00ACC1),
    Color(0xFFD81B60),
    Color(0xFF6D4C41),
  )

private fun mark(index: Int): Color = MARKS[index % MARKS.size]

@Composable
fun ExtractionScreen(state: UiState, model: MainViewModel, onPickPhoto: () -> Unit) {
  MaterialTheme {
    Surface(modifier = Modifier.fillMaxSize()) {
      LazyColumn(
        modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing).imePadding(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
      ) {
        item { Header(state) }
        item { Examples(state, model) }
        item { StateCard(state, model, onPickPhoto) }
        items(state.questions, key = { it.key }) { draft ->
          val index = state.questions.indexOf(draft)
          QuestionCard(draft, index, state.answered?.answers?.get(draft.key), model)
        }
        item {
          Row(modifier = Modifier.padding(horizontal = 12.dp)) {
            OutlinedButton(onClick = model::addQuestion) { Text("Add question") }
          }
        }
        item { DecideBar(state, model) }
        item { Spacer(Modifier.height(24.dp)) }
      }
    }
  }
}

@Composable
private fun Header(state: UiState) {
  Column(modifier = Modifier.padding(start = 12.dp, end = 12.dp, top = 8.dp)) {
    Text("d1-2B Decide", style = MaterialTheme.typography.h6)
    Text(state.status, style = MaterialTheme.typography.caption, color = Color.Gray)
    state.error?.let { Text(it, style = MaterialTheme.typography.body2, color = Color(0xFFC62828)) }
  }
}

@Composable
private fun Examples(state: UiState, model: MainViewModel) {
  Row(
    modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(horizontal = 12.dp),
    horizontalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    for ((id, title) in state.examples) {
      if (id == state.exampleId) {
        Button(onClick = { model.selectExample(id) }, enabled = !state.busy) { Text(title, fontSize = 13.sp) }
      } else {
        OutlinedButton(onClick = { model.selectExample(id) }, enabled = !state.busy) { Text(title, fontSize = 13.sp) }
      }
    }
  }
}

/** The answers' spans as colored backgrounds over the state text, while it is the text they were read from. */
private class Highlights(private val spans: List<Pair<IntRange, Color>>) : VisualTransformation {
  override fun filter(text: AnnotatedString): TransformedText {
    val styled = buildAnnotatedString {
      append(text.text)
      for ((range, color) in spans) {
        if (range.last < text.length) {
          addStyle(SpanStyle(background = color.copy(alpha = 0.28f), fontWeight = FontWeight.SemiBold), range.first, range.last + 1)
        }
      }
    }
    return TransformedText(styled, OffsetMapping.Identity)
  }
}

private fun spans(state: UiState): List<Pair<IntRange, Color>> {
  val answered = state.answered ?: return emptyList()
  if (answered.stateText != state.stateText) return emptyList()
  val text = answered.stateText
  val out = ArrayList<Pair<IntRange, Color>>()
  state.questions.forEachIndexed { index, draft ->
    val span = answered.answers[draft.key]?.get("span") as? List<*> ?: return@forEachIndexed
    val start = text.offsetByCodePoints(0, (span[0] as Number).toInt())
    val end = text.offsetByCodePoints(0, (span[1] as Number).toInt())
    if (start < end) out.add(start until end to mark(index))
  }
  return out
}

@Composable
private fun StateCard(state: UiState, model: MainViewModel, onPickPhoto: () -> Unit) {
  Card(modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp), elevation = 2.dp) {
    Column(modifier = Modifier.padding(10.dp)) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        Text("State", style = MaterialTheme.typography.subtitle2)
        Spacer(Modifier.weight(1f))
        TextButton(onClick = { model.setStateIsJson(false) }) { Text("Text", fontWeight = if (!state.stateIsJson) FontWeight.Bold else FontWeight.Normal) }
        TextButton(onClick = { model.setStateIsJson(true) }) { Text("JSON", fontWeight = if (state.stateIsJson) FontWeight.Bold else FontWeight.Normal) }
      }
      OutlinedTextField(
        value = state.stateText,
        onValueChange = model::setStateText,
        modifier = Modifier.fillMaxWidth(),
        textStyle = TextStyle(fontSize = 15.sp, lineHeight = 21.sp),
        placeholder = { Text(if (state.photo != null) "Optional: text about the photo" else "Paste a message, a note, or a JSON record") },
        visualTransformation = Highlights(spans(state)),
        maxLines = 12,
      )
      Spacer(Modifier.height(6.dp))
      val photo = state.photo
      if (photo != null) {
        PhotoWithBoxes(state)
        Row {
          TextButton(onClick = onPickPhoto) { Text("Change photo") }
          TextButton(onClick = model::removePhoto) { Text("Remove photo") }
        }
      } else {
        TextButton(onClick = onPickPhoto) { Text("Add a photo") }
      }
    }
  }
}

@Composable
private fun PhotoWithBoxes(state: UiState) {
  val photo = state.photo ?: return
  val boxes = ArrayList<Pair<Map<*, *>, Color>>()
  val answered = state.answered
  if (answered != null) {
    state.questions.forEachIndexed { index, draft ->
      val answer = answered.answers[draft.key] ?: return@forEachIndexed
      (answer["box"] as? Map<*, *>)?.let { boxes.add(it to mark(index)) }
      (answer["boxes"] as? List<*>)?.forEach { b -> ((b as Map<*, *>)["box"] as? Map<*, *>)?.let { boxes.add(it to mark(index)) } }
    }
  }
  Box(modifier = Modifier.fillMaxWidth().aspectRatio(photo.width.toFloat() / photo.height)) {
    Image(bitmap = photo.asImageBitmap(), contentDescription = "The photo the questions are about", modifier = Modifier.fillMaxSize())
    Canvas(modifier = Modifier.fillMaxSize()) {
      for ((box, color) in boxes) {
        val left = (box["left"] as Double).toFloat() * size.width
        val top = (box["top"] as Double).toFloat() * size.height
        val right = (box["right"] as Double).toFloat() * size.width
        val bottom = (box["bottom"] as Double).toFloat() * size.height
        drawRect(color, topLeft = Offset(left, top), size = Size(right - left, bottom - top), style = Stroke(width = 3.dp.toPx()))
      }
    }
  }
}

@Composable
private fun QuestionCard(draft: QuestionDraft, index: Int, answer: Map<String, Any?>?, model: MainViewModel) {
  var editing by remember { mutableStateOf(false) }
  Card(modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp), elevation = 2.dp) {
    Column(modifier = Modifier.padding(start = 10.dp, end = 10.dp, top = 6.dp, bottom = 8.dp)) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(12.dp).background(mark(index), CircleShape))
        Spacer(Modifier.width(6.dp))
        Text(draft.name, style = MaterialTheme.typography.subtitle2)
        Text(" · ${draft.type.label}", style = MaterialTheme.typography.caption, color = Color.Gray)
        Spacer(Modifier.weight(1f))
        TextButton(onClick = { editing = !editing }) { Text(if (editing) "Done" else "Edit", fontSize = 13.sp) }
      }
      OutlinedTextField(
        value = draft.instructions,
        onValueChange = { text -> model.updateQuestion(draft.key) { it.copy(instructions = text) } },
        modifier = Modifier.fillMaxWidth(),
        textStyle = TextStyle(fontSize = 15.sp),
        placeholder = { Text("The question") },
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
      )
      if (editing) QuestionDetails(draft, model)
      if (answer != null) AnswerView(answer, mark(index))
    }
  }
}

@Composable
private fun QuestionDetails(draft: QuestionDraft, model: MainViewModel) {
  fun update(change: (QuestionDraft) -> QuestionDraft) = model.updateQuestion(draft.key, change)
  Column(modifier = Modifier.padding(top = 6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Row(verticalAlignment = Alignment.CenterVertically) {
      SmallField("Name", draft.name, Modifier.weight(1f)) { v -> update { it.copy(name = v) } }
      Spacer(Modifier.width(8.dp))
      TypePicker(draft.type) { t -> update { it.copy(type = t) } }
    }
    when (draft.type) {
      QType.CHOICE,
      QType.MULTIPLE -> {
        draft.options.forEachIndexed { i, option ->
          Row(verticalAlignment = Alignment.CenterVertically) {
            SmallField("Option", option.name, Modifier.weight(0.35f)) { v ->
              update { it.copy(options = it.options.toMutableList().also { list -> list[i] = option.copy(name = v) }) }
            }
            Spacer(Modifier.width(4.dp))
            SmallField("What it means", option.text, Modifier.weight(0.65f)) { v ->
              update { it.copy(options = it.options.toMutableList().also { list -> list[i] = option.copy(text = v) }) }
            }
            TextButton(onClick = { update { it.copy(options = it.options.filterIndexed { j, _ -> j != i }) } }) { Text("×") }
          }
        }
        TextButton(onClick = { update { it.copy(options = it.options + OptionDraft("option${it.options.size + 1}", "")) } }) { Text("Add option") }
      }
      QType.SCORE -> {
        draft.levels.forEachIndexed { i, level ->
          Row(verticalAlignment = Alignment.CenterVertically) {
            SmallField("Level $i", level, Modifier.weight(1f)) { v ->
              update { it.copy(levels = it.levels.toMutableList().also { list -> list[i] = v }) }
            }
            TextButton(onClick = { update { it.copy(levels = it.levels.filterIndexed { j, _ -> j != i }) } }) { Text("×") }
          }
        }
        if (draft.levels.size < 10) TextButton(onClick = { update { it.copy(levels = it.levels + "") } }) { Text("Add level") }
      }
      QType.ESTIMATE ->
        Row {
          SmallField("From", draft.low, Modifier.weight(1f)) { v -> update { it.copy(low = v) } }
          Spacer(Modifier.width(4.dp))
          SmallField("To", draft.high, Modifier.weight(1f)) { v -> update { it.copy(high = v) } }
          Spacer(Modifier.width(4.dp))
          SmallField("Step", draft.step, Modifier.weight(1f)) { v -> update { it.copy(step = v) } }
          Spacer(Modifier.width(4.dp))
          SmallField("Unit", draft.unit, Modifier.weight(1f)) { v -> update { it.copy(unit = v) } }
        }
      QType.NUMBER -> SmallField("Unit (optional)", draft.unit, Modifier.fillMaxWidth()) { v -> update { it.copy(unit = v) } }
      QType.BOXES -> SmallField("At most (optional, 1 to 16)", draft.max, Modifier.fillMaxWidth()) { v -> update { it.copy(max = v) } }
      else -> Unit
    }
    TextButton(onClick = { model.removeQuestion(draft.key) }) { Text("Remove this question", color = Color(0xFFC62828)) }
  }
}

@Composable
private fun SmallField(label: String, value: String, modifier: Modifier, onChange: (String) -> Unit) {
  OutlinedTextField(
    value = value,
    onValueChange = onChange,
    modifier = modifier,
    label = { Text(label, fontSize = 12.sp) },
    singleLine = true,
    textStyle = TextStyle(fontSize = 14.sp),
    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
  )
}

@Composable
private fun TypePicker(type: QType, onPick: (QType) -> Unit) {
  var open by remember { mutableStateOf(false) }
  Box {
    OutlinedButton(onClick = { open = true }) { Text(type.label, fontSize = 13.sp) }
    DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
      for (t in QType.entries) {
        DropdownMenuItem(onClick = {
          open = false
          onPick(t)
        }) { Text(t.label) }
      }
    }
  }
}

private fun p(value: Any?): String = String.format("%.2f", (value as Number).toDouble())

private fun plainNumber(value: Any?): String =
  when (value) {
    null -> "?"
    is Double -> if (value == Math.rint(value) && kotlin.math.abs(value) < 1e15) value.toLong().toString() else PythonFloat.repr(value)
    else -> value.toString()
  }

@Composable
private fun AnswerView(answer: Map<String, Any?>, color: Color) {
  val main: AnnotatedString
  var detail: String? = null
  when (answer["type"]) {
    "noul" -> {
      val yes = (answer["noul"] as Double) >= 0.5
      main = bold(if (yes) "Yes" else "No", "  P(yes) ${p(answer["noul"])}")
    }
    "choice" ->
      if (answer.containsKey("choices")) {
        val chosen = (answer["choices"] as List<*>).joinToString(", ").ifEmpty { "none" }
        main = bold(chosen, "")
        detail = (answer["probabilities"] as Map<*, *>).entries.joinToString(" · ") { "${it.key} ${p(it.value)}" }
      } else {
        main = bold(answer["choice"] as String, "  P ${p(answer["confidence"])}")
        detail =
          (answer["probabilities"] as Map<*, *>).entries.filter { it.key != answer["choice"] }.joinToString(" · ") { "${it.key} ${p(it.value)}" }
      }
    "score" -> {
      val level = (answer["score"] as Double).toInt()
      val legend = (answer["legend"] as Map<*, *>)[level.toString()]
      main = bold("$level · $legend", "  P ${p(answer["confidence"])}")
    }
    "box" -> main = if (answer["box"] == null) bold("Not in the photo", "  found ${p(answer["found"])}") else bold("Found", "  found ${p(answer["found"])} · confidence ${p(answer["confidence"])}")
    "boxes" -> {
      val n = answer["count"] as Int
      main = bold(if (n == 0) "None" else "$n${if (answer["at_least"] == true) " or more" else ""}", "  boxes drawn on the photo")
      detail = (answer["boxes"] as List<*>).joinToString(" · ") { "found ${p((it as Map<*, *>)["found"])}" }.ifEmpty { null }
    }
    "number" ->
      if (answer.containsKey("interval")) {
        val interval = answer["interval"] as List<*>
        val unit = (answer["unit"] as String?)?.let { " $it" } ?: ""
        main = bold("${plainNumber(answer["number"])}$unit", "  80% between ${plainNumber(interval[0])} and ${plainNumber(interval[1])}")
      } else {
        main = readLine(answer, answer["number"]?.let { plainNumber(it) + ((answer["unit"] as String?)?.let { u -> "  (unit $u)" } ?: "") })
      }
    "date" -> main = readLine(answer, answer["date"] as String?)
    else -> main = readLine(answer, answer["text"] as String?)
  }
  Row(modifier = Modifier.padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
    Box(Modifier.size(width = 4.dp, height = 22.dp).background(color, RoundedCornerShape(2.dp)))
    Spacer(Modifier.width(8.dp))
    Column {
      Text(main, fontSize = 17.sp)
      detail?.let { Text(it, fontSize = 13.sp, color = Color.Gray) }
    }
  }
}

private fun readLine(answer: Map<String, Any?>, value: String?): AnnotatedString {
  val found = (answer["found"] as Double)
  if (answer["span"] == null) return bold("Not in the text", "  found ${p(found)}")
  val written = answer["text"] as String?
  val shown = value ?: written ?: "?"
  val source = if (written != null && written != shown) "  from “$written”" else ""
  return bold(shown, "$source  found ${p(found)}")
}

private fun bold(head: String, tail: String): AnnotatedString = buildAnnotatedString {
  withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(head) }
  withStyle(SpanStyle(color = Color.Gray, fontSize = 13.sp)) { append(tail) }
}

@Composable
private fun DecideBar(state: UiState, model: MainViewModel) {
  Column(modifier = Modifier.padding(horizontal = 12.dp)) {
    Button(
      onClick = model::decideFromButton,
      enabled = state.ready && !state.busy,
      modifier = Modifier.fillMaxWidth().height(52.dp),
      colors = ButtonDefaults.buttonColors(),
    ) {
      Text(if (state.busy) "Deciding…" else "Decide", fontSize = 18.sp)
    }
    state.answered?.let { a ->
      val load = if (a.loadMs > 0) " (+ ${(a.loadMs / 1000).format(1)} s to compile the graph first)" else ""
      Text(
        "Answered in ${a.totalMs.format(0)} ms on this phone · ${a.tokens} tokens · L${a.bucket} graph$load",
        style = MaterialTheme.typography.caption,
        modifier = Modifier.padding(top = 4.dp),
      )
    }
  }
}

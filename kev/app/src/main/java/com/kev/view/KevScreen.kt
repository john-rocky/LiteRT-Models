package com.kev.view

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.Card
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Scaffold
import androidx.compose.material.Text
import androidx.compose.material.TextButton
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.kev.AnswerCardUi
import com.kev.CardState
import com.kev.EngineUi
import com.kev.KevAnswerView
import com.kev.KevDecider
import com.kev.KevDrafts
import com.kev.KevForm
import com.kev.KevGraphKey
import com.kev.KevPrecision
import com.kev.KevStatus
import com.kev.LaunchMode
import com.kev.LoadStage
import com.kev.QuestionDraft
import com.kev.QuestionType
import com.kev.R
import com.kev.StateFormat
import com.kev.StateLineUi
import com.kev.UiState

/**
 * The editable sample: backend, one of three bundled requests, the state, typed questions, Decide,
 * then the state line (a request on the shared-state pair), one answer card per question, the
 * footer and the response JSON. Gate and timing launches show their progress instead.
 */
@Composable
fun KevScreen(
  state: UiState,
  onExample: (Int) -> Unit,
  onState: (String) -> Unit,
  onQuestionId: (Long, String) -> Unit,
  onQuestionType: (Long, QuestionType) -> Unit,
  onInstructions: (Long, String) -> Unit,
  onOptions: (Long, String) -> Unit,
  onAddQuestion: () -> Unit,
  onRemoveQuestion: (Long) -> Unit,
  onBackend: (KevDecider.Backend) -> Unit,
  onDecide: () -> Unit,
  onToggleResponse: () -> Unit,
) {
  Scaffold(
    modifier = Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding(),
    topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
  ) { contentPadding ->
    Column(
      Modifier.padding(contentPadding)
        .fillMaxSize()
        .verticalScroll(rememberScrollState())
        .padding(16.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      StatusLine(state)
      if (state.mode != LaunchMode.INTERACTIVE) {
        Text(stringResource(R.string.diagnostics_description))
        state.diagnostics?.let {
          SelectionContainer { Text(it, style = MaterialTheme.typography.body2) }
        }
        state.requestError?.let { Text(it, color = MaterialTheme.colors.error) }
      } else {
        Editor(
          state,
          onExample,
          onState,
          onQuestionId,
          onQuestionType,
          onInstructions,
          onOptions,
          onAddQuestion,
          onRemoveQuestion,
          onBackend,
        )
        Button(onClick = onDecide, enabled = state.canDecide, modifier = Modifier.fillMaxWidth()) {
          Text(stringResource(R.string.decide))
        }
        state.requestError?.let { Text(it, color = MaterialTheme.colors.error) }
        state.stateLine?.let {
          Text(
            stateLineText(it),
            style = MaterialTheme.typography.body2,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
          )
        }
        state.cards.forEach { AnswerCard(it) }
        if (state.footerLines.isNotEmpty()) {
          Column {
            state.footerLines.forEach {
              Text(wrapBetweenItems(it), style = MaterialTheme.typography.caption)
            }
          }
        }
        state.responseJson?.let { json ->
          TextButton(onClick = onToggleResponse) {
            Text(
              stringResource(
                if (state.showResponse) R.string.response_hide else R.string.response_show
              )
            )
          }
          if (state.showResponse) {
            SelectionContainer {
              Text(
                json,
                fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.caption,
              )
            }
          }
        }
      }
    }
  }
}

@Composable
private fun StatusLine(state: UiState) {
  val status = state.status
  val text =
    when (status) {
      is KevStatus.MissingFiles ->
        stringResource(R.string.status_missing, status.files.joinToString(", "))
      is KevStatus.Loading ->
        when {
          status.backend == KevDecider.Backend.NPU && status.npuFirst ->
            stringResource(
              R.string.status_npu_first,
              graphName(status.graph),
              status.elapsedSeconds,
            )
          status.backend == KevDecider.Backend.NPU ->
            stringResource(
              R.string.status_npu_cached,
              graphName(status.graph),
              status.elapsedSeconds,
            )
          status.switching ->
            stringResource(
              R.string.status_switching,
              graphName(status.graph),
              status.elapsedSeconds,
            )
          status.stage == LoadStage.TOKENIZER ->
            stringResource(R.string.status_tokenizer, status.elapsedSeconds)
          status.stage == LoadStage.HEAD ->
            stringResource(R.string.status_head, status.elapsedSeconds)
          else ->
            stringResource(R.string.status_graph, graphName(status.graph), status.elapsedSeconds)
        }
      is KevStatus.Ready -> stringResource(R.string.status_ready)
      is KevStatus.Running ->
        stringResource(R.string.status_running, status.questionIndex + 1, status.total)
      is KevStatus.Done ->
        stringResource(R.string.status_done, status.questions, status.requestTotalMs)
      is KevStatus.Error -> stringResource(R.string.status_error, status.text)
    }
  Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
    Text(
      text,
      style = MaterialTheme.typography.subtitle1,
      color =
        if (status is KevStatus.Error) MaterialTheme.colors.error
        else MaterialTheme.colors.onSurface,
    )
    state.engine?.let { engine ->
      Text(engineLine(engine), style = MaterialTheme.typography.caption)
      engine.npuFailure?.let {
        Text(
          stringResource(R.string.npu_fallback, it),
          style = MaterialTheme.typography.caption,
          color = MaterialTheme.colors.error,
        )
      }
      engine.gpuFailure?.let {
        Text(
          stringResource(R.string.gpu_fallback, it),
          style = MaterialTheme.typography.caption,
          color = MaterialTheme.colors.error,
        )
      }
    }
    if (status is KevStatus.Loading || status is KevStatus.Running) {
      LinearProgressIndicator(Modifier.fillMaxWidth())
    }
  }
}

@Composable
private fun Editor(
  state: UiState,
  onExample: (Int) -> Unit,
  onState: (String) -> Unit,
  onQuestionId: (Long, String) -> Unit,
  onQuestionType: (Long, QuestionType) -> Unit,
  onInstructions: (Long, String) -> Unit,
  onOptions: (Long, String) -> Unit,
  onAddQuestion: () -> Unit,
  onRemoveQuestion: (Long) -> Unit,
  onBackend: (KevDecider.Backend) -> Unit,
) {
  Text(stringResource(R.string.backend_title), style = MaterialTheme.typography.subtitle2)
  // Short names that fit one row at 360 dp; the precision is on the engine line and the footer.
  Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
    KevDecider.Backend.entries.forEach { backend ->
      Choice(
        shortBackendName(backend),
        state.backendChoice == backend,
        state.canDecide && (backend != KevDecider.Backend.NPU || state.npuAvailable),
      ) {
        onBackend(backend)
      }
    }
  }
  if (!state.npuAvailable) {
    Text(
      stringResource(R.string.npu_unavailable),
      style = MaterialTheme.typography.caption,
      maxLines = 1,
      overflow = TextOverflow.Ellipsis,
    )
  }
  Text(stringResource(R.string.examples_title), style = MaterialTheme.typography.subtitle2)
  Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    EXAMPLE_TITLES.forEachIndexed { index, title ->
      if (state.example == index) {
        Button(onClick = { onExample(index) }, enabled = state.editable) {
          Text(stringResource(title))
        }
      } else {
        OutlinedButton(onClick = { onExample(index) }, enabled = state.editable) {
          Text(stringResource(title))
        }
      }
    }
  }
  OutlinedTextField(
    value = state.draft.state,
    onValueChange = onState,
    modifier = Modifier.fillMaxWidth(),
    enabled = state.editable,
    minLines = 4,
    maxLines = 12,
    label = { Text(stringResource(R.string.state_label)) },
  )
  Text(
    stringResource(
      when (KevDrafts.stateFormat(state.draft.state)) {
        StateFormat.JSON -> R.string.state_json
        StateFormat.TEXT -> R.string.state_text
        StateFormat.TEXT_INVALID_JSON -> R.string.state_invalid_json
      }
    ),
    style = MaterialTheme.typography.caption,
  )
  state.draft.questions.forEachIndexed { index, question ->
    QuestionEditor(
      index,
      question,
      state.editable,
      onQuestionId,
      onQuestionType,
      onInstructions,
      onOptions,
      onRemoveQuestion,
    )
  }
  TextButton(onClick = onAddQuestion, enabled = state.editable) {
    Text(stringResource(R.string.add_question))
  }
}

@Composable
private fun QuestionEditor(
  index: Int,
  question: QuestionDraft,
  enabled: Boolean,
  onQuestionId: (Long, String) -> Unit,
  onQuestionType: (Long, QuestionType) -> Unit,
  onInstructions: (Long, String) -> Unit,
  onOptions: (Long, String) -> Unit,
  onRemoveQuestion: (Long) -> Unit,
) {
  Card(Modifier.fillMaxWidth(), elevation = 2.dp) {
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
          stringResource(R.string.question_title, index + 1),
          style = MaterialTheme.typography.subtitle2,
        )
        Spacer(Modifier.weight(1f))
        TextButton(onClick = { onRemoveQuestion(question.key) }, enabled = enabled) {
          Text(stringResource(R.string.remove_question))
        }
      }
      OutlinedTextField(
        value = question.id,
        onValueChange = { onQuestionId(question.key, it) },
        modifier = Modifier.fillMaxWidth(),
        enabled = enabled,
        singleLine = true,
        label = { Text(stringResource(R.string.question_id_label)) },
      )
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        QuestionType.entries.forEach { type ->
          Choice(type.wireName, question.type == type, enabled) {
            onQuestionType(question.key, type)
          }
        }
      }
      OutlinedTextField(
        value = question.instructions,
        onValueChange = { onInstructions(question.key, it) },
        modifier = Modifier.fillMaxWidth(),
        enabled = enabled,
        maxLines = 4,
        label = { Text(stringResource(R.string.instructions_label)) },
      )
      OutlinedTextField(
        value = question.options,
        onValueChange = { onOptions(question.key, it) },
        modifier = Modifier.fillMaxWidth(),
        enabled = enabled,
        minLines = 2,
        maxLines = 12,
        label = { Text(stringResource(R.string.options_label)) },
      )
      Text(
        stringResource(
          when (question.type) {
            QuestionType.CHOICE -> R.string.options_help_choice
            QuestionType.NOUL -> R.string.options_help_noul
            QuestionType.SCORE -> R.string.options_help_score
          }
        ),
        style = MaterialTheme.typography.caption,
      )
    }
  }
}

/** One question's answer: pending, running, then the answer with every option's probability. */
@Composable
private fun AnswerCard(card: AnswerCardUi) {
  Card(Modifier.fillMaxWidth(), elevation = 2.dp) {
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(14.dp).background(stateColor(card.state), CircleShape))
        Spacer(Modifier.width(8.dp))
        Text(
          stringResource(R.string.card_header, card.qid, card.type.wireName),
          style = MaterialTheme.typography.subtitle2,
        )
        Spacer(Modifier.weight(1f))
        cardTime(card)?.let { Text(it, style = MaterialTheme.typography.caption) }
      }
      Text(card.question, style = MaterialTheme.typography.body2)
      when (card.state) {
        CardState.PENDING ->
          Text(stringResource(R.string.card_pending), style = MaterialTheme.typography.caption)
        CardState.RUNNING -> LinearProgressIndicator(Modifier.fillMaxWidth())
        CardState.FAILED -> Text(card.error.orEmpty(), color = MaterialTheme.colors.error)
        CardState.DONE -> card.view?.let { AnswerContent(it) }
      }
    }
  }
}

@Composable
private fun AnswerContent(view: KevAnswerView) {
  Row(verticalAlignment = Alignment.CenterVertically) {
    Text(
      stringResource(
        when (view.type) {
          QuestionType.CHOICE -> R.string.answer_choice
          QuestionType.NOUL -> R.string.answer_noul
          QuestionType.SCORE -> R.string.answer_score
        },
        view.answer,
      ),
      fontWeight = FontWeight.Bold,
      color = MaterialTheme.colors.primary,
    )
    Spacer(Modifier.weight(1f))
    view.confidence?.let {
      Text(stringResource(R.string.answer_confidence, it), style = MaterialTheme.typography.caption)
    }
  }
  view.bars.forEach { bar ->
    Column {
      Row {
        Text(bar.label, style = MaterialTheme.typography.body2, modifier = Modifier.weight(1f))
        Text(bar.value, style = MaterialTheme.typography.body2, fontFamily = FontFamily.Monospace)
      }
      LinearProgressIndicator(progress = bar.fraction, modifier = Modifier.fillMaxWidth())
    }
  }
}

/** A radio choice whose label selects it too (the whole row is one radio button). */
@Composable
private fun Choice(title: String, selected: Boolean, enabled: Boolean, onSelect: () -> Unit) {
  Row(
    Modifier.heightIn(min = 48.dp)
      .selectable(
        selected = selected,
        enabled = enabled,
        role = Role.RadioButton,
        onClick = onSelect,
      )
      .padding(end = 8.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    RadioButton(selected = selected, enabled = enabled, onClick = null)
    Spacer(Modifier.width(8.dp))
    Text(title)
  }
}

/**
 * "68 ms · L128 · NPU" (a row graph) or "62 ms · Q64 · GPU" (a branch on the pair): the card's
 * time, the window it ran in and where its graph ran, once the card is done.
 */
@Composable
internal fun cardTime(card: AnswerCardUi): String? {
  val ms = card.msText ?: return null
  val window = card.window ?: return ms
  val name = if (card.form == KevForm.PAIR) R.string.question_window_name else R.string.window_name
  val backend =
    card.backend ?: return stringResource(R.string.card_ms_window, ms, stringResource(name, window))
  return stringResource(
    R.string.card_ms_window_backend,
    ms,
    stringResource(name, window),
    shortBackendName(backend),
  )
}

/** "State · 72 tokens · 266 ms": the pair's state call, without the ms while it runs. */
@Composable
internal fun stateLineText(line: StateLineUi): String =
  line.ms?.let { stringResource(R.string.state_line, line.tokens, it) }
    ?: stringResource(R.string.state_line_pending, line.tokens)

/** "L256" or "S128+Q64". */
@Composable
private fun graphName(graph: KevGraphKey?): String =
  when (graph) {
    is KevGraphKey.Window -> stringResource(R.string.window_name, graph.window)
    is KevGraphKey.Pair ->
      stringResource(R.string.pair_name, graph.shape.stateLength, graph.shape.questionLength)
    null -> ""
  }

/**
 * "NPU · L128 + L256 · loaded in 4.3 s (graph compile 3.4 s)" ([KevEngineTimes]); while no graph is
 * compiled (a switch to another backend compiling) only "NPU · no graph compiled".
 */
@Composable
private fun engineLine(engine: EngineUi): String {
  val times =
    engine.times ?: return stringResource(R.string.engine_line_no_graph, engineBackends(engine))
  return stringResource(
    R.string.engine_line,
    engineBackends(engine),
    residentText(engine),
    times.loadMs / MILLIS_PER_SECOND,
    times.compileMs / MILLIS_PER_SECOND,
  )
}

/**
 * "L128 + L256" or "S128+Q64": the resident graphs of [engine] (windows ascending, then the pair);
 * graphs on different backends each with its own ("L128 NPU + L512 GPU"), GPU graphs at different
 * precisions each with its own ("L128 FP16 (FP32 accum) + L256 FP32").
 */
@Composable
private fun residentText(engine: EngineUi): String {
  val eachBackend = engine.backends.distinct().size > 1
  val gpu = gpuPrecisions(engine)
  val eachPrecision = gpu.isNotEmpty() && KevPrecision.common(gpu, null) == null
  return engine.resident
    .mapIndexed { index, graph ->
      val backend = engine.backends.getOrNull(index)
      val precision = engine.precisions.getOrNull(index)
      listOfNotNull(
          graphName(graph),
          backend?.let { shortBackendName(it) }?.takeIf { eachBackend },
          precision
            ?.takeIf { eachPrecision && backend == KevDecider.Backend.GPU }
            ?.let { precisionName(it) },
        )
        .joinToString(" ")
    }
    .joinToString(WINDOW_SEPARATOR)
}

/**
 * Where the resident graphs of [engine] run: one backend title, or the titles of different ones
 * joined by " + "; with nothing resident, the chosen backend (and the launch's precision).
 */
@Composable
private fun engineBackends(engine: EngineUi): String {
  if (engine.backends.isEmpty()) return backendTitle(engine.requested, engine.forcedPrecision)
  val gpu = KevPrecision.common(gpuPrecisions(engine), engine.forcedPrecision)
  return engine.backends
    .distinct()
    .map { backendTitle(it, if (it == KevDecider.Backend.GPU) gpu else null) }
    .joinToString(WINDOW_SEPARATOR)
}

/** The GPU precisions of the resident graphs of [engine] that run on the GPU. */
private fun gpuPrecisions(engine: EngineUi): List<KevPrecision> =
  engine.precisions.filterIndexed { index, _ ->
    engine.backends.getOrNull(index) == KevDecider.Backend.GPU
  }

/**
 * "GPU FP32", "GPU FP16 (FP32 accum)", "GPU" (no single [precision]: graphs at different ones, or
 * each graph at its own default before any compiles), "NPU" or "CPU 4 threads".
 */
@Composable
private fun backendTitle(backend: KevDecider.Backend, precision: KevPrecision?): String =
  stringResource(
    when {
      backend == KevDecider.Backend.CPU -> R.string.backend_cpu
      backend == KevDecider.Backend.NPU -> R.string.backend_npu
      precision == KevPrecision.FP16_FP32_ACCUM -> R.string.backend_gpu_fp16acc
      precision == KevPrecision.FP32 -> R.string.backend_gpu
      else -> R.string.backend_gpu_any
    }
  )

/** "GPU", "NPU" or "CPU": the choices of "Run on" and the cards' last item. */
@Composable
internal fun shortBackendName(backend: KevDecider.Backend): String =
  stringResource(
    when (backend) {
      KevDecider.Backend.GPU -> R.string.run_on_gpu
      KevDecider.Backend.NPU -> R.string.run_on_npu
      KevDecider.Backend.CPU -> R.string.run_on_cpu
    }
  )

/** "FP32" or "FP16 (FP32 accum)". */
@Composable
private fun precisionName(precision: KevPrecision): String =
  stringResource(
    when (precision) {
      KevPrecision.FP32 -> R.string.precision_fp32
      KevPrecision.FP16_FP32_ACCUM -> R.string.precision_fp16acc
    }
  )

/**
 * [line] with the spaces inside each " · "-separated item made non-breaking, so that a narrow
 * screen wraps the line only between items and never inside one such as "GPU FP32".
 */
private fun wrapBetweenItems(line: String): String =
  line.split(ITEM_SEPARATOR).joinToString(ITEM_SEPARATOR) { it.replace(' ', NO_BREAK_SPACE) }

private val EXAMPLE_TITLES =
  listOf(R.string.example_ticket, R.string.example_incident, R.string.example_review)
private const val MILLIS_PER_SECOND = 1000f
private const val ITEM_SEPARATOR = " · "
private const val WINDOW_SEPARATOR = " + "
private const val NO_BREAK_SPACE = '\u00A0'

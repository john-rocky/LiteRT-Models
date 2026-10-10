package com.d1omni.view

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.RadioButton
import androidx.compose.material.Text
import androidx.compose.material.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.d1omni.D1DraftException
import com.d1omni.D1Drafts
import com.d1omni.D1Input
import com.d1omni.InputUi
import com.d1omni.QuestionDraft
import com.d1omni.QuestionType

/** What the questions editor can ask for, per input screen. */
class EditorActions(
  val onToggle: (D1Input) -> Unit,
  val onId: (D1Input, Long, String) -> Unit,
  val onType: (D1Input, Long, QuestionType) -> Unit,
  val onInstructions: (D1Input, Long, String) -> Unit,
  val onOptions: (D1Input, Long, String) -> Unit,
  val onAdd: (D1Input) -> Unit,
  val onRemove: (D1Input, Long) -> Unit,
  val onReset: (D1Input) -> Unit,
)

/**
 * An input's questions: folded, each question as asked and its options in one line, with Edit; open, the Kev Decide
 * sample's editor (name, type, the question, the options one per line) with Add, Reset and Done.
 */
@Composable
fun QuestionsBlock(input: D1Input, ui: InputUi, enabled: Boolean, actions: EditorActions) {
  Column(
    Modifier.fillMaxWidth().background(CardBackground, RoundedCornerShape(12.dp)).padding(horizontal = 12.dp, vertical = 8.dp),
    verticalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    Row(verticalAlignment = Alignment.CenterVertically) {
      Text(if (ui.drafts.size == 1) "Question" else "Questions (${ui.drafts.size})", fontSize = 14.sp, fontWeight = FontWeight.Bold,
        color = Muted)
      Spacer(Modifier.weight(1f))
      TextButton(onClick = { actions.onToggle(input) }, enabled = enabled) { Text(if (ui.editing) "Done" else "Edit") }
    }
    if (!ui.editing) {
      for (draft in ui.drafts) {
        Column {
          Text(draft.instructions.ifBlank { "(no question yet)" }, fontSize = 17.sp, color = Ink)
          Text(summaryOf(draft), fontSize = 13.sp, color = Muted)
        }
      }
    } else {
      ui.drafts.forEachIndexed { index, draft -> QuestionEditor(input, index, draft, enabled, actions) }
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        TextButton(onClick = { actions.onAdd(input) }, enabled = enabled) { Text("Add a question") }
        TextButton(onClick = { actions.onReset(input) }, enabled = enabled) { Text("Reset") }
      }
    }
  }
}

@Composable
private fun QuestionEditor(input: D1Input, index: Int, draft: QuestionDraft, enabled: Boolean, actions: EditorActions) {
  Column(
    Modifier.fillMaxWidth().background(Background, RoundedCornerShape(8.dp)).padding(10.dp),
    verticalArrangement = Arrangement.spacedBy(6.dp),
  ) {
    Row(verticalAlignment = Alignment.CenterVertically) {
      Text("Question ${index + 1}", fontSize = 14.sp, fontWeight = FontWeight.Bold, color = Ink)
      Spacer(Modifier.weight(1f))
      TextButton(onClick = { actions.onRemove(input, draft.key) }, enabled = enabled) { Text("Remove") }
    }
    OutlinedTextField(
      value = draft.id,
      onValueChange = { actions.onId(input, draft.key, it) },
      modifier = Modifier.fillMaxWidth(),
      enabled = enabled,
      singleLine = true,
      label = { Text("Name") },
    )
    Row {
      QuestionType.entries.forEach { type ->
        Row(
          Modifier.heightIn(min = 48.dp)
            .selectable(selected = draft.type == type, enabled = enabled, role = Role.RadioButton) {
              actions.onType(input, draft.key, type)
            }
            .padding(end = 8.dp),
          verticalAlignment = Alignment.CenterVertically,
        ) {
          RadioButton(selected = draft.type == type, enabled = enabled, onClick = null)
          Spacer(Modifier.width(4.dp))
          Text(type.wireName)
        }
      }
    }
    OutlinedTextField(
      value = draft.instructions,
      onValueChange = { actions.onInstructions(input, draft.key, it) },
      modifier = Modifier.fillMaxWidth(),
      enabled = enabled,
      maxLines = 4,
      label = { Text("Question") },
    )
    OutlinedTextField(
      value = draft.options,
      onValueChange = { actions.onOptions(input, draft.key, it) },
      modifier = Modifier.fillMaxWidth(),
      enabled = enabled,
      minLines = 2,
      maxLines = 10,
      label = { Text("Options, one per line") },
    )
    Text(
      when (draft.type) {
        QuestionType.CHOICE -> "One option per line: name: description, or just name (at least two)."
        QuestionType.NOUL -> "Optional: true: what yes means, and false: what no means."
        QuestionType.SCORE -> "One level per line, from the lowest (2 to 10)."
      },
      fontSize = 13.sp,
      color = Muted,
    )
  }
}

/** The folded line of a question: its type and options, or what is wrong with it. */
private fun summaryOf(draft: QuestionDraft): String =
  try {
    D1Drafts.summary(D1Drafts.toQuestions(listOf(draft.copy(id = draft.id.ifBlank { "q" }))).values.single())
  } catch (failure: D1DraftException) {
    "${draft.type.wireName} · ${D1Drafts.message(failure).substringAfter(": ")}"
  }

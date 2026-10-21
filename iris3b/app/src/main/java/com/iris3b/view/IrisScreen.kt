package com.iris3b.view

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.Button
import androidx.compose.material.LinearProgressIndicator
import androidx.compose.material.MaterialTheme
import androidx.compose.material.OutlinedButton
import androidx.compose.material.OutlinedTextField
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.iris3b.Example
import com.iris3b.ImageSession
import com.iris3b.ModelState
import com.iris3b.R
import com.iris3b.UiState

/** The one screen: model status, prompt editor, examples, seed, Generate / Stop, progress and the image. */
@Composable
fun IrisScreen(
    state: UiState,
    onPromptChange: (String) -> Unit,
    onExample: (Example) -> Unit,
    onSeedChange: (String) -> Unit,
    onRandomSeed: () -> Unit,
    onGenerate: () -> Unit,
    onStop: () -> Unit,
    onSave: () -> Unit,
    onCheckAgain: () -> Unit,
) {
    // An image takes minutes; keep the screen on meanwhile so the activity is not stopped halfway.
    val view = LocalView.current
    DisposableEffect(state.running) {
        view.keepScreenOn = state.running
        onDispose { view.keepScreenOn = false }
    }
    // A new image lands below the controls: scroll down to it once it has been laid out.
    val scroll = rememberScrollState()
    LaunchedEffect(state.image) {
        if (state.image != null) {
            withFrameNanos {}
            scroll.animateScrollTo(scroll.maxValue)
        }
    }
    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colors.background) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .verticalScroll(scroll)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(stringResource(R.string.title), style = MaterialTheme.typography.h5, color = MaterialTheme.colors.primary)
            Text(stringResource(R.string.subtitle), style = MaterialTheme.typography.body2)
            ModelStatus(state.model, onCheckAgain)
            PromptEditor(state, onPromptChange)
            ExampleRow(enabled = !state.running, onExample = onExample)
            SeedRow(state, onSeedChange, onRandomSeed)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onGenerate, enabled = state.canGenerate) { Text(stringResource(R.string.generate)) }
                if (state.running) {
                    OutlinedButton(onClick = onStop) { Text(stringResource(R.string.stop)) }
                }
            }
            if (state.running) Progress(state)
            state.message?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.body2,
                    color = if (state.messageIsError) MaterialTheme.colors.error else MaterialTheme.colors.onSurface,
                )
            }
            state.image?.let { Result(it, state.imageInfo, saveEnabled = !state.running, onSave = onSave) }
        }
    }
}

@Composable
private fun ModelStatus(model: ModelState, onCheckAgain: () -> Unit) {
    when (model) {
        ModelState.Ready -> Unit
        ModelState.Checking -> Text(stringResource(R.string.model_checking), style = MaterialTheme.typography.body2)
        is ModelState.Failed -> Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(stringResource(R.string.model_failed, model.error), color = MaterialTheme.colors.error)
            OutlinedButton(onClick = onCheckAgain) { Text(stringResource(R.string.check_again)) }
        }
        is ModelState.Missing -> Surface(
            shape = RoundedCornerShape(8.dp),
            border = BorderStroke(1.dp, MaterialTheme.colors.error),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(stringResource(R.string.model_missing_title), style = MaterialTheme.typography.subtitle1, color = MaterialTheme.colors.error)
                val body = if (model.files == listOf(ImageSession.CONFIG_FILE)) {
                    stringResource(R.string.model_missing_all)
                } else {
                    stringResource(R.string.model_missing_some, model.files.size, model.files.first())
                }
                Text(body, style = MaterialTheme.typography.body2)
                SelectionContainer {
                    Text(stringResource(R.string.install_command), style = MaterialTheme.typography.body2, fontFamily = FontFamily.Monospace)
                }
                Text(stringResource(R.string.model_missing_after), style = MaterialTheme.typography.caption)
                OutlinedButton(onClick = onCheckAgain) { Text(stringResource(R.string.check_again)) }
            }
        }
    }
}

@Composable
private fun PromptEditor(state: UiState, onPromptChange: (String) -> Unit) {
    Column {
        OutlinedTextField(
            value = state.prompt,
            onValueChange = onPromptChange,
            label = { Text(stringResource(R.string.prompt_label)) },
            enabled = !state.running,
            maxLines = 6,
            modifier = Modifier.fillMaxWidth(),
        )
        val tokens = state.promptTokens
        val over = tokens != null && tokens > state.maxPromptTokens
        val hint = if (state.prompt.isBlank()) {
            stringResource(R.string.prompt_empty)
        } else if (tokens == null) {
            null
        } else if (tokens > state.maxPromptTokens) {
            stringResource(R.string.prompt_cut, tokens, state.maxPromptTokens)
        } else {
            stringResource(R.string.prompt_tokens, tokens, state.maxPromptTokens)
        }
        if (hint != null) {
            Text(
                hint,
                style = MaterialTheme.typography.caption,
                color = if (over) MaterialTheme.colors.error else MaterialTheme.colors.onSurface.copy(alpha = 0.7f),
                modifier = Modifier.padding(start = 4.dp, top = 4.dp),
            )
        }
    }
}

@Composable
private fun ExampleRow(enabled: Boolean, onExample: (Example) -> Unit) {
    Row(
        modifier = Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(stringResource(R.string.examples_label), style = MaterialTheme.typography.caption)
        for (example in Example.entries) {
            ExampleChip(stringResource(example.label), enabled) { onExample(example) }
        }
    }
}

@Composable
private fun ExampleChip(text: String, enabled: Boolean, onClick: () -> Unit) {
    val shape = RoundedCornerShape(50)
    Surface(
        shape = shape,
        border = BorderStroke(1.dp, MaterialTheme.colors.primary.copy(alpha = if (enabled) 0.6f else 0.2f)),
        modifier = Modifier.clip(shape).clickable(enabled = enabled, role = Role.Button, onClick = onClick),
    ) {
        Text(text, style = MaterialTheme.typography.body2, modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
    }
}

@Composable
private fun SeedRow(state: UiState, onSeedChange: (String) -> Unit, onRandomSeed: () -> Unit) {
    val randomLabel = stringResource(R.string.random_seed)
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(
            value = state.seedText,
            onValueChange = onSeedChange,
            label = { Text(stringResource(R.string.seed_label)) },
            singleLine = true,
            enabled = !state.running,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.weight(1f),
        )
        OutlinedButton(
            onClick = onRandomSeed,
            enabled = !state.running,
            modifier = Modifier.semantics { contentDescription = randomLabel },
        ) {
            Text(DICE)
        }
    }
}

@Composable
private fun Progress(state: UiState) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(state.stage, style = MaterialTheme.typography.body2)
        LinearProgressIndicator(progress = state.progress, modifier = Modifier.fillMaxWidth())
        Text(stringResource(R.string.elapsed, state.elapsedSeconds), style = MaterialTheme.typography.caption)
    }
}

@Composable
private fun Result(image: Bitmap, info: String?, saveEnabled: Boolean, onSave: () -> Unit) {
    val bitmap = remember(image) { image.asImageBitmap() }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Image(
            bitmap = bitmap,
            contentDescription = stringResource(R.string.image_description),
            modifier = Modifier.fillMaxWidth().aspectRatio(1f),
        )
        info?.let { Text(it, style = MaterialTheme.typography.caption) }
        OutlinedButton(onClick = onSave, enabled = saveEnabled) { Text(stringResource(R.string.save)) }
    }
}

/** U+1F3B2 game die. */
private const val DICE = "🎲"

@Preview(widthDp = 360, heightDp = 760)
@Composable
private fun IrisScreenPreview() {
    IrisTheme {
        IrisScreen(
            state = UiState(model = ModelState.Missing(listOf(ImageSession.CONFIG_FILE)), prompt = Example.APPLE.text, promptTokens = 10),
            onPromptChange = {},
            onExample = {},
            onSeedChange = {},
            onRandomSeed = {},
            onGenerate = {},
            onStop = {},
            onSave = {},
            onCheckAgain = {},
        )
    }
}

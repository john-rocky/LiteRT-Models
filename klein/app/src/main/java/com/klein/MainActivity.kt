package com.klein

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * Runs FLUX.2-klein-4B on the GPU: generate an image, or edit one you pick.
 *
 * Both paths share the same twelve-graph sequential-residency loop. Picking an
 * image switches to the `kce_*` graphs, whose only difference is that the DiT's
 * image sequence carries the reference's latent tokens alongside the noise ones.
 * The prompt is baked into the staged `.bin` tensors, so this demo edits with the
 * prompt that `scripts/gen_prep_klein.py --edit` was run with.
 *
 * For scripted device verification the run can be started without a tap:
 *     adb shell am start -n com.klein/.MainActivity --es mode edit
 * which edits the staged `klein_bins_edit/edit_source.png`, so the device output
 * is directly comparable with the host loop's. `--es mode generate` starts the
 * text-to-image path, and `--es prompt "..."` overrides the prompt so the typed
 * path can be scored against an fp32 reference.
 */
class MainActivity : ComponentActivity() {

    private var status by mutableStateOf("Pick an image to edit, or generate one.")
    private var result by mutableStateOf<Bitmap?>(null)
    private var source by mutableStateOf<Bitmap?>(null)
    private var running by mutableStateOf(false)
    private var prompt by mutableStateOf(DEFAULT_PROMPT)
    private var editPrompt by mutableStateOf(DEFAULT_EDIT_PROMPT)
    private var promptEditable by mutableStateOf(false)
    private var job: Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // A run takes minutes; letting the screen sleep tears the activity down.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        promptEditable = KleinGen.isPromptEditable(this)
        intent?.getStringExtra("prompt")?.let {
            prompt = it
            editPrompt = it
        }
        when (intent?.getStringExtra("mode")) {
            "edit" -> start(stagedSource())
            "generate" -> start(null)
        }
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    KleinScreen()
                }
            }
        }
    }

    /**
     * Runs one generation or edit on the activity's lifecycle scope.
     *
     * Not on a `LaunchedEffect`: a five-minute loop outlives the composition, and
     * a recomposition would cancel it mid-run. A single in-flight job is enforced
     * because two concurrent loops each hold a ~900 MB graph and the low-memory
     * killer takes the process down.
     */
    private fun start(reference: Bitmap?) {
        if (running) {
            return
        }
        running = true
        source = reference
        result = null
        val typed = if (!promptEditable) null else if (reference == null) prompt else editPrompt
        job = lifecycleScope.launch {
            val logFile = File(getExternalFilesDir(null), "gen_log.txt")
            logFile.writeText("")
            try {
                val produced = withContext(Dispatchers.Default) {
                    KleinGen.run(this@MainActivity, reference, typed) { line ->
                        status = line
                        logFile.appendText(line + "\n")
                    }
                }
                val name = if (reference == null) "generated.png" else "edited.png"
                val out = File(getExternalFilesDir(null), name)
                FileOutputStream(out).use { produced.compress(Bitmap.CompressFormat.PNG, 100, it) }
                result = produced
                status = "Done -> ${out.absolutePath}"
                logFile.appendText("DONE\n")
            } catch (e: Throwable) {
                status = "FAILED: $e"
                logFile.appendText("FAILED: $e\n")
            }
            running = false
        }
    }

    @Composable
    private fun KleinScreen() {
        val picker = rememberLauncherForActivityResult(
            ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
            if (uri != null) {
                start(decode(uri))
            }
        }

        // targetSdk 35 draws edge to edge: without safeDrawingPadding the title and
        // the buttons render underneath the status bar and the action bar.
        Column(modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
            Text("FLUX.2-klein-4B · chunked DiT on LiteRT GPU",
                style = MaterialTheme.typography.titleMedium)
            if (promptEditable) {
                OutlinedTextField(
                    value = prompt,
                    onValueChange = { prompt = it },
                    enabled = !running,
                    label = { Text("Prompt (generate)") },
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                OutlinedTextField(
                    value = editPrompt,
                    onValueChange = { editPrompt = it },
                    enabled = !running,
                    label = { Text("Prompt (edit)") },
                    modifier = Modifier.fillMaxWidth().padding(top = 4.dp))
            } else {
                Text("Prompt is baked into the staged tensors. Stage klein_tokenizer/ " +
                    "to type your own.", modifier = Modifier.padding(top = 8.dp))
            }
            Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(enabled = !running, onClick = {
                    picker.launch(PickVisualMediaRequest(
                        ActivityResultContracts.PickVisualMedia.ImageOnly))
                }) { Text("Edit an image") }
                Button(enabled = !running, onClick = { start(null) }) { Text("Generate") }
            }
            Text(status, modifier = Modifier.padding(top = 8.dp))
            source?.let {
                Image(it.asImageBitmap(), contentDescription = "source",
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp))
            }
            result?.let {
                Image(it.asImageBitmap(), contentDescription = "result",
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp))
            }
        }
    }

    /** Loads the picked image; the model squares and resizes it itself. */
    private fun decode(uri: Uri): Bitmap? =
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }

    private companion object {
        const val DEFAULT_PROMPT = "a red apple on a wooden table, studio lighting"
        const val DEFAULT_EDIT_PROMPT = "turn the apple into a green apple"
    }

    /** The reference `gen_prep_klein.py --edit` staged, for scripted runs. */
    private fun stagedSource(): Bitmap? {
        val file = File(getExternalFilesDir(null), "klein_bins_edit/edit_source.png")
        return if (file.exists()) BitmapFactory.decodeFile(file.absolutePath) else null
    }
}

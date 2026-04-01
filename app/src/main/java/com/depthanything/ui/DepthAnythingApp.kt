package com.depthanything.ui

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.depthanything.ml.InferenceMode
import com.depthanything.viewmodel.BenchmarkEntry
import com.depthanything.viewmodel.MainViewModel
import com.depthanything.viewmodel.UiState

@Composable
fun DepthAnythingApp(viewModel: MainViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    MaterialTheme(colorScheme = darkColorScheme()) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("DepthAnything V2") },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    )
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ModeSelector(uiState, viewModel)
                ImagePickerSection(viewModel)
                ResultSection(uiState)
                BenchmarkSection(uiState, viewModel)
            }

            if (uiState.isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        if (uiState.loadingMessage.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                uiState.loadingMessage,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModeSelector(uiState: UiState, viewModel: MainViewModel) {
    var expanded by remember { mutableStateOf(false) }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Inference Mode", style = MaterialTheme.typography.labelMedium)
            Spacer(modifier = Modifier.height(4.dp))

            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = it }
            ) {
                OutlinedTextField(
                    value = uiState.currentMode.label,
                    onValueChange = {},
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    InferenceMode.entries.forEach { mode ->
                        val available = mode in uiState.availableModes
                        DropdownMenuItem(
                            text = {
                                Column {
                                    Text(
                                        mode.label,
                                        fontWeight = if (available) FontWeight.Normal else FontWeight.Light,
                                        color = if (available)
                                            MaterialTheme.colorScheme.onSurface
                                        else
                                            MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
                                    )
                                    Text(
                                        if (available) mode.description else "${mode.description} (model not found)",
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(
                                            alpha = if (available) 0.7f else 0.3f
                                        )
                                    )
                                }
                            },
                            onClick = {
                                if (available) {
                                    viewModel.switchMode(mode)
                                    expanded = false
                                }
                            },
                            enabled = available
                        )
                    }
                }
            }
        }
    }

    uiState.error?.let { error ->
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer
            )
        ) {
            Text(
                text = error,
                modifier = Modifier.padding(16.dp),
                color = MaterialTheme.colorScheme.onErrorContainer,
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
private fun ImagePickerSection(viewModel: MainViewModel) {
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.loadImage(it) }
    }

    Button(
        onClick = { launcher.launch("image/*") },
        modifier = Modifier.fillMaxWidth()
    ) {
        Text("Select Image")
    }
}

@Composable
private fun ResultSection(uiState: UiState) {
    uiState.inputBitmap?.let { input ->
        ImageCard("Input", input)
    }

    uiState.coloredDepthBitmap?.let { depth ->
        ImageCard("Depth Map - ${uiState.currentMode.label}", depth)
        Card(modifier = Modifier.fillMaxWidth()) {
            Row(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Inference Time", style = MaterialTheme.typography.bodyMedium)
                Text(
                    "${uiState.inferenceTimeMs} ms",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
            }
        }
    }
}

@Composable
private fun ImageCard(title: String, bitmap: Bitmap) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(8.dp)) {
            Text(
                title,
                style = MaterialTheme.typography.labelSmall,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = title,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(bitmap.width.toFloat() / bitmap.height),
                contentScale = ContentScale.Fit
            )
        }
    }
}

@Composable
private fun BenchmarkSection(uiState: UiState, viewModel: MainViewModel) {
    if (uiState.inputBitmap == null) return

    OutlinedButton(
        onClick = { viewModel.runBenchmark() },
        modifier = Modifier.fillMaxWidth(),
        enabled = !uiState.isLoading
    ) {
        Text("Run Benchmark (all modes)")
    }

    if (uiState.benchmarkEntries.isNotEmpty()) {
        // Timing table
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    "Benchmark Results",
                    style = MaterialTheme.typography.titleSmall,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                // Header
                Row(modifier = Modifier.fillMaxWidth()) {
                    Text("Mode", style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.weight(1.4f))
                    Text("Init", style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.weight(1f), fontWeight = FontWeight.Bold)
                    Text("1st", style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.weight(1f), fontWeight = FontWeight.Bold)
                    Text("Avg", style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.weight(1f), fontWeight = FontWeight.Bold)
                }

                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))

                uiState.benchmarkEntries.forEach { entry ->
                    BenchmarkRow(entry)
                }
            }
        }

        // Depth map comparison
        Text(
            "Depth Map Comparison",
            style = MaterialTheme.typography.titleSmall,
            modifier = Modifier.padding(top = 8.dp)
        )

        uiState.benchmarkEntries.forEach { entry ->
            if (entry.depthBitmap != null) {
                ImageCard("${entry.mode.label} (${entry.avgMs} ms)", entry.depthBitmap)
            } else if (entry.error != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(
                        "${entry.mode.label}: ${entry.error}",
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }
        }
    }
}

@Composable
private fun BenchmarkRow(entry: BenchmarkEntry) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            entry.mode.label,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.weight(1.4f),
            fontSize = 11.sp
        )
        TimingCell(entry.initTimeMs, Modifier.weight(1f))
        TimingCell(entry.firstRunMs, Modifier.weight(1f))
        TimingCell(entry.avgMs, Modifier.weight(1f), highlight = true)
    }
}

@Composable
private fun TimingCell(ms: Long, modifier: Modifier, highlight: Boolean = false) {
    Text(
        if (ms >= 0) "${ms}ms" else "ERR",
        style = MaterialTheme.typography.bodySmall,
        modifier = modifier,
        fontWeight = if (highlight) FontWeight.Bold else FontWeight.Normal,
        color = if (ms < 0) MaterialTheme.colorScheme.error
        else if (highlight) MaterialTheme.colorScheme.primary
        else MaterialTheme.colorScheme.onSurface,
        fontSize = 12.sp
    )
}

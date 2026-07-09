package com.zimage

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/** On-launch: generate an image on the GPU with the chunked Z-Image DiT and show it. */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    var status by remember { mutableStateOf("Generating…") }
                    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
                    LaunchedEffect(Unit) {
                        val logFile = File(getExternalFilesDir(null), "gen_log.txt")
                        logFile.writeText("")
                        try {
                            val bmp = withContext(Dispatchers.Default) {
                                ZImageGen.run(this@MainActivity) { line ->
                                    status = line; logFile.appendText(line + "\n")
                                }
                            }
                            val out = File(getExternalFilesDir(null), "generated.png")
                            FileOutputStream(out).use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
                            bitmap = bmp
                            status = "Done -> ${out.absolutePath}"
                            logFile.appendText("DONE\n")
                        } catch (e: Throwable) {
                            status = "FAILED: $e"
                            logFile.appendText("FAILED: $e\n")
                        }
                    }
                    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                        Text("Z-Image-Turbo · chunked DiT on LiteRT GPU",
                            style = MaterialTheme.typography.titleMedium)
                        Text(status, modifier = Modifier.padding(top = 8.dp))
                        bitmap?.let {
                            Image(it.asImageBitmap(), contentDescription = "generated",
                                modifier = Modifier.fillMaxWidth().padding(top = 16.dp))
                        }
                    }
                }
            }
        }
    }
}

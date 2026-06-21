package com.da3

import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/**
 * Minimal DA3-SMALL GPU verification app: loads the bundled test image, runs the GPU-clean
 * tflite via CompiledModel, and shows the accelerator used, inference time, and the depth map.
 */
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 48, 24, 24)
        }
        val status = TextView(this).apply { textSize = 18f; text = "Loading DA3-SMALL…" }
        fun imageView() = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                .also { it.marginStart = 8; it.marginEnd = 8 }
            adjustViewBounds = true                       // match the bitmap's aspect ratio
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        val inputView = imageView()
        val depthView = imageView()
        val pair = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).also { it.topMargin = 24 }
            addView(inputView); addView(depthView)
        }
        root.addView(status); root.addView(pair)
        setContentView(ScrollView(this).apply { setBackgroundColor(Color.BLACK); addView(root) })
        status.setTextColor(Color.WHITE)

        thread {
            try {
                val bmp = assets.open("test.jpg").use { BitmapFactory.decodeStream(it) }
                val predictor = DA3Predictor(this)
                val res = predictor.predict(bmp)
                val depthBmp = res.depthBitmap()
                predictor.close()
                runOnUiThread {
                    status.text = "DA3-SMALL  |  ${res.accelerator}  |  ${res.inferenceMs} ms"
                    status.setTextColor(if (res.accelerator == "GPU") Color.GREEN else Color.YELLOW)
                    inputView.setImageBitmap(bmp)
                    depthView.setImageBitmap(depthBmp)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    status.text = "ERROR: ${e.message}"
                    status.setTextColor(Color.RED)
                }
            }
        }
    }
}

package com.ram

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

/**
 * RAM++ multi-label image tagging, on-device. Pick a photo (or use the bundled sample) and get the
 * tags. Swin encoder stages 0-2 + Query2Label tag head run on the LiteRT CompiledModel GPU; the
 * fp16-fragile deep Swin block + the 479MB frozen tag bank run on CPU.
 */
class MainActivity : Activity() {

    private val bg = Executors.newSingleThreadExecutor()
    private var tagger: RamTagger? = null
    private lateinit var status: TextView
    private lateinit var tagsView: TextView
    private lateinit var imageView: ImageView
    private var bitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(36, 90, 36, 36) }
        status = TextView(this).apply { textSize = 15f; text = "Loading RAM++ …" }
        val pick = Button(this).apply { text = "🖼  Pick image"; setOnClickListener { pickImage() } }
        imageView = ImageView(this).apply { adjustViewBounds = true }
        tagsView = TextView(this).apply { textSize = 15f; setPadding(0, 20, 0, 0) }
        root.addView(status); root.addView(pick); root.addView(imageView); root.addView(tagsView)
        setContentView(ScrollView(this).apply { addView(root) })

        bg.execute {
            try {
                tagger = RamTagger(this)
                bitmap = assets.open("test_image.jpg").use { BitmapFactory.decodeStream(it) }
                runOnUiThread { imageView.setImageBitmap(bitmap); status.text = "Ready — tagging sample…" }
                runTag()
            } catch (e: Throwable) {
                Log.e("RAM", "load", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    private fun pickImage() {
        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE); type = "image/*" }, 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        val uri: Uri = data?.data ?: return
        if (resultCode != RESULT_OK) return
        contentResolver.openInputStream(uri).use { bitmap = BitmapFactory.decodeStream(it) }
        imageView.setImageBitmap(bitmap)
        runTag()
    }

    private fun runTag() {
        val t = tagger ?: return; val bm = bitmap ?: return
        runOnUiThread { status.text = "Tagging on GPU…"; tagsView.text = "" }
        bg.execute {
            try {
                val t0 = System.nanoTime()
                val res = t.tag(bm)
                val ms = (System.nanoTime() - t0) / 1_000_000
                val line = res.joinToString(" · ") { it.name }
                Log.i("RAM", "TAGS ($ms ms): $line")
                runOnUiThread {
                    status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
                    status.text = "✓ ${res.size} tags in ${ms}ms · RAM++ hybrid GPU/CPU"
                    tagsView.text = res.joinToString("\n") { "• ${it.name}   %.2f".format(it.prob) }
                }
            } catch (e: Throwable) {
                Log.e("RAM", "tag", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "Failed: ${e.message}" }
            }
        }
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); tagger?.close() }
}

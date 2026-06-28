package com.anime

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * AnimeGANv2 photo→anime demo, fully on the CompiledModel GPU. Applies one of 2 anime styles to a bundled
 * image and any image picked from the gallery; tap a style button to switch.
 */
class MainActivity : Activity() {

    private val tag = "ANIME"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: AnimeStylizer? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var imgView: ImgView
    private var content: Bitmap? = null
    private var style = "paprika"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        status = TextView(this).apply { textSize = 15f; text = "Loading AnimeGAN on GPU…" }
        val pick = Button(this).apply {
            text = "🖼  Pick image"; isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        val styleRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        AnimeStylizer.STYLES.forEach { s ->
            styleRow.addView(Button(this).apply {
                text = if (s.startsWith("face")) "face" else s; textSize = 12f
                setOnClickListener { style = s; runStyle() }
            }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }
        imgView = ImgView(this)
        root.addView(status); root.addView(pick); root.addView(styleRow)
        root.addView(imgView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 900))
        setContentView(root)

        bg.execute {
            try {
                net = AnimeStylizer(this)
                try {
                    content = squareResize(BitmapFactory.decodeStream(assets.open("test_image.jpg")))
                    apply(warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick an image, then tap a style." }
                }
                runOnUiThread { pick.isEnabled = true }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != pickReq || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        bg.execute {
            try { content = squareResize(loadOriented(uri)); apply(warm = false) }
            catch (e: Throwable) { Log.e(tag, "stylize failed", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun runStyle() { bg.execute { apply(warm = false) } }

    private fun apply(warm: Boolean) {
        val c = content ?: return
        val n = net!!
        val rgb = bitmapToRgb(c)
        if (warm) n.stylize(rgb, style)
        val t0 = System.nanoTime()
        val out = n.stylize(rgb, style)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "stylize $style ${ms}ms")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU anime: $style ✓  ${ms} ms  ·  AnimeGANv2, CompiledModel GPU"
            imgView.bitmap = out; imgView.invalidate()
        }
    }

    private fun loadOriented(uri: Uri): Bitmap {
        val bm = contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) } ?: error("cannot decode image")
        val rot = contentResolver.openInputStream(uri).use {
            when (ExifInterface(it!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        }
        if (rot == 0f) return bm
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, Matrix().apply { postRotate(rot) }, true)
    }

    private fun squareResize(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, AnimeStylizer.SIZE, AnimeStylizer.SIZE, true)
    }

    private fun bitmapToRgb(bm: Bitmap): FloatArray {
        val n = bm.width * bm.height; val px = IntArray(n)
        bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val p = px[i]
            out[i * 3] = ((p shr 16) and 0xFF).toFloat(); out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
            out[i * 3 + 2] = (p and 0xFF).toFloat()
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); net?.close() }

    class ImgView(ctx: Context) : View(ctx) {
        var bitmap: Bitmap? = null
        private val paint = Paint().apply { isFilterBitmap = true }
        override fun onDraw(canvas: Canvas) {
            val bm = bitmap ?: return
            val s = min(width.toFloat() / bm.width, height.toFloat() / bm.height)
            val w = bm.width * s; val h = bm.height * s
            canvas.drawBitmap(bm, null, android.graphics.RectF((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2), paint)
        }
    }
}

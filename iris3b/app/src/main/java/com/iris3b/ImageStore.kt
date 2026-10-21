package com.iris3b

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Saves a generated image where the user finds it. */
object ImageStore {

    /**
     * Writes [bitmap] as a PNG to Pictures/iris-3b (MediaStore, Android 10+, no permission needed) or,
     * on older phones, to the app's own pictures dir. Returns where it went.
     */
    fun savePng(context: Context, bitmap: Bitmap): String {
        val name = "iris3b_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".png"
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, name)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
                put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$FOLDER")
            }
            val resolver = context.contentResolver
            val uri = checkNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)) {
                "MediaStore did not take the image"
            }
            val out = checkNotNull(resolver.openOutputStream(uri)) { "cannot open $uri" }
            out.use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
            return "${Environment.DIRECTORY_PICTURES}/$FOLDER/$name"
        }
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES) ?: context.filesDir
        val file = File(dir, name)
        FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return file.absolutePath
    }

    private const val FOLDER = "iris-3b"
}

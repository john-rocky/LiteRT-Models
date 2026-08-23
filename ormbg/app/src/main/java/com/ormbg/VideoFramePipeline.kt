package com.ormbg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaPlayer
import android.view.Surface
import android.view.TextureView

/**
 * Feeds frames from a bundled clip instead of the camera, so a demo is reproducible and
 * both accelerator builds see identical input. Same callback shape as
 * [RealtimeCameraPipeline]: [onFrame] runs off the main thread and must not touch views.
 */
class VideoFramePipeline(
    private val context: Context,
    private val textureView: TextureView,
    private val assetName: String,
    private val onFrame: (Bitmap) -> Unit,
) {
    var enabled: Boolean = true

    private var player: MediaPlayer? = null
    private var worker: Thread? = null
    @Volatile private var running = false

    fun start() {
        // The surface is often already available by the time this is called — starting
        // after the model has loaded, for instance — and the listener would never fire.
        textureView.surfaceTexture?.let { begin(it); return }

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = begin(st)
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) = Unit
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean { stop(); return true }
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) = Unit
        }
    }

    private fun begin(st: SurfaceTexture) {
        val afd = context.assets.openFd(assetName)
        player = MediaPlayer().apply {
            setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
            setSurface(Surface(st))
            isLooping = true
            setOnPreparedListener { it.start() }
            prepareAsync()
        }
        afd.close()

        running = true
        worker = Thread {
            while (running) {
                if (enabled && textureView.isAvailable) {
                    // getBitmap allocates per call; at demo frame rates that is cheaper
                    // than wiring an ImageReader, and keeps this file short.
                    textureView.bitmap?.let(onFrame)
                }
                Thread.sleep(FRAME_INTERVAL_MS)
            }
        }.also { it.start() }
    }

    fun stop() {
        running = false
        worker?.join(500)
        worker = null
        player?.run { if (isPlaying) stop(); release() }
        player = null
    }

    private companion object {
        const val FRAME_INTERVAL_MS = 33L
    }
}

package com.edgeagent.lab

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import kotlin.coroutines.resume

/**
 * Holds one MediaProjection for a whole agent session and serves frames on
 * demand.
 *
 * The earlier version took a single frame and tore the projection down, which
 * meant a consent dialog per capture. An agent loop is capture → act → capture,
 * so the projection has to outlive the frame: the consent token is single-use
 * on API 34+, but the *projection* it creates can be read as many times as you
 * like. [grab] is therefore cheap after the first one.
 *
 * Android rules this still dances around: the foreground service of type
 * mediaProjection must be running before `getMediaProjection()`, and a
 * MediaProjection.Callback must be registered before `createVirtualDisplay`.
 */
class CaptureService : Service() {

    companion object {
        private const val TAG = "CaptureService"
        private const val CHANNEL = "capture"
        private const val NOTIFICATION_ID = 42
        private const val FRAME_TIMEOUT_MS = 4000L

        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_RESULT_DATA = "resultData"
        const val EXTRA_WIDTH = "width"
        const val EXTRA_HEIGHT = "height"
        const val EXTRA_DENSITY = "density"

        @Volatile
        private var instance: CaptureService? = null

        /** Called once when the projection is live, or with false if it failed. */
        @Volatile
        var onReady: ((Boolean) -> Unit)? = null

        val isRunning: Boolean get() = instance?.projection != null

        fun start(
            context: Context,
            resultCode: Int,
            data: Intent,
            width: Int,
            height: Int,
            density: Int,
        ) {
            context.startForegroundService(
                Intent(context, CaptureService::class.java).apply {
                    putExtra(EXTRA_RESULT_CODE, resultCode)
                    putExtra(EXTRA_RESULT_DATA, data)
                    putExtra(EXTRA_WIDTH, width)
                    putExtra(EXTRA_HEIGHT, height)
                    putExtra(EXTRA_DENSITY, density)
                }
            )
        }

        /** Grabs the current screen. Null if the projection is not live. */
        fun grab(onFrame: (Bitmap?) -> Unit) {
            val service = instance
            if (service == null) {
                onFrame(null)
                return
            }
            service.grabFrame(onFrame)
        }

        fun finish(context: Context) {
            context.stopService(Intent(context, CaptureService::class.java))
        }
    }

    private var projection: MediaProjection? = null
    private var reader: ImageReader? = null
    private var display: VirtualDisplay? = null
    private var width = 0
    private var height = 0
    private val handler = Handler(Looper.getMainLooper())

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundNotification()
        if (intent == null || projection != null) return START_NOT_STICKY

        @Suppress("DEPRECATION")
        val data: Intent? = intent.getParcelableExtra(EXTRA_RESULT_DATA)
        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
        width = intent.getIntExtra(EXTRA_WIDTH, 0)
        height = intent.getIntExtra(EXTRA_HEIGHT, 0)
        val density = intent.getIntExtra(EXTRA_DENSITY, 320)

        if (data == null || width <= 0 || height <= 0) {
            notifyReady(false)
            return START_NOT_STICKY
        }

        val media = try {
            getSystemService(MediaProjectionManager::class.java)
                .getMediaProjection(resultCode, data)
        } catch (t: Throwable) {
            Log.e(TAG, "getMediaProjection failed", t)
            null
        }
        if (media == null) {
            notifyReady(false)
            return START_NOT_STICKY
        }
        projection = media
        // Required before createVirtualDisplay on API 34+.
        media.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                Log.i(TAG, "projection stopped by the system")
                teardown()
            }
        }, handler)

        reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        display = media.createVirtualDisplay(
            "edge-agent-lab",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader!!.surface,
            null,
            handler,
        )
        notifyReady(true)
        return START_NOT_STICKY
    }

    /**
     * Waits for the next frame rather than reading whatever is buffered — after
     * a tap the buffered frame is the screen from *before* it.
     */
    private fun grabFrame(onFrame: (Bitmap?) -> Unit) {
        val r = reader
        if (r == null || projection == null) {
            handler.post { onFrame(null) }
            return
        }
        // Drain anything stale first.
        while (true) {
            val old = r.acquireLatestImage() ?: break
            old.close()
        }
        var delivered = false
        r.setOnImageAvailableListener({ source ->
            if (delivered) return@setOnImageAvailableListener
            val image = source.acquireLatestImage() ?: return@setOnImageAvailableListener
            val bitmap = try {
                com.edgeagent.sdk.ScreenFrames.toBitmap(image, width, height)
            } catch (t: Throwable) {
                Log.e(TAG, "frame conversion failed", t)
                null
            } finally {
                image.close()
            }
            delivered = true
            source.setOnImageAvailableListener(null, null)
            handler.post { onFrame(bitmap) }
        }, handler)

        handler.postDelayed({
            if (!delivered) {
                delivered = true
                r.setOnImageAvailableListener(null, null)
                onFrame(null)
            }
        }, FRAME_TIMEOUT_MS)
    }

    private fun notifyReady(ok: Boolean) {
        val callback = onReady
        handler.post { callback?.invoke(ok) }
    }

    private fun teardown() {
        runCatching { display?.release() }
        runCatching { reader?.close() }
        runCatching { projection?.stop() }
        display = null
        reader = null
        projection = null
    }

    private fun startForegroundNotification() {
        instance = this
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL) == null) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL, "Screen capture", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val tap = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val notification: Notification = Notification.Builder(this, CHANNEL)
            .setContentTitle("Screen agent is watching")
            .setContentText("Grounding and tapping run on-device; tap to return")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentIntent(tap)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    override fun onDestroy() {
        teardown()
        instance = null
        super.onDestroy()
    }
}

/** Adapts the projection service to the SDK's [com.edgeagent.sdk.ScreenSource] seam. */
suspend fun captureScreen(): android.graphics.Bitmap? =
    kotlinx.coroutines.suspendCancellableCoroutine { cont ->
        CaptureService.grab { bitmap -> if (cont.isActive) cont.resume(bitmap) }
    }

package com.yolotracking

import android.content.Context
import android.graphics.*
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import java.io.File
import java.nio.ByteBuffer

/**
 * Processes a video file: extracts frames via MediaExtractor + MediaCodec,
 * runs YOLO + Re-ID + DeepSORT, draws tracking overlay, encodes to MP4 via EGL.
 *
 * Decoding pipeline:
 *   MediaExtractor → MediaCodec(decoder) → ImageReader → YUV→Bitmap
 *
 * Encoding pipeline:
 *   composite Bitmap → EGL → MediaCodec(encoder) → MediaMuxer
 */
class VideoProcessor(
    private val context: Context,
    private val detector: ObjectDetector,
    private val reIdExtractor: ReIDExtractor,
    private val reIdInterval: Int = 3,
) {
    companion object {
        private const val TAG = "VideoProcessor"
        private const val ENC_MIME = "video/avc"
        private const val BIT_RATE = 8_000_000
        private const val I_FRAME_INTERVAL = 1
        private const val DECODE_TIMEOUT_US = 10_000L
    }

    interface ProgressListener {
        fun onProgress(currentFrame: Int, totalFrames: Int)
        fun onComplete(outputFile: File)
        fun onError(message: String)
    }

    fun process(videoUri: Uri, listener: ProgressListener) {
        try {
            processInternal(videoUri, listener)
        } catch (e: Exception) {
            Log.e(TAG, "Video processing failed", e)
            listener.onError("${e.javaClass.simpleName}: ${e.message ?: "unknown"}")
        }
    }

    private fun processInternal(videoUri: Uri, listener: ProgressListener) {
        // ── Open the source video with MediaExtractor ──
        val extractor = MediaExtractor()
        extractor.setDataSource(context, videoUri, null)

        var videoTrack = -1
        var videoFormat: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val fmt = extractor.getTrackFormat(i)
            if (fmt.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true) {
                videoTrack = i
                videoFormat = fmt
                break
            }
        }
        if (videoTrack < 0 || videoFormat == null) {
            extractor.release()
            listener.onError("No video track found")
            return
        }
        extractor.selectTrack(videoTrack)

        val srcMime = videoFormat.getString(MediaFormat.KEY_MIME)!!
        val srcW = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
        val srcH = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
        val rotation = if (videoFormat.containsKey(MediaFormat.KEY_ROTATION))
            videoFormat.getInteger(MediaFormat.KEY_ROTATION) else 0
        val durationUs = if (videoFormat.containsKey(MediaFormat.KEY_DURATION))
            videoFormat.getLong(MediaFormat.KEY_DURATION) else 0L
        Log.i(TAG, "Source: $srcMime ${srcW}x${srcH} rot=$rotation dur=${durationUs / 1000}ms")

        // Estimated total frames for progress reporting (best-effort)
        val srcFps = if (videoFormat.containsKey(MediaFormat.KEY_FRAME_RATE))
            videoFormat.getInteger(MediaFormat.KEY_FRAME_RATE).coerceAtLeast(1) else 30
        val estimatedTotal = if (durationUs > 0) (durationUs * srcFps / 1_000_000).toInt() else -1

        // Output dimensions (apply rotation, even-aligned for H.264)
        val outW: Int
        val outH: Int
        if (rotation == 90 || rotation == 270) {
            outW = srcH; outH = srcW
        } else {
            outW = srcW; outH = srcH
        }
        val encW = outW and 1.inv()
        val encH = outH and 1.inv()

        // ── Setup decoder with ImageReader output (YUV_420_888) ──
        val maxImages = 3
        val imageReader = ImageReader.newInstance(srcW, srcH, android.graphics.ImageFormat.YUV_420_888, maxImages)

        val decoder = MediaCodec.createDecoderByType(srcMime)
        decoder.configure(videoFormat, imageReader.surface, null, 0)
        decoder.start()

        // ── Setup encoder with EGL surface input ──
        val outputFile = File(context.getExternalFilesDir(null), "tracked_${System.currentTimeMillis()}.mp4")
        val encFormat = MediaFormat.createVideoFormat(ENC_MIME, encW, encH).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE)
            setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL)
        }
        val encoder = MediaCodec.createEncoderByType(ENC_MIME)
        encoder.configure(encFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val encoderInputSurface = encoder.createInputSurface()
        encoder.start()

        val egl = EncoderEglSurface(encoderInputSurface)
        egl.makeCurrent()

        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var trackIndex = -1
        var muxerStarted = false
        val encBufferInfo = MediaCodec.BufferInfo()

        val tracker = DeepSORTTracker()

        // Composite bitmap for drawing each frame + overlays
        val composite = Bitmap.createBitmap(encW, encH, Bitmap.Config.ARGB_8888)
        val compositeCanvas = Canvas(composite)
        val compositePaint = Paint(Paint.FILTER_BITMAP_FLAG)
        val rotMat = Matrix()

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = 4f }
        val idPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFFFFF.toInt(); textSize = 36f; isFakeBoldText = true
            setShadowLayer(3f, 0f, 0f, 0xFF000000.toInt())
        }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFFFFF.toInt(); textSize = 28f
            setShadowLayer(3f, 0f, 0f, 0xFF000000.toInt())
        }
        val bgPaint = Paint().apply { style = Paint.Style.FILL }
        val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE; strokeWidth = 3f
            strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND
        }
        val trailPath = Path()

        // Bitmap to hold the YUV→RGB converted frame at source dimensions
        val srcBitmap = Bitmap.createBitmap(srcW, srcH, Bitmap.Config.ARGB_8888)

        var frameIndex = 0
        var inputDone = false
        var decodeDone = false
        val decodeBufferInfo = MediaCodec.BufferInfo()

        try {
            while (!decodeDone) {
                // Feed input
                if (!inputDone) {
                    val inIdx = decoder.dequeueInputBuffer(DECODE_TIMEOUT_US)
                    if (inIdx >= 0) {
                        val inBuf = decoder.getInputBuffer(inIdx)!!
                        val sampleSize = extractor.readSampleData(inBuf, 0)
                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(inIdx, 0, 0, 0L,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            decoder.queueInputBuffer(inIdx, 0, sampleSize,
                                extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                // Drain decoder output
                val outIdx = decoder.dequeueOutputBuffer(decodeBufferInfo, DECODE_TIMEOUT_US)
                when {
                    outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> { /* loop */ }
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* ignore */ }
                    outIdx >= 0 -> {
                        val isEnd = (decodeBufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        val render = decodeBufferInfo.size > 0
                        decoder.releaseOutputBuffer(outIdx, render)

                        if (render) {
                            // Pull image from ImageReader
                            val image = imageReader.acquireLatestImage()
                            if (image != null) {
                                yuv420ToBitmap(image, srcBitmap)
                                image.close()

                                processFrame(
                                    srcBitmap, rotation, encW, encH, frameIndex,
                                    tracker, compositeCanvas, composite, rotMat, compositePaint,
                                    boxPaint, idPaint, labelPaint, bgPaint, trailPaint, trailPath,
                                    egl, decodeBufferInfo.presentationTimeUs,
                                )
                                frameIndex++

                                // Drain encoder
                                drainEncoder(encoder, encBufferInfo, muxer, trackIndex, muxerStarted).let { (ti, ms) ->
                                    trackIndex = ti
                                    muxerStarted = ms
                                }

                                val total = if (estimatedTotal > 0) estimatedTotal else frameIndex + 1
                                listener.onProgress(frameIndex, total)
                            }
                        }
                        if (isEnd) decodeDone = true
                    }
                }
            }

            // Signal end of stream to encoder
            encoder.signalEndOfInputStream()
            drainEncoder(encoder, encBufferInfo, muxer, trackIndex, muxerStarted, endOfStream = true)
        } finally {
            try { decoder.stop() } catch (_: Exception) {}
            try { decoder.release() } catch (_: Exception) {}
            try { encoder.stop() } catch (_: Exception) {}
            try { encoder.release() } catch (_: Exception) {}
            try { egl.release() } catch (_: Exception) {}
            try { if (muxerStarted) muxer.stop() } catch (_: Exception) {}
            try { muxer.release() } catch (_: Exception) {}
            try { extractor.release() } catch (_: Exception) {}
            try { imageReader.close() } catch (_: Exception) {}
            srcBitmap.recycle()
            composite.recycle()
        }

        Log.i(TAG, "Output: ${outputFile.absolutePath} (${outputFile.length() / 1024}KB, $frameIndex frames)")
        listener.onComplete(outputFile)
    }

    private fun processFrame(
        srcBitmap: Bitmap,
        rotation: Int,
        encW: Int,
        encH: Int,
        frameIndex: Int,
        tracker: DeepSORTTracker,
        compositeCanvas: Canvas,
        composite: Bitmap,
        rotMat: Matrix,
        compositePaint: Paint,
        boxPaint: Paint,
        idPaint: Paint,
        labelPaint: Paint,
        bgPaint: Paint,
        trailPaint: Paint,
        trailPath: Path,
        egl: EncoderEglSurface,
        timestampUs: Long,
    ) {
        // Apply rotation if needed: composite ends up at encW x encH (rotated dims)
        // We draw srcBitmap into composite with rotation matrix
        compositeCanvas.drawColor(0xFF000000.toInt())
        rotMat.reset()
        when (rotation) {
            90 -> {
                rotMat.postRotate(90f)
                rotMat.postTranslate(srcBitmap.height.toFloat(), 0f)
            }
            180 -> {
                rotMat.postRotate(180f)
                rotMat.postTranslate(srcBitmap.width.toFloat(), srcBitmap.height.toFloat())
            }
            270 -> {
                rotMat.postRotate(270f)
                rotMat.postTranslate(0f, srcBitmap.width.toFloat())
            }
        }
        // Scale to fill encW x encH
        val rotW = if (rotation == 90 || rotation == 270) srcBitmap.height else srcBitmap.width
        val rotH = if (rotation == 90 || rotation == 270) srcBitmap.width else srcBitmap.height
        val sx = encW.toFloat() / rotW
        val sy = encH.toFloat() / rotH
        rotMat.postScale(sx, sy)
        compositeCanvas.drawBitmap(srcBitmap, rotMat, compositePaint)

        // Run detection on the COMPOSITE bitmap so coordinates match
        val (rawDets, _) = detector.detect(composite)
        val runReId = (frameIndex % reIdInterval == 0)
        val detsWithFeatures = if (runReId) {
            reIdExtractor.extractFeatures(composite, rawDets)
        } else {
            rawDets
        }
        val tracked = tracker.update(detsWithFeatures)
        val activeTracks = tracker.activeTracks

        // Draw trails
        for (track in activeTracks) {
            if (track.trail.size < 2) continue
            val color = CocoLabels.trackColor(track.trackId)
            trailPaint.color = (color and 0x00FFFFFF) or 0x99000000.toInt()
            trailPath.reset()
            var first = true
            for ((tx, ty) in track.trail) {
                if (first) { trailPath.moveTo(tx, ty); first = false }
                else trailPath.lineTo(tx, ty)
            }
            compositeCanvas.drawPath(trailPath, trailPaint)
        }

        // Draw boxes
        for (det in tracked) {
            val color = CocoLabels.trackColor(det.trackId)
            boxPaint.color = color
            compositeCanvas.drawRect(det.xMin, det.yMin, det.xMax, det.yMax, boxPaint)

            val idLabel = "#${det.trackId}"
            val idW = idPaint.measureText(idLabel)
            val idH = idPaint.textSize
            bgPaint.color = (color and 0x00FFFFFF) or 0xDD000000.toInt()
            compositeCanvas.drawRect(det.xMin, det.yMin - idH - 8, det.xMin + idW + 16, det.yMin, bgPaint)
            compositeCanvas.drawText(idLabel, det.xMin + 8, det.yMin - 8, idPaint)

            val clsW = labelPaint.measureText(det.className)
            val clsH = labelPaint.textSize
            bgPaint.color = 0x88000000.toInt()
            compositeCanvas.drawRect(det.xMin, det.yMin, det.xMin + clsW + 16, det.yMin + clsH + 8, bgPaint)
            compositeCanvas.drawText(det.className, det.xMin + 8, det.yMin + clsH, labelPaint)
        }

        // Render to encoder
        egl.drawBitmap(composite)
        egl.setPresentationTime(timestampUs * 1000L)  // us → ns
        egl.swapBuffers()
    }

    /**
     * Convert YUV_420_888 Image to ARGB_8888 bitmap. Uses straightforward
     * software conversion - acceptable since this runs once per frame
     * alongside heavier ML inference.
     */
    private fun yuv420ToBitmap(image: Image, dst: Bitmap) {
        val w = image.width
        val h = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val pixels = IntArray(w * h)
        var idx = 0
        for (y in 0 until h) {
            val yRow = y * yRowStride
            val uvRow = (y shr 1) * uvRowStride
            for (x in 0 until w) {
                val Y = (yBuf.get(yRow + x).toInt() and 0xFF)
                val uvCol = (x shr 1) * uvPixelStride
                val U = (uBuf.get(uvRow + uvCol).toInt() and 0xFF) - 128
                val V = (vBuf.get(uvRow + uvCol).toInt() and 0xFF) - 128
                // BT.601 YUV → RGB
                var r = Y + (1.402f * V).toInt()
                var g = Y - (0.344f * U + 0.714f * V).toInt()
                var b = Y + (1.772f * U).toInt()
                if (r < 0) r = 0 else if (r > 255) r = 255
                if (g < 0) g = 0 else if (g > 255) g = 255
                if (b < 0) b = 0 else if (b > 255) b = 255
                pixels[idx++] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        dst.setPixels(pixels, 0, w, 0, 0, w, h)
    }

    private fun drainEncoder(
        encoder: MediaCodec,
        bufferInfo: MediaCodec.BufferInfo,
        muxer: MediaMuxer,
        trackIndex: Int,
        muxerStarted: Boolean,
        endOfStream: Boolean = false,
    ): Pair<Int, Boolean> {
        var ti = trackIndex
        var ms = muxerStarted
        val timeoutUs = if (endOfStream) 10_000L else 0L

        while (true) {
            val outIdx = encoder.dequeueOutputBuffer(bufferInfo, timeoutUs)
            if (outIdx == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) break
                else continue
            } else if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                ti = muxer.addTrack(encoder.outputFormat)
                muxer.start()
                ms = true
            } else if (outIdx >= 0) {
                val buf = encoder.getOutputBuffer(outIdx) ?: continue
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                    bufferInfo.size = 0
                }
                if (bufferInfo.size > 0 && ms) {
                    buf.position(bufferInfo.offset)
                    buf.limit(bufferInfo.offset + bufferInfo.size)
                    muxer.writeSampleData(ti, buf, bufferInfo)
                }
                encoder.releaseOutputBuffer(outIdx, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
            }
        }
        return ti to ms
    }
}

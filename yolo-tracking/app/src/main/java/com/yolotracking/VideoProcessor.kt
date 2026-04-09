package com.yolotracking

import android.content.Context
import android.graphics.*
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.util.Log
import java.io.File

/**
 * Processes a video file: extracts frames, runs YOLO + Re-ID + DeepSORT,
 * draws tracking overlay, and encodes the result to an MP4 file.
 */
class VideoProcessor(
    private val context: Context,
    private val detector: ObjectDetector,
    private val reIdExtractor: ReIDExtractor,
    private val reIdInterval: Int = 3,
) {
    companion object {
        private const val TAG = "VideoProcessor"
        private const val MIME_TYPE = "video/avc"
        private const val BIT_RATE = 8_000_000
        private const val I_FRAME_INTERVAL = 1
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
            listener.onError(e.message ?: "Unknown error")
        }
    }

    private fun processInternal(videoUri: Uri, listener: ProgressListener) {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)

        val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        val videoWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
        val videoHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
        val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
        val frameRate = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull() ?: 30f

        // Account for rotation
        val outWidth: Int
        val outHeight: Int
        if (rotation == 90 || rotation == 270) {
            outWidth = videoHeight
            outHeight = videoWidth
        } else {
            outWidth = videoWidth
            outHeight = videoHeight
        }

        Log.i(TAG, "Video: ${outWidth}x${outHeight}, ${durationMs}ms, rotation=$rotation, fps=$frameRate")

        val fps = frameRate.coerceIn(1f, 60f)
        val totalFrames = (durationMs / 1000.0 * fps).toInt().coerceAtLeast(1)

        // Setup encoder
        val outputFile = File(context.getExternalFilesDir(null), "tracked_${System.currentTimeMillis()}.mp4")
        val format = MediaFormat.createVideoFormat(MIME_TYPE, outWidth, outHeight).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps.toInt())
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL)
        }

        val encoder = MediaCodec.createEncoderByType(MIME_TYPE)
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val inputSurface = encoder.createInputSurface()
        encoder.start()

        val muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var trackIndex = -1
        var muxerStarted = false

        val tracker = DeepSORTTracker()

        // Drawing tools
        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        val idPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFFFFF.toInt()
            textSize = 36f
            isFakeBoldText = true
            setShadowLayer(3f, 0f, 0f, 0xFF000000.toInt())
        }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFFFFF.toInt()
            textSize = 28f
            setShadowLayer(3f, 0f, 0f, 0xFF000000.toInt())
        }
        val bgPaint = Paint().apply { style = Paint.Style.FILL }
        val trailPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3f
            strokeCap = Paint.Cap.ROUND
            strokeJoin = Paint.Join.ROUND
        }
        val trailPath = Path()

        val bufferInfo = MediaCodec.BufferInfo()
        var frameIndex = 0

        // Extract and process frames
        val intervalUs = (1_000_000.0 / fps).toLong()
        for (i in 0 until totalFrames) {
            val timeUs = i * intervalUs
            val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                ?: continue

            // Run detection + tracking
            val (rawDets, _) = detector.detect(frame)
            val runReId = (frameIndex % reIdInterval == 0)
            val detsWithFeatures = if (runReId) {
                reIdExtractor.extractFeatures(frame, rawDets)
            } else {
                rawDets
            }
            val tracked = tracker.update(detsWithFeatures)
            val activeTracks = tracker.activeTracks
            frameIndex++

            // Draw overlay on encoder surface
            val canvas = inputSurface.lockCanvas(null)
            canvas.drawBitmap(frame, 0f, 0f, null)

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
                canvas.drawPath(trailPath, trailPaint)
            }

            // Draw boxes
            for (det in tracked) {
                val color = CocoLabels.trackColor(det.trackId)
                boxPaint.color = color
                canvas.drawRect(det.xMin, det.yMin, det.xMax, det.yMax, boxPaint)

                val idLabel = "#${det.trackId}"
                val idW = idPaint.measureText(idLabel)
                val idH = idPaint.textSize
                bgPaint.color = (color and 0x00FFFFFF) or 0xDD000000.toInt()
                canvas.drawRect(det.xMin, det.yMin - idH - 8, det.xMin + idW + 16, det.yMin, bgPaint)
                canvas.drawText(idLabel, det.xMin + 8, det.yMin - 8, idPaint)

                val clsW = labelPaint.measureText(det.className)
                val clsH = labelPaint.textSize
                bgPaint.color = 0x88000000.toInt()
                canvas.drawRect(det.xMin, det.yMin, det.xMin + clsW + 16, det.yMin + clsH + 8, bgPaint)
                canvas.drawText(det.className, det.xMin + 8, det.yMin + clsH, labelPaint)
            }

            inputSurface.unlockCanvasAndPost(canvas)
            frame.recycle()

            // Drain encoder
            drainEncoder(encoder, bufferInfo, muxer, trackIndex, muxerStarted).let { (ti, ms) ->
                trackIndex = ti
                muxerStarted = ms
            }

            listener.onProgress(i + 1, totalFrames)
        }

        // Signal end of stream
        encoder.signalEndOfInputStream()
        drainEncoder(encoder, bufferInfo, muxer, trackIndex, muxerStarted, endOfStream = true)

        encoder.stop()
        encoder.release()
        inputSurface.release()
        if (muxerStarted) muxer.stop()
        muxer.release()
        retriever.release()

        Log.i(TAG, "Output: ${outputFile.absolutePath} (${outputFile.length() / 1024}KB)")
        listener.onComplete(outputFile)
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

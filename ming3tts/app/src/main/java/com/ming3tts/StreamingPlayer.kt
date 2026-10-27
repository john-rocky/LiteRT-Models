package com.ming3tts

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import java.util.concurrent.LinkedBlockingQueue
import kotlin.math.abs
import kotlin.math.max

/**
 * Plays 44.1 kHz float PCM as it arrives. The engine thread [offer]s chunks without blocking;
 * a player thread writes them to a MODE_STREAM AudioTrack, so a slow speaker never stalls the
 * model and the model never waits for the speaker.
 */
class StreamingPlayer(private val sampleRate: Int = HostMath.SAMPLE_RATE) {
    private val queue = LinkedBlockingQueue<FloatArray>()
    private val end = FloatArray(0)
    @Volatile private var stopped = false
    @Volatile var firstWriteNanos = 0L
        private set
    @Volatile var samplesWritten = 0L
        private set
    var underruns = 0
        private set

    private val track: AudioTrack = run {
        val min = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        AudioTrack(
            AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build(),
            AudioFormat.Builder().setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build(),
            max(min, sampleRate / 2 * 4), AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE,
        )
    }

    private val thread = Thread({
        try {
            track.play()
            while (!stopped) {
                val pcm = queue.take()
                if (pcm === end) break
                var off = 0
                while (off < pcm.size && !stopped) {
                    val n = track.write(pcm, off, pcm.size - off, AudioTrack.WRITE_BLOCKING)
                    if (n <= 0) break
                    if (firstWriteNanos == 0L) firstWriteNanos = System.nanoTime()
                    off += n
                    samplesWritten += n
                }
            }
            if (!stopped) {
                // let the buffered tail play out
                val total = samplesWritten
                val deadline = System.nanoTime() + 3_000_000_000L
                while (!stopped && track.playbackHeadPosition < total && System.nanoTime() < deadline) {
                    Thread.sleep(20)
                }
            }
        } catch (e: InterruptedException) {
            // stop()
        } catch (e: Throwable) {
            Log.e(Graph.TAG, "playback failed", e)
        } finally {
            underruns = track.underrunCount
            try {
                track.stop()
            } catch (_: IllegalStateException) {
            }
            track.release()
        }
    }, "ming3tts-player")

    init {
        thread.start()
    }

    fun offer(pcm: FloatArray) {
        if (!stopped && pcm.isNotEmpty()) queue.offer(pcm)
    }

    /** No more audio; the thread ends once the queue has played out. */
    fun finish() {
        queue.offer(end)
    }

    fun awaitEnd(timeoutMs: Long) = thread.join(timeoutMs)

    /** Stop now and drop what is queued. */
    fun stop() {
        stopped = true
        queue.clear()
        queue.offer(end)
        thread.interrupt()
    }
}

/**
 * Streaming form of upstream `trim_trailing_silence` (ming_tts.py), one per fragment: scores
 * 0.1 s frames by mean |x|, keeps everything up to the last frame above 1e-3 plus a 0.3 s tail,
 * drops the sub-frame remainder. Audio inside the kept region passes through at once; quiet
 * frames past the tail wait until a loud frame proves they are kept.
 */
class SilenceTrimmer(sampleRate: Int, private val out: (FloatArray) -> Unit) {
    private val frame = (sampleRate * 0.1).toInt()
    private val tail = (sampleRate * 0.3).toInt()
    private val cur = FloatArray(frame)
    private var curN = 0
    private val held = ArrayList<FloatArray>()   // complete quiet frames beyond the kept region
    private var frames = 0                        // complete frames seen
    private var keepUpTo = tail.toLong()          // samples [0, keepUpTo) are kept for sure
    private var emitted = 0L
    private val head = ArrayList<FloatArray>()    // everything, while fewer than one frame arrived

    fun push(pcm: FloatArray) {
        if (frames == 0) head.add(pcm.copyOf())
        var i = 0
        while (i < pcm.size) {
            val n = minOf(frame - curN, pcm.size - i)
            System.arraycopy(pcm, i, cur, curN, n)
            curN += n
            i += n
            if (curN == frame) completeFrame()
        }
    }

    private fun completeFrame() {
        var s = 0.0
        for (v in cur) s += abs(v)
        val loud = s / frame > 1e-3
        val f = cur.copyOf()
        val startSample = frames.toLong() * frame
        frames++
        curN = 0
        head.clear()
        if (loud) {
            for (h in held) emit(h)
            held.clear()
            keepUpTo = startSample + frame + tail
            emit(f)
            return
        }
        // quiet: the part inside [.., keepUpTo) is kept whatever comes later
        val keep = (keepUpTo - startSample).coerceIn(0L, frame.toLong()).toInt()
        if (held.isEmpty() && keep == frame) {
            emit(f)
        } else if (held.isEmpty() && keep > 0) {
            emit(f.copyOfRange(0, keep))
            held.add(f.copyOfRange(keep, frame))
        } else {
            held.add(f)
        }
    }

    private fun emit(a: FloatArray) {
        if (a.isEmpty()) return
        emitted += a.size
        out(a)
    }

    /** End of the fragment. */
    fun finish() {
        if (frames == 0) {
            // shorter than one frame: upstream keeps speech[:tail] = all of it
            for (h in head) emit(h)
        }
        held.clear()
        head.clear()
    }
}

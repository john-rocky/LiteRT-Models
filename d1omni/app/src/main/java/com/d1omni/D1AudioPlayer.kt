package com.d1omni

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTimestamp
import android.media.AudioTrack
import java.io.Closeable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * What one playback did ([D1AudioPlayer]); every time is `System.nanoTime()`. [playNanos] = the
 * `play()` call; [startNanos] = when the playback head was first seen moving: the moment frame 0
 * left the output by the [AudioTimestamp] (`timestamp`), else the first poll that read a non-zero
 * playback head position (`head`); [endNanos] = when the last frame had left the output (or the
 * stop / the time limit, see [end]).
 */
class D1Playback(
  val frames: Int,
  val sampleRate: Int,
  val playNanos: Long,
  val startNanos: Long?,
  val startSource: String?,
  val endNanos: Long,
  /** "played", "stopped" or "timeout". */
  val end: String,
  val headAtEnd: Int,
) {
  /** From `play()` to the first moving head, in milliseconds (null when it never moved). */
  val startDelayMs: Double?
    get() = startNanos?.let { (it - playNanos) / 1e6 }

  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "frames" to frames,
      "sample_rate" to sampleRate,
      "start_delay_ms" to startDelayMs,
      "start_source" to startSource,
      "play_to_end_ms" to (endNanos - playNanos) / 1e6,
      "end" to end,
      "head_at_end" to headAtEnd,
    )
}

/**
 * Plays a 16 kHz mono int16 clip through the speaker (`AudioTrack`: USAGE_MEDIA / CONTENT_TYPE_SPEECH,
 * PCM 16-bit, one static buffer) at the phone's own media volume (never changed here), and reports
 * when the sound really started: the playback head starts moving a fraction of a second after
 * `play()`, so a "playing" mark shown at `play()` would lead the sound. A watcher thread polls the
 * track every [POLL_MILLIS] ms: the first [AudioTimestamp] with a moving frame position gives the
 * time frame 0 left the output; without one, the first non-zero `playbackHeadPosition`. The clip
 * has played once the head is within 20 ms of its last frame and the output time of its last frame
 * has passed (without a timestamp: 200 ms after the head got there). One clip at a
 * time; [stop] ends it, [close] releases the track (a clip that played out keeps its track until
 * the next [play], [stop] or [close]). Do not call [play] from the callbacks: they run on the watcher.
 */
class D1AudioPlayer : Closeable {
  private val lock = Any()
  private var track: AudioTrack? = null
  private var watcher: Thread? = null
  @Volatile private var stopRequested = false

  /** `System.nanoTime()` of the last `play()` call (set before its watcher starts, so its callbacks can read it). */
  @Volatile
  var lastPlayNanos: Long = 0L
    private set

  /**
   * Starts [samples] (16 kHz mono) and returns at once. [onStarted] runs once on the watcher thread
   * with the start time (see [D1Playback.startNanos]); [onDone] runs on it at the end.
   */
  fun play(
    samples: ShortArray,
    sampleRate: Int = D1Audio.SAMPLE_RATE,
    onStarted: (startNanos: Long) -> Unit = {},
    onDone: (D1Playback) -> Unit = {},
  ) {
    require(samples.isNotEmpty()) { "nothing to play" }
    synchronized(lock) {
      stopLocked()
      stopRequested = false
      val bytes = samples.size * Short.SIZE_BYTES
      val created =
        AudioTrack.Builder()
          .setAudioAttributes(
            AudioAttributes.Builder()
              .setUsage(AudioAttributes.USAGE_MEDIA)
              .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
              .build()
          )
          .setAudioFormat(
            AudioFormat.Builder()
              .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
              .setSampleRate(sampleRate)
              .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
              .build()
          )
          .setTransferMode(AudioTrack.MODE_STATIC)
          .setBufferSizeInBytes(bytes)
          .build()
      try {
        val written = created.write(samples, 0, samples.size)
        check(written == samples.size) { "AudioTrack took $written of ${samples.size} samples" }
      } catch (failure: Throwable) {
        created.release()
        throw failure
      }
      track = created
      val playNanos = System.nanoTime()
      lastPlayNanos = playNanos
      created.play()
      watcher =
        Thread({ watch(created, samples.size, sampleRate, playNanos, onStarted, onDone) }, "D1Omni-Playback")
          .apply {
            isDaemon = true
            start()
          }
    }
  }

  /** [play], then waits until the clip has played out (or was stopped) and returns its record. */
  fun playBlocking(
    samples: ShortArray,
    sampleRate: Int = D1Audio.SAMPLE_RATE,
    onStarted: (startNanos: Long) -> Unit = {},
  ): D1Playback {
    val done = CountDownLatch(1)
    var result: D1Playback? = null
    play(samples, sampleRate, onStarted) {
      result = it
      done.countDown()
    }
    val limitMs = samples.size * 1000L / sampleRate + END_SLACK_MILLIS + START_LIMIT_MILLIS
    check(done.await(limitMs + POLL_MILLIS * 10, TimeUnit.MILLISECONDS)) { "the playback did not end" }
    return requireNotNull(result)
  }

  /** Ends the current clip (its record says `stopped`). */
  fun stop() {
    synchronized(lock) { stopLocked() }
  }

  override fun close() = stop()

  private fun stopLocked() {
    stopRequested = true
    val running = watcher
    watcher = null
    running?.interrupt()
    running?.join(JOIN_MILLIS)
    track?.let {
      runCatching { it.stop() }
      it.release()
    }
    track = null
  }

  private fun watch(
    played: AudioTrack,
    frames: Int,
    rate: Int,
    playNanos: Long,
    onStarted: (Long) -> Unit,
    onDone: (D1Playback) -> Unit,
  ) {
    val timestamp = AudioTimestamp()
    val durationNanos = frames * NANOS_PER_SECOND / rate
    var startNanos: Long? = null
    var startSource: String? = null
    var lastFrame = 0L
    var firstOutputNanos = 0L
    var head = 0
    var headEndNanos = 0L
    var end = "timeout"
    try {
      while (!stopRequested) {
        val now = System.nanoTime()
        // Only a timestamp whose position moved: after the buffer runs out the track keeps reporting
        // the last position with a fresh time.
        if (played.getTimestamp(timestamp) && timestamp.framePosition > lastFrame && timestamp.framePosition <= frames) {
          lastFrame = timestamp.framePosition
          firstOutputNanos = timestamp.nanoTime - timestamp.framePosition * NANOS_PER_SECOND / rate
        }
        head = played.playbackHeadPosition
        // The head of a static track can stop short of the last frame (the Fun-ASR demo's player on the
        // same phone counts it at the end within 20 ms of it).
        if (head >= frames - rate / HEAD_END_PARTS && headEndNanos == 0L) headEndNanos = now
        if (startNanos == null && (firstOutputNanos > 0 || head > 0)) {
          startNanos = if (firstOutputNanos > 0) firstOutputNanos else now
          startSource = if (firstOutputNanos > 0) "timestamp" else "head"
          onStarted(startNanos)
        }
        val outputDone = firstOutputNanos > 0 && headEndNanos > 0 && now >= firstOutputNanos + durationNanos
        val headDone = firstOutputNanos == 0L && headEndNanos > 0 && now >= headEndNanos + END_SLACK_MILLIS * 1_000_000L
        if (outputDone || headDone) {
          end = "played"
          break
        }
        if (now - playNanos > durationNanos + (START_LIMIT_MILLIS + END_SLACK_MILLIS) * 1_000_000L) break
        Thread.sleep(POLL_MILLIS)
      }
      if (stopRequested && end != "played") end = "stopped"
    } catch (interrupted: InterruptedException) {
      end = "stopped"
    }
    onDone(D1Playback(frames, rate, playNanos, startNanos, startSource, System.nanoTime(), end, head))
  }

  private companion object {
    const val POLL_MILLIS = 2L
    const val NANOS_PER_SECOND = 1_000_000_000L
    /** The longest wait for the head to start moving, beyond the clip's length. */
    const val START_LIMIT_MILLIS = 3_000L
    /** After the head reached the last frame without an output timestamp: the output's latency. */
    const val END_SLACK_MILLIS = 200L
    const val JOIN_MILLIS = 500L
    /** The head counts as at the end within 1 / 50 s of the last frame. */
    const val HEAD_END_PARTS = 50
  }
}

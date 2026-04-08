package com.voiceassistant

/**
 * Speech-segment hysteresis on top of per-chunk Silero VAD probabilities.
 *
 * One chunk = 32 ms (512 samples @ 16 kHz). Time-based parameters are converted
 * to chunk counts at construction.
 *
 * State machine:
 *   SILENCE  --(prob >= startThreshold)-->  SPEECH (emit START)
 *   SPEECH   --(prob <  endThreshold for >= minSilenceChunks consecutive)-->
 *                                           SILENCE (emit END if segment >= minSpeechChunks)
 *
 * Lower [endThreshold] than [startThreshold] gives hysteresis so a single dip
 * mid-utterance does not chop the segment in half.
 */
class SegmentTracker(
    private val startThreshold: Float = 0.5f,
    private val endThreshold: Float = 0.35f,
    minSpeechMs: Int = 250,
    minSilenceMs: Int = 600,
    chunkMs: Int = 32,
) {
    enum class Event { NONE, SPEECH_START, SPEECH_END }

    private val minSpeechChunks = (minSpeechMs + chunkMs - 1) / chunkMs
    private val minSilenceChunks = (minSilenceMs + chunkMs - 1) / chunkMs

    private var inSpeech = false
    private var speechChunks = 0
    private var silenceChunks = 0

    var completedSegments = 0; private set
    val isSpeaking: Boolean get() = inSpeech

    fun update(prob: Float): Event {
        return if (!inSpeech) {
            if (prob >= startThreshold) {
                inSpeech = true
                speechChunks = 1
                silenceChunks = 0
                Event.SPEECH_START
            } else {
                Event.NONE
            }
        } else {
            speechChunks++
            if (prob < endThreshold) {
                silenceChunks++
                if (silenceChunks >= minSilenceChunks) {
                    val length = speechChunks - silenceChunks
                    inSpeech = false
                    val emitted = length >= minSpeechChunks
                    if (emitted) completedSegments++
                    speechChunks = 0
                    silenceChunks = 0
                    if (emitted) Event.SPEECH_END else Event.NONE
                } else {
                    Event.NONE
                }
            } else {
                silenceChunks = 0
                Event.NONE
            }
        }
    }

    fun reset() {
        inSpeech = false
        speechChunks = 0
        silenceChunks = 0
        completedSegments = 0
    }
}

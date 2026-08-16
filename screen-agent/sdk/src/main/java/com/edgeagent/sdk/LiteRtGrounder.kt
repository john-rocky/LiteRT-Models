package com.edgeagent.sdk

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.File
import kotlin.coroutines.resume

/**
 * LFM2.5-VL behind the [Grounding] contract: a screen view plus an instruction
 * in, a list of [Grounded] out.
 *
 * Two findings from the chat-app build are baked into [create] and are not
 * preferences:
 *  - `audioBackend` must be left unset for bundles with no audio encoder, or
 *    conversation creation fails with NOT_FOUND on TF_LITE_AUDIO_ENCODER_HW;
 *  - the SigLIP2 vision encoder does not compile on the Pixel GPU delegate
 *    while the text graph delegates fully to OpenCL, so text-GPU + vision-CPU
 *    is a dedicated stage rather than a fallback nobody expects to hit.
 *
 * The bundle must be one where the vision adapter consumes pooled tokens (see
 * FINDINGS.md). Against a stock bundle this class still runs and still returns
 * plausible JSON — it is simply wrong below the top quarter of the screen.
 */
class LiteRtGrounder private constructor(
    private val engine: Engine,
    val effectiveBackend: String,
) : Grounder {

    /**
     * Runs one grounding turn on a fresh conversation. Each screen is judged on
     * its own, so carrying history would only grow the KV cache and let an
     * earlier screen bias the answer.
     */
    override suspend fun locate(view: Framing.View, instruction: String): Grounder.Located =
        ground(view, instruction, raw = false)

    override suspend fun ask(view: Framing.View, question: String): String =
        ground(view, question, raw = true).raw

    suspend fun ground(
        view: Framing.View,
        instruction: String,
        /**
         * Ask mode: send the instruction alone, with no grounding system
         * prompt, and just report what came back. This is how the visibility
         * rulers get run on-device ("list every number you can see") — a check
         * that does not depend on the model being any good at coordinates.
         */
        raw: Boolean = false,
    ): Grounder.Located {
        val conversation = try {
            engine.createConversation(GREEDY)
        } catch (t: Throwable) {
            Log.e(TAG, "createConversation failed", t)
            return Grounder.Located(emptyList(), "", false)
        }
        return try {
            val prompt = if (raw) instruction else Grounding.promptFor(instruction)
            val parts = listOf(
                // Image before text, matching what `litert-lm run --attachment`
                // does ("attachments are placed before the first user text
                // prompt"). The desktop runs that ground correctly all use that
                // order; putting the text first is the one input difference the
                // Mac control never had.
                Content.ImageFile(view.file.absolutePath),
                Content.Text(prompt),
            )
            val reply = generate(conversation, Contents.of(parts))
                ?: return Grounder.Located(emptyList(), "", false)
            // The result line is one glance; the full reply goes to logcat so a
            // wrong answer can be read back verbatim.
            Log.i(TAG, "reply: $reply")
            Grounder.Located(if (raw) emptyList() else Grounding.parse(reply), reply, true)
        } finally {
            runCatching { conversation.close() }
        }
    }

    private suspend fun generate(conversation: Conversation, contents: Contents): String? =
        suspendCancellableCoroutine { cont ->
            val accumulated = StringBuilder()
            try {
                conversation.sendMessageAsync(contents, object : MessageCallback {
                    override fun onMessage(message: Message) {
                        message.contents.contents
                            .filterIsInstance<Content.Text>()
                            .forEach { accumulated.append(it.text) }
                    }

                    override fun onDone() {
                        if (cont.isActive) cont.resume(accumulated.toString())
                    }

                    override fun onError(t: Throwable) {
                        Log.e(TAG, "generation failed", t)
                        if (cont.isActive) cont.resume(null)
                    }
                })
            } catch (t: Throwable) {
                Log.e(TAG, "sendMessageAsync threw", t)
                if (cont.isActive) cont.resume(null)
            }
            cont.invokeOnCancellation { runCatching { conversation.cancelProcess() } }
        }

    fun close() {
        runCatching { engine.close() }
    }

    companion object {
        private const val TAG = "LiteRtGrounder"

        /**
         * Greedy decoding, pinned.
         *
         * A grounding answer is three digits; one sampled token turns y=551 into
         * y=100 or collapses the array to []. Prose hides this — there are many
         * acceptable wordings — so leaving the sampler at its default looks fine
         * right up until the output has to be a number. Never rely on the
         * default here.
         */
        private val GREEDY = ConversationConfig(
            samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0, seed = 0)
        )

        fun create(model: File, requestedBackend: String, cacheDir: File): LiteRtGrounder? {
            // Stage 1: everything on the requested backend.
            attempt(model, backendOf(requestedBackend), backendOf(requestedBackend), cacheDir)
                ?.let { return LiteRtGrounder(it, requestedBackend) }
            Log.w(TAG, "multi-modal on $requestedBackend failed")

            // Stage 2: text-GPU + vision-CPU — the correct split for this family.
            if (requestedBackend == "gpu") {
                attempt(model, Backend.GPU(), Backend.CPU(), cacheDir)
                    ?.let { return LiteRtGrounder(it, "gpu (vision on cpu)") }
                Log.w(TAG, "text-gpu + vision-cpu failed")
            }

            // Stage 3: everything on CPU.
            attempt(model, Backend.CPU(), Backend.CPU(), cacheDir)
                ?.let { return LiteRtGrounder(it, "cpu") }
            Log.e(TAG, "all backends failed")
            return null
        }

        private fun backendOf(name: String): Backend =
            if (name == "gpu") Backend.GPU() else Backend.CPU()

        private fun attempt(
            model: File,
            backend: Backend,
            visionBackend: Backend,
            cacheDir: File,
        ): Engine? = try {
            val config = EngineConfig(
                modelPath = model.absolutePath,
                backend = backend,
                visionBackend = visionBackend,
                // audioBackend deliberately omitted — see the class comment.
                cacheDir = cacheDir.absolutePath,
            )
            val engine = Engine(config)
            engine.initialize()
            // Prove a conversation can actually be created before committing to
            // this stage: audio/vision section mismatches surface here, not in
            // initialize().
            engine.createConversation().close()
            engine
        } catch (t: Throwable) {
            null
        }
    }
}

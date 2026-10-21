package com.iris3b

import java.io.Closeable
import java.io.File

/**
 * Watches the phone's free memory while an image is made. A daemon thread reads MemAvailable every
 * [PERIOD_MS] ms, keeps the minimum, and trips once it falls below [guardKb]. The pipeline reads
 * [tripped] before each graph and stops there (a compile already running cannot be interrupted), so
 * the app says why instead of being closed by Android's low-memory killer.
 */
class MemoryWatch(val guardKb: Long) : Closeable {

    /** The lowest MemAvailable read so far, in kB (Long.MAX_VALUE before the first reading). */
    @Volatile
    var minKb: Long = Long.MAX_VALUE
        private set

    /** MemAvailable (kB) of the reading that tripped the guard, or null. */
    @Volatile
    var trippedKb: Long? = null
        private set

    val tripped: Boolean
        get() = trippedKb != null

    @Volatile
    private var running = true
    private val thread = Thread({ poll() }, "iris3b-memory").apply { isDaemon = true }

    fun start(): MemoryWatch = apply { thread.start() }

    private fun poll() {
        while (running) {
            val kb = availableKb()
            if (kb != null) {
                if (kb < minKb) minKb = kb
                if (kb < guardKb && trippedKb == null) trippedKb = kb
            }
            try {
                Thread.sleep(PERIOD_MS)
            } catch (interrupted: InterruptedException) {
                return
            }
        }
    }

    override fun close() {
        running = false
        thread.interrupt()
    }

    companion object {
        /**
         * The default guard. On a Pixel 8a (8 GB) the FP32 compile of each 0.9 GB DiT graph takes
         * MemAvailable from about 4.0 GB down to 0.3-0.5 GB; the measurement harness ran those graphs
         * at this floor.
         */
        const val DEFAULT_GUARD_KB = 200_000L
        const val PERIOD_MS = 200L

        /** /proc/meminfo's MemAvailable in kB, or null when it cannot be read. */
        fun availableKb(): Long? = runCatching {
            File("/proc/meminfo").useLines { lines ->
                lines.first { it.startsWith("MemAvailable:") }.split(Regex("\\s+"))[1].toLong()
            }
        }.getOrNull()

        /** MemAvailable in bytes, for the engine's prefetch rule (unlimited when unreadable). */
        fun availableBytes(): Long = availableKb()?.let { it * 1024 } ?: Long.MAX_VALUE

        /** This process's VmHWM / VmRSS / VmSwap and the phone's MemAvailable, in kB. */
        fun snapshot(): Map<String, Long> {
            val out = LinkedHashMap<String, Long>()
            runCatching {
                for (line in File("/proc/self/status").readLines()) {
                    val parts = line.split(Regex("\\s+"))
                    if (parts.size >= 2 && parts[0] in STATUS_KEYS) {
                        out[parts[0].removeSuffix(":") + "_kb"] = parts[1].toLong()
                    }
                }
            }
            availableKb()?.let { out["MemAvailable_kb"] = it }
            return out
        }

        private val STATUS_KEYS = setOf("VmHWM:", "VmRSS:", "VmSwap:")
    }
}

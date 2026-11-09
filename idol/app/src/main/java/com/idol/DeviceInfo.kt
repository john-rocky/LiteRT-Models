package com.idol

import android.content.Context
import android.os.Build
import android.os.PowerManager
import java.io.File

/** Device facts for the status line and the run JSON. */
object DeviceInfo {
    /** `MemAvailable` of `/proc/meminfo` in kB (what the memory guard compares with), or -1. */
    fun memAvailableKb(): Long = meminfo("MemAvailable:")

    fun memTotalKb(): Long = meminfo("MemTotal:")

    private fun meminfo(key: String): Long = runCatching {
        File("/proc/meminfo").useLines { lines ->
            lines.first { it.startsWith(key) }.trim().split(Regex("\\s+"))[1].toLong()
        }
    }.getOrDefault(-1L)

    /** VmHWM / VmRSS / VmSwap of this process in kB (`/proc/self/status`). */
    fun selfStatusKb(): Map<String, Long> = runCatching {
        File("/proc/self/status").readLines().mapNotNull { line ->
            val key = line.substringBefore(':')
            if (key in setOf("VmHWM", "VmRSS", "VmSwap")) key to line.substringAfter(':').trim().split(' ')[0].toLong()
            else null
        }.toMap()
    }.getOrDefault(emptyMap())

    /**
     * Resets this process's VmHWM to its current RSS (`echo 5 > /proc/self/clear_refs`), so that VmHWM read after a run
     * is that run's peak. Returns false when the system does not allow it (the VmHWM is then the process's peak).
     */
    fun resetPeakRss(): Boolean = runCatching {
        File("/proc/self/clear_refs").writeText("5")
        true
    }.getOrDefault(false)

    /** The system's GPU pages (`/sys/class/kgsl/kgsl/page_alloc`, bytes), or -1 when the file cannot be read. */
    fun gpuPageAllocBytes(): Long = runCatching {
        File("/sys/class/kgsl/kgsl/page_alloc").readText().trim().split(Regex("\\s+"))[0].toLong()
    }.getOrDefault(-1L)

    /** "Galaxy S26 (SM-S942Q)" for the phones this sample was measured on, else `Build.MODEL`. */
    fun displayName(): String {
        val model = Build.MODEL
        val market = MARKET_NAMES.entries.firstOrNull { model.startsWith(it.key) }?.value
        return if (market == null || market == model) model else "$market ($model)"
    }

    fun thermalStatus(context: Context): Int =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            context.getSystemService(PowerManager::class.java).currentThermalStatus
        } else {
            -1
        }

    /** The cpuset of this process (`/top-app` while the app is in front). */
    fun cpuset(): String = runCatching {
        File("/proc/self/cgroup").readLines().firstOrNull { ":cpuset:" in it } ?: "none"
    }.getOrDefault("unreadable")

    private val MARKET_NAMES = mapOf("SM-S942" to "Galaxy S26")
}

/**
 * Samples `MemAvailable` every [periodMs] while one call runs and keeps the lowest reading (the memory guard's view
 * of the call's peak). [stop] returns it, or -1 when `/proc/meminfo` cannot be read.
 */
class MemAvailableWatch(private val periodMs: Long) {
    @Volatile private var lowKb = -1L
    @Volatile private var running = false
    private var thread: Thread? = null

    private fun sample() {
        val v = DeviceInfo.memAvailableKb()
        if (v >= 0 && (lowKb < 0 || v < lowKb)) lowKb = v
    }

    fun start() {
        sample()
        running = true
        thread = Thread({
            while (running) {
                sample()
                try {
                    Thread.sleep(periodMs)
                } catch (_: InterruptedException) {
                    break
                }
            }
        }, "IdolMemWatch").apply { isDaemon = true; start() }
    }

    fun stop(): Long {
        running = false
        thread?.interrupt()
        thread?.join(500)
        thread = null
        sample()
        return lowKb
    }
}

/**
 * Samples the system's GPU pages every [periodMs] while a run is in progress and keeps the maximum (CONTRACT.md
 * line (e)). Best effort: when the app cannot read the file, [maxBytes] stays -1.
 */
class GpuPageSampler(private val periodMs: Long = 100) {
    @Volatile var maxBytes = -1L
        private set
    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        maxBytes = DeviceInfo.gpuPageAllocBytes()
        if (maxBytes < 0) return
        running = true
        thread = Thread({
            while (running) {
                val v = DeviceInfo.gpuPageAllocBytes()
                if (v > maxBytes) maxBytes = v
                try {
                    Thread.sleep(periodMs)
                } catch (_: InterruptedException) {
                    break
                }
            }
        }, "IdolGpuPages").apply { isDaemon = true; start() }
    }

    fun stop(): Long {
        running = false
        thread?.interrupt()
        thread?.join(500)
        thread = null
        val v = DeviceInfo.gpuPageAllocBytes()
        if (v > maxBytes) maxBytes = v
        return maxBytes
    }
}

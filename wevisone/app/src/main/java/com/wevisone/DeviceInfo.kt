package com.wevisone

import android.content.Context
import android.os.Build
import android.os.PowerManager
import java.io.File

/** Device facts for the status line and the run JSON. */
object DeviceInfo {
    /** `MemAvailable` of `/proc/meminfo` in kB (what the memory checks compare with), or -1. */
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

    private val MARKET_NAMES = mapOf("SM-S942" to "Galaxy S26", "akita" to "Pixel 8a", "Pixel 8a" to "Pixel 8a")
}

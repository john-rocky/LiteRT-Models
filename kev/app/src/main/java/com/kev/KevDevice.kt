package com.kev

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import java.io.File

/** Device facts the diagnostic reports and the demo run JSON record. */
object KevDevice {
  /** Reported instead of a thermal status below API 29, which has none. */
  const val THERMAL_STATUS_UNAVAILABLE = -1

  /** `PowerManager.getCurrentThermalStatus()` (0 = none … 6 = shutdown). */
  fun thermalStatus(context: Context): Int =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      context.getSystemService(PowerManager::class.java).currentThermalStatus
    } else {
      THERMAL_STATUS_UNAVAILABLE
    }

  /** Battery temperature in °C from the sticky battery broadcast, or null. */
  fun batteryTemperature(context: Context): Double? {
    val battery =
      context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return null
    val tenths = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
    return if (tenths == Int.MIN_VALUE) null else tenths / TENTHS
  }

  /**
   * The cpuset line of `/proc/self/cgroup` (`…:cpuset:/top-app` while the app is in the
   * foreground), or every line when there is none.
   */
  fun cgroup(): String = runCatching {
    val lines = File("/proc/self/cgroup").readLines().filter { it.isNotBlank() }
    lines.firstOrNull { ":cpuset:" in it } ?: lines.joinToString("; ")
  }
    .getOrElse { "unreadable: ${it.message}" }

  /**
   * `Build.MODEL`, with the market name in front for the models in [MARKET_NAMES]: "Galaxy S26
   * (SM-S942Q)".
   */
  fun displayName(): String = displayName(Build.MODEL)

  fun displayName(model: String): String {
    val marketName = marketNameOrNull(model)
    return if (marketName == null) model else "$marketName ($model)"
  }

  /** The market name for the models in [MARKET_NAMES] ("Galaxy S26"), else `Build.MODEL`. */
  fun marketName(): String = marketNameOrNull(Build.MODEL) ?: Build.MODEL

  private fun marketNameOrNull(model: String): String? =
    MARKET_NAMES.entries.firstOrNull { model.startsWith(it.key) }?.value

  /** `ActivityManager.MemoryInfo.availMem`: the memory available to apps, in bytes. */
  fun availableMemoryBytes(context: Context): Long {
    val info = ActivityManager.MemoryInfo()
    context.getSystemService(ActivityManager::class.java).getMemoryInfo(info)
    return info.availMem
  }

  /** `MemAvailable` of `/proc/meminfo` in kB, or null when it cannot be read. */
  fun procMemAvailableKb(): Long? = runCatching {
    File("/proc/meminfo")
      .readLines()
      .first { it.startsWith("MemAvailable:") }
      .split(Regex("\\s+"))[1]
      .toLong()
  }
    .getOrNull()

  /**
   * Both readings of the free memory at one moment: the app's [availableMemoryBytes] (what the plan
   * and the memory limits use) and the kernel's `MemAvailable`, which differ.
   */
  fun memory(context: Context): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "avail_mem_bytes" to availableMemoryBytes(context),
      "proc_mem_available_kb" to procMemAvailableKb(),
    )

  fun airplaneMode(context: Context): Boolean =
    Settings.Global.getInt(context.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) != 0

  /**
   * The GPU's clock ceiling and temperature from kgsl (Qualcomm Adreno), or null when the files are
   * missing or the app may not read them.
   */
  fun gpuState(): KevGpuState? = runCatching {
    KevGpuState(
      File(KGSL_DIR, "max_clock_mhz").readText().trim().toInt(),
      File(KGSL_DIR, "temp").readText().trim().toInt(),
    )
  }
    .getOrNull()

  /**
   * The CPU frequency policies whose ceiling is under the hardware maximum ("policy6:2668800/
   * 4742400", scaling_max_freq / cpuinfo_max_freq in kHz), empty when none is capped, or null when
   * the files cannot be read.
   */
  fun cpuCaps(): List<String>? = runCatching {
    val policies =
      requireNotNull(File(CPUFREQ_DIR).listFiles { file -> file.name.startsWith("policy") })
    policies
      .sortedBy { it.name }
      .mapNotNull { policy ->
        val ceiling = File(policy, "scaling_max_freq").readText().trim().toLong()
        val hardware = File(policy, "cpuinfo_max_freq").readText().trim().toLong()
        if (ceiling < hardware) "${policy.name}:$ceiling/$hardware" else null
      }
  }
    .getOrNull()

  /** The caps a timed call starts under: CPU policies capped or not, and the thermal status. */
  fun caps(context: Context): KevCaps =
    KevCaps(cpuCaps()?.isNotEmpty(), thermalStatus(context).takeIf { it >= 0 })

  /** Files and bytes under [directory] (LiteRT may keep compiled GPU programs in the cache dir). */
  fun directoryUsage(directory: File): LinkedHashMap<String, Any?> {
    val files = directory.walkTopDown().filter { it.isFile }.toList()
    return linkedMapOf("files" to files.size, "bytes" to files.sumOf { it.length() })
  }

  /** The common header of diagnostic reports. */
  fun header(context: Context): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "device_model" to Build.MODEL,
      "manufacturer" to Build.MANUFACTURER,
      "android_release" to Build.VERSION.RELEASE,
      "android_sdk" to Build.VERSION.SDK_INT,
      "build_fingerprint" to Build.FINGERPRINT,
      "build_type" to BuildConfig.BUILD_TYPE,
      "litert" to KevDecider.LITERT_VERSION,
      "airplane_mode" to airplaneMode(context),
    )

  private const val TENTHS = 10.0
  private const val KGSL_DIR = "/sys/class/kgsl/kgsl-3d0"
  private const val CPUFREQ_DIR = "/sys/devices/system/cpu/cpufreq"

  /** Models this sample was measured on, shown by their market name (key: `Build.MODEL` prefix). */
  private val MARKET_NAMES = mapOf("SM-S942" to "Galaxy S26")
}

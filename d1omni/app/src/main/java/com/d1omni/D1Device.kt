package com.d1omni

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import java.io.File

/** The GPU's clock ceiling, clock, thermal power level and temperature from kgsl (Adreno). */
class D1GpuState(
  val maxClockMhz: Int,
  val clockMhz: Int?,
  val thermalPowerLevel: Int?,
  val tempMilliC: Int,
) {
  fun toJson(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "max_clock_mhz" to maxClockMhz,
      "clock_mhz" to clockMhz,
      "thermal_pwrlevel" to thermalPowerLevel,
      "temp_mc" to tempMilliC,
    )
}

/** Device facts the reports record: thermal state, clocks, memory, the cgroup of this process. */
object D1Device {
  /** Reported instead of a thermal status below API 29, which has none. */
  const val THERMAL_STATUS_UNAVAILABLE = -1

  private const val TENTHS = 10.0
  private const val KGSL_DIR = "/sys/class/kgsl/kgsl-3d0"
  private const val CPUFREQ_DIR = "/sys/devices/system/cpu/cpufreq"

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

  /** The cpuset line of `/proc/self/cgroup` (`…:cpuset:/top-app` in the foreground). */
  fun cgroup(): String =
    runCatching {
        val lines = File("/proc/self/cgroup").readLines().filter { it.isNotBlank() }
        lines.firstOrNull { ":cpuset:" in it } ?: lines.joinToString("; ")
      }
      .getOrElse { "unreadable: ${it.message}" }

  /** `ActivityManager.MemoryInfo.availMem`: the memory available to apps, in bytes. */
  fun availableMemoryBytes(context: Context): Long {
    val info = ActivityManager.MemoryInfo()
    context.getSystemService(ActivityManager::class.java).getMemoryInfo(info)
    return info.availMem
  }

  /** `MemAvailable` of `/proc/meminfo` in kB, or null. */
  fun procMemAvailableKb(): Long? = procValue("/proc/meminfo", "MemAvailable:")

  /** This process's `VmHWM` / `VmRSS` / `VmSwap` from `/proc/self/status`, in kB. */
  fun processMemoryKb(): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "vm_hwm_kb" to procValue("/proc/self/status", "VmHWM:"),
      "vm_rss_kb" to procValue("/proc/self/status", "VmRSS:"),
      "vm_swap_kb" to procValue("/proc/self/status", "VmSwap:"),
    )

  /** Both readings of the free memory at one moment and this process's resident memory. */
  fun memory(context: Context): LinkedHashMap<String, Any?> =
    linkedMapOf<String, Any?>(
        "avail_mem_bytes" to availableMemoryBytes(context),
        "proc_mem_available_kb" to procMemAvailableKb(),
      )
      .apply { putAll(processMemoryKb()) }

  fun airplaneMode(context: Context): Boolean =
    Settings.Global.getInt(context.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) != 0

  /** The market name of the models in [MARKET_NAMES] ("Galaxy S26"), else `Build.MODEL`. */
  fun marketName(model: String = Build.MODEL): String =
    MARKET_NAMES.entries.firstOrNull { model.startsWith(it.key) }?.value ?: model

  private val MARKET_NAMES = mapOf("SM-S942" to "Galaxy S26")

  /** kgsl's clock ceiling and temperature (and clock, thermal power level), or null. */
  fun gpuState(): D1GpuState? =
    runCatching {
        D1GpuState(
          File(KGSL_DIR, "max_clock_mhz").readText().trim().toInt(),
          runCatching { File(KGSL_DIR, "clock_mhz").readText().trim().toInt() }.getOrNull(),
          runCatching { File(KGSL_DIR, "thermal_pwrlevel").readText().trim().toInt() }.getOrNull(),
          File(KGSL_DIR, "temp").readText().trim().toInt(),
        )
      }
      .getOrNull()

  /**
   * The CPU frequency policies whose ceiling is under the hardware maximum
   * ("policy6:2668800/4742400", scaling_max_freq / cpuinfo_max_freq in kHz), empty when none is
   * capped, or null when the files cannot be read.
   */
  fun cpuCaps(): List<String>? =
    runCatching {
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

  /** The device and runtime facts every report starts with. */
  fun header(context: Context): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "device_model" to Build.MODEL,
      "manufacturer" to Build.MANUFACTURER,
      "android_release" to Build.VERSION.RELEASE,
      "android_sdk" to Build.VERSION.SDK_INT,
      "build_fingerprint" to Build.FINGERPRINT,
      "build_type" to BuildConfig.BUILD_TYPE,
      "litert" to D1Decider.LITERT_VERSION,
      "airplane_mode" to airplaneMode(context),
    )

  /** The phone's state at one moment: thermal status, battery, GPU, CPU caps, memory, cgroup. */
  fun state(context: Context): LinkedHashMap<String, Any?> =
    linkedMapOf(
      "wall_ms" to System.currentTimeMillis(),
      "thermal_status" to thermalStatus(context),
      "battery_temperature_c" to batteryTemperature(context),
      "gpu" to gpuState()?.toJson(),
      "cpu_caps" to cpuCaps(),
      "memory" to memory(context),
      "cgroup" to cgroup(),
    )

  private fun procValue(path: String, key: String): Long? =
    runCatching {
        File(path).readLines().first { it.startsWith(key) }.split(Regex("\\s+"))[1].toLong()
      }
      .getOrNull()
}

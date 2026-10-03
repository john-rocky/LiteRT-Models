package com.kev

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

  fun airplaneMode(context: Context): Boolean =
    Settings.Global.getInt(context.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) != 0

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

  /** Models this sample was measured on, shown by their market name (key: `Build.MODEL` prefix). */
  private val MARKET_NAMES = mapOf("SM-S942" to "Galaxy S26")
}

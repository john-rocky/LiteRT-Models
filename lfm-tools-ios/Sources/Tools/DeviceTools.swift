// Read-only facts about the phone. Nothing here asks the user for permission,
// so these are the tools to try first when checking whether a model is calling
// anything at all.
import Foundation
import FoundationModels
import Network
import UIKit

/// Tools that take no input still need a `Generable` argument type; an empty
/// object is the smallest schema the router can be asked to fill.
@available(iOS 27.0, *)
@Generable struct NoArguments {}

@available(iOS 27.0, *)
struct CurrentTimeTool: Tool {
  let name = "get_current_time"
  let description = "What time it is now, and the date."

  @Generable struct Arguments {
    @Guide(description: "IANA time zone such as Asia/Tokyo. Omit for the device's own zone.")
    var timeZone: String?
  }

  func call(arguments: Arguments) async throws -> String {
    let formatter = DateFormatter()
    formatter.dateStyle = .full
    formatter.timeStyle = .medium
    if let name = arguments.timeZone, let zone = TimeZone(identifier: name) {
      formatter.timeZone = zone
    }
    ArtifactBox.shared.post(.clock(date: Date(), zone: formatter.timeZone.identifier))
    return "\(formatter.string(from: Date())) (\(formatter.timeZone.identifier))"
  }
}

@available(iOS 27.0, *)
struct DeviceInfoTool: Tool {
  let name = "get_device_info"
  let description = "Model and iOS version."

  func call(arguments: NoArguments) async throws -> String {
    await MainActor.run {
      let device = UIDevice.current
      return "\(device.name) — \(device.model), \(device.systemName) \(device.systemVersion)"
    }
  }
}

@available(iOS 27.0, *)
struct BatteryTool: Tool {
  let name = "get_battery"
  let description = "Battery level and charging state."

  func call(arguments: NoArguments) async throws -> String {
    await MainActor.run {
      let device = UIDevice.current
      // Monitoring is off by default and `batteryLevel` returns -1 until it is on.
      device.isBatteryMonitoringEnabled = true
      let percent = Int((device.batteryLevel * 100).rounded())
      let state: String
      switch device.batteryState {
      case .charging: state = "charging"
      case .full: state = "full"
      case .unplugged: state = "on battery"
      default: state = "unknown"
      }
      if percent >= 0 {
        ArtifactBox.shared.post(
          .gauge(title: "Battery", value: Double(percent), unit: "%", caption: state))
      }
      return percent < 0 ? "battery level unavailable (\(state))" : "\(percent)%, \(state)"
    }
  }
}

@available(iOS 27.0, *)
struct StorageTool: Tool {
  let name = "get_storage"
  let description = "Free and total storage."

  func call(arguments: NoArguments) async throws -> String {
    let url = URL(fileURLWithPath: NSHomeDirectory())
    let values = try url.resourceValues(forKeys: [
      .volumeAvailableCapacityForImportantUsageKey, .volumeTotalCapacityKey,
    ])
    let free = values.volumeAvailableCapacityForImportantUsage ?? 0
    let total = Int64(values.volumeTotalCapacity ?? 0)
    let fmt = ByteCountFormatter()
    fmt.countStyle = .file
    return "\(fmt.string(fromByteCount: free)) free of \(fmt.string(fromByteCount: total))"
  }
}

@available(iOS 27.0, *)
struct PowerStateTool: Tool {
  let name = "get_power_state"
  let description = "Thermal state, Low Power Mode, uptime."

  func call(arguments: NoArguments) async throws -> String {
    let info = ProcessInfo.processInfo
    let thermal: String
    switch info.thermalState {
    case .nominal: thermal = "nominal"
    case .fair: thermal = "fair"
    case .serious: thermal = "serious"
    case .critical: thermal = "critical"
    @unknown default: thermal = "unknown"
    }
    let uptime = Int(info.systemUptime)
    return """
      thermal: \(thermal)
      low power mode: \(info.isLowPowerModeEnabled ? "on" : "off")
      cores: \(info.processorCount), memory: \(info.physicalMemory / 1_073_741_824) GB
      uptime: \(uptime / 3600)h \(uptime % 3600 / 60)m
      """
  }
}

@available(iOS 27.0, *)
struct LocaleTool: Tool {
  let name = "get_locale"
  let description = "Language, region, currency."

  func call(arguments: NoArguments) async throws -> String {
    let locale = Locale.current
    let language = locale.language.languageCode?.identifier ?? "?"
    let region = locale.region?.identifier ?? "?"
    let currency = locale.currency?.identifier ?? "?"
    return "language \(language), region \(region), currency \(currency), "
      + "calendar \(locale.calendar.identifier), 24-hour: \(locale.hourCycle == .zeroToTwentyThree)"
  }
}

@available(iOS 27.0, *)
struct NetworkTool: Tool {
  let name = "get_network_status"
  let description = "Online state and interface."

  func call(arguments: NoArguments) async throws -> String {
    let monitor = NWPathMonitor()
    defer { monitor.cancel() }
    let queue = DispatchQueue(label: "lfm.tools.network")
    // The monitor delivers the current path once started; one update is enough.
    let path: NWPath = await withCheckedContinuation { continuation in
      let once = OnceBox()
      monitor.pathUpdateHandler = { path in
        if once.claim() { continuation.resume(returning: path) }
      }
      monitor.start(queue: queue)
    }
    guard path.status == .satisfied else { return "offline" }
    let interface: String
    if path.usesInterfaceType(.wifi) {
      interface = "Wi-Fi"
    } else if path.usesInterfaceType(.cellular) {
      interface = "cellular"
    } else if path.usesInterfaceType(.wiredEthernet) {
      interface = "wired"
    } else {
      interface = "unknown interface"
    }
    return "online over \(interface)\(path.isExpensive ? " (expensive)" : "")"
  }
}

/// Guards a continuation that a callback may fire more than once.
final class OnceBox: @unchecked Sendable {
  private let lock = NSLock()
  private var used = false
  func claim() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    if used { return false }
    used = true
    return true
  }
}

@available(iOS 27.0, *)
struct CalculateTool: Tool {
  let name = "calculate"
  let description = "Work out an arithmetic expression."

  @Generable struct Arguments {
    @Guide(description: "Arithmetic only: digits, + - * / and parentheses.")
    var expression: String
  }

  func call(arguments: Arguments) async throws -> String {
    // NSExpression parses far more than arithmetic — including FUNCTION(), which
    // reaches arbitrary selectors. Whitelist the characters before it is handed
    // anything a model wrote.
    let allowed = CharacterSet(charactersIn: "0123456789.+-*/() ")
    guard arguments.expression.unicodeScalars.allSatisfy({ allowed.contains($0) }),
      !arguments.expression.isEmpty
    else { return "refused: arithmetic only (digits, + - * / and parentheses)" }
    guard let value = NSExpression(format: arguments.expression).expressionValue(
      with: nil, context: nil) as? NSNumber
    else { return "could not evaluate \(arguments.expression)" }
    ArtifactBox.shared.post(
      .equation(expression: arguments.expression, result: "\(value)"))
    return "\(arguments.expression) = \(value)"
  }
}

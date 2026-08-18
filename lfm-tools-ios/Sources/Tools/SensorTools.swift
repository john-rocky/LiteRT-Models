// The sensors and the settings the phone will tell you about without a
// capability or a paid team. Motion needs one Info.plist string; the rest need
// nothing at all.
import AVFoundation
import CoreLocation
import CoreMotion
import Foundation
import FoundationModels
import UIKit

@available(iOS 27.0, *)
struct HeadingTool: Tool {
  let name = "get_heading"
  let description = "Which direction the phone is pointing: a compass bearing in degrees."

  func call(arguments: NoArguments) async throws -> String {
    try await withDeadline(8, "the compass") {
      let heading = try await MotionBox.shared.heading()
      ArtifactBox.shared.post(.compass(degrees: heading))
      let points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
      let point = points[Int((heading / 45).rounded()) % 8]
      return String(format: "%.0f° (%@)", heading, point)
    }
  }
}

@available(iOS 27.0, *)
struct AltitudeTool: Tool {
  let name = "get_air_pressure"
  let description = "Air pressure and height change."

  func call(arguments: NoArguments) async throws -> String {
    guard CMAltimeter.isRelativeAltitudeAvailable() else {
      return "this device has no barometer"
    }
    return try await withDeadline(8, "the barometer") { try await MotionBox.shared.altitude() }
  }
}

@available(iOS 27.0, *)
struct StepsTool: Tool {
  let name = "get_steps_today"
  let description = "How many steps the user has walked today, and how far."

  func call(arguments: NoArguments) async throws -> String {
    guard CMPedometer.isStepCountingAvailable() else { return "step counting unavailable" }
    return try await withDeadline(8, "the pedometer") { try await MotionBox.shared.stepsToday() }
  }
}

@available(iOS 27.0, *)
struct OrientationTool: Tool {
  let name = "get_orientation"
  let description = "How the phone is being held: flat, upright or on its side."

  func call(arguments: NoArguments) async throws -> String {
    await MainActor.run {
      switch UIDevice.current.orientation {
      case .portrait: return "portrait"
      case .portraitUpsideDown: return "upside down"
      case .landscapeLeft: return "landscape (left)"
      case .landscapeRight: return "landscape (right)"
      case .faceUp: return "face up"
      case .faceDown: return "face down"
      default: return "unknown"
      }
    }
  }
}

@available(iOS 27.0, *)
struct AudioRouteTool: Tool {
  let name = "get_audio_route"
  let description = "Where audio is playing."

  func call(arguments: NoArguments) async throws -> String {
    let route = AVAudioSession.sharedInstance().currentRoute
    let outputs = route.outputs.map { "\($0.portName) (\($0.portType.rawValue))" }
    return outputs.isEmpty ? "no output route" : outputs.joined(separator: ", ")
  }
}

@available(iOS 27.0, *)
struct ScreenInfoTool: Tool {
  let name = "get_screen_info"
  let description = "Screen size and dark mode."

  func call(arguments: NoArguments) async throws -> String {
    await MainActor.run {
      guard let screen = activeScreen() else { return "no screen" }
      let bounds = screen.bounds
      let dark = UITraitCollection.current.userInterfaceStyle == .dark
      return "\(Int(bounds.width))×\(Int(bounds.height)) pt @\(Int(screen.scale))x, "
        + "up to \(screen.maximumFramesPerSecond) Hz, \(dark ? "dark" : "light") mode"
    }
  }
}

@available(iOS 27.0, *)
struct AccessibilitySettingsTool: Tool {
  let name = "get_accessibility_settings"
  let description = "Accessibility settings in use."

  func call(arguments: NoArguments) async throws -> String {
    await MainActor.run {
      var on: [String] = []
      if UIAccessibility.isVoiceOverRunning { on.append("VoiceOver") }
      if UIAccessibility.isReduceMotionEnabled { on.append("Reduce Motion") }
      if UIAccessibility.isReduceTransparencyEnabled { on.append("Reduce Transparency") }
      if UIAccessibility.isBoldTextEnabled { on.append("Bold Text") }
      if UIAccessibility.isDarkerSystemColorsEnabled { on.append("Increase Contrast") }
      if UIAccessibility.isSwitchControlRunning { on.append("Switch Control") }
      let size = UIApplication.shared.preferredContentSizeCategory.rawValue
        .replacingOccurrences(of: "UICTContentSizeCategory", with: "")
      return (on.isEmpty ? "none enabled" : on.joined(separator: ", ")) + "; text size \(size)"
    }
  }
}

/// CoreLocation heading and CoreMotion altitude both arrive by callback and both
/// die with their manager, so one long-lived owner serves every call.
@available(iOS 27.0, *)
final class MotionBox: NSObject, CLLocationManagerDelegate, @unchecked Sendable {
  static let shared = MotionBox()

  enum Failure: LocalizedError {
    case unavailable(String)
    var errorDescription: String? {
      if case .unavailable(let why) = self { return why }
      return nil
    }
  }

  private let locations = CLLocationManager()
  private let altimeter = CMAltimeter()
  // Held for the lifetime of the app on purpose: a CMPedometer created inside
  // the call is deallocated the moment the call returns, and its completion
  // handler then never fires — the tool hangs forever rather than failing.
  private let pedometer = CMPedometer()
  private var headingWaiters: [CheckedContinuation<CLLocationDirection, Error>] = []
  private let lock = NSLock()

  override private init() {
    super.init()
    locations.delegate = self
  }

  func heading() async throws -> CLLocationDirection {
    guard CLLocationManager.headingAvailable() else {
      throw Failure.unavailable("this device has no compass")
    }
    return try await withCheckedThrowingContinuation { continuation in
      lock.lock()
      headingWaiters.append(continuation)
      let first = headingWaiters.count == 1
      lock.unlock()
      if first { locations.startUpdatingHeading() }
    }
  }

  func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
    guard newHeading.headingAccuracy >= 0 else { return }
    manager.stopUpdatingHeading()
    lock.lock()
    let waiting = headingWaiters
    headingWaiters = []
    lock.unlock()
    for continuation in waiting { continuation.resume(returning: newHeading.magneticHeading) }
  }

  private let motion = CMMotionManager()
  private let activityManager = CMMotionActivityManager()

  func attitude() async throws -> String {
    guard motion.isDeviceMotionAvailable else { return "no gyroscope on this device" }
    motion.deviceMotionUpdateInterval = 0.1
    return try await withCheckedThrowingContinuation { continuation in
      let once = OnceBox()
      motion.startDeviceMotionUpdates(to: .main) { [weak self] data, error in
        guard once.claim() else { return }
        self?.motion.stopDeviceMotionUpdates()
        guard let a = data?.attitude else {
          continuation.resume(throwing: error ?? Failure.unavailable("no motion reading"))
          return
        }
        let degrees = 180 / Double.pi
        continuation.resume(
          returning: String(
            format: "pitch %.0f°, roll %.0f°, yaw %.0f°", a.pitch * degrees, a.roll * degrees,
            a.yaw * degrees))
      }
    }
  }

  func activity() async throws -> String {
    try await withCheckedThrowingContinuation { continuation in
      let once = OnceBox()
      activityManager.startActivityUpdates(to: .main) { [weak self] activity in
        guard once.claim() else { return }
        self?.activityManager.stopActivityUpdates()
        guard let activity else {
          continuation.resume(throwing: Failure.unavailable("no activity reading"))
          return
        }
        var kinds: [String] = []
        if activity.stationary { kinds.append("still") }
        if activity.walking { kinds.append("walking") }
        if activity.running { kinds.append("running") }
        if activity.cycling { kinds.append("cycling") }
        if activity.automotive { kinds.append("in a vehicle") }
        let confidence = ["low", "medium", "high"][min(2, activity.confidence.rawValue)]
        let kind = kinds.isEmpty ? "unknown" : kinds.joined(separator: " / ")
        ArtifactBox.shared.post(.activity(kind: kind, confidence: confidence))
        continuation.resume(returning: "\(kind) (\(confidence) confidence)")
      }
    }
  }

  /// One pedometer query per day, because CMPedometer has no grouping — and the
  /// count has to leave the callback as a number, not as CMPedometerData.
  func stepsByDay(_ days: Int) async throws -> [DayStepCount] {
    var out: [DayStepCount] = []
    let calendar = Calendar.current
    for back in stride(from: days - 1, through: 0, by: -1) {
      guard let start = calendar.date(byAdding: .day, value: -back, to: calendar.startOfDay(for: Date())),
        let end = calendar.date(byAdding: .day, value: 1, to: start)
      else { continue }
      let steps: Int = await withCheckedContinuation { continuation in
        let once = OnceBox()
        pedometer.queryPedometerData(from: start, to: min(end, Date())) { data, _ in
          guard once.claim() else { return }
          continuation.resume(returning: data?.numberOfSteps.intValue ?? 0)
        }
      }
      out.append(DayStepCount(day: start, steps: steps))
    }
    return out
  }

  func stepsToday() async throws -> String {
    let start = Calendar.current.startOfDay(for: Date())
    return try await withCheckedThrowingContinuation { continuation in
      let once = OnceBox()
      pedometer.queryPedometerData(from: start, to: Date()) { data, error in
        guard once.claim() else { return }
        if let data {
          let distance = data.distance.map { " over \(Int($0.doubleValue)) m" } ?? ""
          ArtifactBox.shared.post(
            .gauge(
              title: "\(data.numberOfSteps) steps",
              value: min(100, data.numberOfSteps.doubleValue / 80),  // 8000 = a full ring
              unit: "today", caption: distance.isEmpty ? "walked today" : distance))
          continuation.resume(returning: "\(data.numberOfSteps) steps today\(distance)")
        } else {
          continuation.resume(throwing: error ?? Failure.unavailable("no step data"))
        }
      }
    }
  }

  /// Returns the formatted reading, not the sample: `CMAltitudeData` is not
  /// `Sendable` and cannot cross the continuation.
  func altitude() async throws -> String {
    try await withCheckedThrowingContinuation { continuation in
      let once = OnceBox()
      altimeter.startRelativeAltitudeUpdates(to: .main) { [weak self] data, error in
        guard once.claim() else { return }
        self?.altimeter.stopRelativeAltitudeUpdates()
        if let data {
          continuation.resume(
            returning: String(
              format: "%.1f kPa, %+.1f m since launch", data.pressure.doubleValue,
              data.relativeAltitude.doubleValue))
        } else {
          continuation.resume(
            throwing: error ?? Failure.unavailable("no barometer reading"))
        }
      }
    }
  }
}

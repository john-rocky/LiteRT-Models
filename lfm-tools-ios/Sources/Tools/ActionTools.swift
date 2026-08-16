// Tools that change something. These are the ones worth watching in a demo:
// the model does not describe an action, it performs it.
import AVFoundation
import Foundation
import FoundationModels
import UIKit
import UserNotifications

@available(iOS 27.0, *)
struct BrightnessTool: Tool {
  let name = "set_brightness"
  let description = "Set the screen brightness."

  @Generable struct Arguments {
    @Guide(description: "Brightness from 0 (dimmest) to 100 (brightest).")
    var percent: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let clamped = min(100, max(0, arguments.percent))
    guard let screen = await activeScreen() else { return "no screen to dim" }
    await MainActor.run { screen.brightness = CGFloat(clamped) / 100 }
    return "brightness set to \(clamped)%"
  }
}

/// `UIScreen.main` is deprecated as of iOS 26; the screen has to come from the
/// scene the app is actually on.
@available(iOS 27.0, *)
@MainActor
func activeScreen() -> UIScreen? {
  UIApplication.shared.connectedScenes
    .compactMap { $0 as? UIWindowScene }
    .first { $0.activationState == .foregroundActive }?.screen
    ?? UIApplication.shared.connectedScenes.compactMap { ($0 as? UIWindowScene)?.screen }.first
}

@available(iOS 27.0, *)
struct ReadBrightnessTool: Tool {
  let name = "get_brightness"
  let description = "Read the current screen brightness."

  func call(arguments: NoArguments) async throws -> String {
    guard let screen = await activeScreen() else { return "no screen" }
    let percent = await MainActor.run { Int((screen.brightness * 100).rounded()) }
    return "\(percent)%"
  }
}

@available(iOS 27.0, *)
struct TorchTool: Tool {
  let name = "set_torch"
  let description = "Turn the rear flashlight on or off."

  @Generable struct Arguments {
    @Guide(description: "true to switch the torch on, false to switch it off.")
    var on: Bool
  }

  func call(arguments: Arguments) async throws -> String {
    guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else {
      return "no torch on this device"
    }
    try device.lockForConfiguration()
    defer { device.unlockForConfiguration() }
    device.torchMode = arguments.on ? .on : .off
    return "torch \(arguments.on ? "on" : "off")"
  }
}

@available(iOS 27.0, *)
struct SpeakTool: Tool {
  let name = "speak"
  let description = "Say something out loud through the device speaker."

  @Generable struct Arguments {
    @Guide(description: "The text to speak.")
    var text: String
    @Guide(description: "BCP-47 voice language such as en-US or ja-JP. Omit for the device's.")
    var language: String?
  }

  func call(arguments: Arguments) async throws -> String {
    let utterance = AVSpeechUtterance(string: arguments.text)
    if let language = arguments.language {
      utterance.voice = AVSpeechSynthesisVoice(language: language)
    }
    await SpeechBox.shared.speak(utterance)
    return "spoke: \(arguments.text)"
  }
}

/// `AVSpeechSynthesizer` stops speaking the moment it is deallocated, so it has
/// to outlive the tool call that started it.
@available(iOS 27.0, *)
@MainActor
final class SpeechBox {
  static let shared = SpeechBox()
  private let synthesizer = AVSpeechSynthesizer()
  func speak(_ utterance: AVSpeechUtterance) { synthesizer.speak(utterance) }
}

@available(iOS 27.0, *)
struct HapticTool: Tool {
  let name = "vibrate"
  let description = "Play a haptic tap the user can feel."

  @Generable struct Arguments {
    @Guide(description: "One of: success, warning, error.")
    var kind: String
  }

  func call(arguments: Arguments) async throws -> String {
    let type: UINotificationFeedbackGenerator.FeedbackType
    switch arguments.kind.lowercased() {
    case "warning": type = .warning
    case "error": type = .error
    default: type = .success
    }
    await MainActor.run { UINotificationFeedbackGenerator().notificationOccurred(type) }
    return "played the \(arguments.kind) haptic"
  }
}

@available(iOS 27.0, *)
struct ReadClipboardTool: Tool {
  let name = "read_clipboard"
  let description = "Read the text currently on the clipboard."

  func call(arguments: NoArguments) async throws -> String {
    // iOS shows the user a paste banner for this; it is not a silent read.
    let text = await MainActor.run { UIPasteboard.general.string }
    return text.map { "clipboard: \($0)" } ?? "the clipboard holds no text"
  }
}

@available(iOS 27.0, *)
struct WriteClipboardTool: Tool {
  let name = "write_clipboard"
  let description = "Put text on the clipboard."

  @Generable struct Arguments {
    @Guide(description: "The text to copy.")
    var text: String
  }

  func call(arguments: Arguments) async throws -> String {
    await MainActor.run { UIPasteboard.general.string = arguments.text }
    return "copied \(arguments.text.count) characters"
  }
}

@available(iOS 27.0, *)
struct OpenURLTool: Tool {
  let name = "open_url"
  let description = "Open a link, or an app URL scheme such as tel: or maps:."

  @Generable struct Arguments {
    @Guide(description: "A full URL including the scheme, e.g. https://example.com")
    var url: String
  }

  func call(arguments: Arguments) async throws -> String {
    guard let url = URL(string: arguments.url) else { return "not a URL: \(arguments.url)" }
    let opened = await UIApplication.shared.open(url)
    return opened ? "opened \(arguments.url)" : "nothing can open \(arguments.url)"
  }
}

@available(iOS 27.0, *)
struct NotificationTool: Tool {
  let name = "schedule_notification"
  let description = "Schedule a local notification a number of seconds from now."

  @Generable struct Arguments {
    @Guide(description: "Notification title.")
    var title: String
    @Guide(description: "Notification body.")
    var body: String
    @Guide(description: "How many seconds from now to fire it. Minimum 1.")
    var seconds: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let center = UNUserNotificationCenter.current()
    let granted = try await center.requestAuthorization(options: [.alert, .sound])
    guard granted else { return "notifications are not permitted" }
    let content = UNMutableNotificationContent()
    content.title = arguments.title
    content.body = arguments.body
    content.sound = .default
    let delay = max(1, arguments.seconds)
    try await center.add(
      UNNotificationRequest(
        identifier: UUID().uuidString, content: content,
        trigger: UNTimeIntervalNotificationTrigger(timeInterval: TimeInterval(delay), repeats: false)
      ))
    return "notification scheduled for \(delay)s from now"
  }
}

@available(iOS 27.0, *)
struct VolumeTool: Tool {
  let name = "get_volume"
  let description = "Read the current output volume."

  func call(arguments: NoArguments) async throws -> String {
    let session = AVAudioSession.sharedInstance()
    try session.setActive(true)
    return "\(Int((session.outputVolume * 100).rounded()))%"
  }
}

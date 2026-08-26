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
    ArtifactBox.shared.post(.brightness(percent: clamped))
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
  let description = "Say something out loud."

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
    ArtifactBox.shared.post(.speaking(text: arguments.text))
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
struct SpeakLastAnswerTool: Tool {
  // Named for the verb the demo speaks: "Speak that out loud." routed to
  // nothing as `read_last_answer_aloud` — the 1.2B matches names, not
  // paraphrases.
  let name = "speak_out_loud"
  let description = "Speak the previous answer out loud."

  func call(arguments: NoArguments) async throws -> String {
    let text = LastAnswer.shared.value
    guard !text.isEmpty else { return "there is nothing to read yet" }
    let utterance = AVSpeechUtterance(string: text)
    await SpeechBox.shared.speak(utterance)
    ArtifactBox.shared.post(.speaking(text: text))
    return "spoke: \(text)"
  }
}

/// The last thing the assistant said, so a tool can read it back without the
/// model having to copy it into an argument.
final class LastAnswer: @unchecked Sendable {
  static let shared = LastAnswer()
  private let lock = NSLock()
  private var stored = ""
  var value: String {
    lock.lock()
    defer { lock.unlock() }
    return stored
  }
  func set(_ text: String) {
    lock.lock()
    stored = text
    lock.unlock()
  }
}

@available(iOS 27.0, *)
struct HapticTool: Tool {
  let name = "vibrate"
  let description = "Play a haptic tap the user can feel."

  @Generable struct Arguments {
    @Guide(description: "Which haptic.", .anyOf(["success", "warning", "error"]))
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
    ArtifactBox.shared.post(.note(text: arguments.text))
    return "copied \(arguments.text.count) characters"
  }
}

@available(iOS 27.0, *)
struct OpenURLTool: Tool {
  let name = "open_url"
  let description = "Open a link or app URL."

  @Generable struct Arguments {
    @Guide(description: "A full URL including the scheme, e.g. https://example.com")
    var url: String
  }

  func call(arguments: Arguments) async throws -> String {
    guard let url = URL(string: arguments.url) else { return "not a URL: \(arguments.url)" }
    // The default `options:` dictionary is not Sendable, so the call has to be
    // made on the main actor rather than awaited across one.
    let opened = await MainActor.run { UIApplication.shared.canOpenURL(url) }
      ? await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
        Task { @MainActor in
          UIApplication.shared.open(url, options: [:]) { continuation.resume(returning: $0) }
        }
      }
      : false
    return opened ? "opened \(arguments.url)" : "nothing can open \(arguments.url)"
  }
}

@available(iOS 27.0, *)
struct OpenMapsTool: Tool {
  let name = "open_in_maps"
  let description = "Open Apple Maps showing a named place."

  @Generable struct Arguments {
    @Guide(description: "Place name, exactly as given.")
    var place: String
  }

  func call(arguments: Arguments) async throws -> String {
    // The URL is built here, not by the model: asked to compose a maps: link a
    // small model invents a plausible website instead, and the demo lands in
    // Safari on a domain nobody chose.
    let query = arguments.place.addingPercentEncoding(
      withAllowedCharacters: .urlQueryAllowed) ?? arguments.place
    guard let url = URL(string: "http://maps.apple.com/?q=\(query)") else {
      return "could not build a map link for \(arguments.place)"
    }
    let opened = await MainActor.run { UIApplication.shared.canOpenURL(url) }
      ? await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
        Task { @MainActor in
          UIApplication.shared.open(url, options: [:]) { continuation.resume(returning: $0) }
        }
      }
      : false
    return opened ? "opened \(arguments.place) in Maps" : "Maps did not open"
  }
}

@available(iOS 27.0, *)
struct NotificationTool: Tool {
  let name = "schedule_notification"
  let description = "Schedule a notification."

  @Generable struct Arguments {
    @Guide(description: "Notification title.")
    var title: String
    @Guide(description: "Notification body.")
    var body: String
    @Guide(description: "How many seconds from now to fire it.", .range(1...3600))
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
    ArtifactBox.shared.post(
      .notice(title: arguments.title, body: arguments.body, seconds: delay))
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

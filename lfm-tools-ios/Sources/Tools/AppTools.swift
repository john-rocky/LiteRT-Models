// Things the agent can leave behind: notifications it can list and cancel, a
// badge, a sound, and a scratch note it can write and read back. The note is
// what gives it memory across turns without any of the model's context.
import AudioToolbox
import Foundation
import FoundationModels
import UIKit
import UserNotifications

@available(iOS 27.0, *)
struct PendingNotificationsTool: Tool {
  let name = "list_pending_notifications"
  let description = "Scheduled notifications."

  func call(arguments: NoArguments) async throws -> String {
    let pending = await UNUserNotificationCenter.current().pendingNotificationRequests()
    guard !pending.isEmpty else { return "nothing scheduled" }
    return pending.prefix(10).map { request in
      let when = (request.trigger as? UNTimeIntervalNotificationTrigger)
        .map { " in \(Int($0.timeInterval))s" } ?? ""
      return "- \(request.content.title)\(when)"
    }.joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
struct CancelNotificationsTool: Tool {
  let name = "cancel_notifications"
  let description = "Cancel every scheduled notification."

  func call(arguments: NoArguments) async throws -> String {
    let center = UNUserNotificationCenter.current()
    let count = await center.pendingNotificationRequests().count
    center.removeAllPendingNotificationRequests()
    return "cancelled \(count)"
  }
}

@available(iOS 27.0, *)
struct BadgeTool: Tool {
  let name = "set_app_badge"
  let description = "Put a number badge on the app icon."

  @Generable struct Arguments {
    @Guide(description: "The number to show. 0 clears the badge.")
    var count: Int
  }

  func call(arguments: Arguments) async throws -> String {
    try await UNUserNotificationCenter.current().setBadgeCount(max(0, arguments.count))
    return arguments.count <= 0 ? "badge cleared" : "badge set to \(arguments.count)"
  }
}

@available(iOS 27.0, *)
struct SystemSoundTool: Tool {
  let name = "play_sound"
  let description = "Play a short sound effect."

  @Generable struct Arguments {
    @Guide(description: "Which sound.", .anyOf(["tink", "tock", "sms", "mail", "anticipate", "complete", "fail"]))
    var sound: String
  }

  func call(arguments: Arguments) async throws -> String {
    // SystemSoundIDs, not files: no bundle resources and no AVAudioPlayer to
    // keep alive past the end of the call.
    let ids: [String: SystemSoundID] = [
      "tink": 1103, "tock": 1104, "sms": 1007, "mail": 1000,
      "anticipate": 1020, "complete": 1021, "fail": 1073,
    ]
    guard let id = ids[arguments.sound.lowercased()] else {
      return "unknown sound \(arguments.sound); try one of \(ids.keys.sorted().joined(separator: ", "))"
    }
    AudioServicesPlaySystemSound(id)
    return "played \(arguments.sound)"
  }
}

@available(iOS 27.0, *)
struct WriteNoteTool: Tool {
  let name = "write_note"
  let description = "Remember something for later."

  @Generable struct Arguments {
    @Guide(description: "What to remember.")
    var text: String
  }

  func call(arguments: Arguments) async throws -> String {
    try NoteStore.append(arguments.text)
    ArtifactBox.shared.post(.note(text: arguments.text))
    return "noted"
  }
}

@available(iOS 27.0, *)
struct ReadNotesTool: Tool {
  let name = "read_notes"
  let description = "Read back notes saved earlier."

  func call(arguments: NoArguments) async throws -> String {
    let notes = NoteStore.all()
    return notes.isEmpty ? "no notes yet" : notes.joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
enum NoteStore {
  private static var url: URL {
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    return documents.appendingPathComponent("agent-notes.txt")
  }

  static func append(_ text: String) throws {
    let stamped = "\(ISO8601DateFormatter().string(from: Date()))  \(text)\n"
    if let handle = try? FileHandle(forWritingTo: url) {
      defer { try? handle.close() }
      try handle.seekToEnd()
      try handle.write(contentsOf: Data(stamped.utf8))
    } else {
      try Data(stamped.utf8).write(to: url)
    }
  }

  static func all() -> [String] {
    guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
    return text.split(separator: "\n").map(String.init)
  }
}

@available(iOS 27.0, *)
struct ListDocumentsTool: Tool {
  let name = "list_files"
  let description = "List saved files."

  func call(arguments: NoArguments) async throws -> String {
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let files = try FileManager.default.contentsOfDirectory(
      at: documents, includingPropertiesForKeys: [.fileSizeKey])
    guard !files.isEmpty else { return "empty" }
    let formatter = ByteCountFormatter()
    return files.prefix(15).map { url in
      let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
      return "- \(url.lastPathComponent) (\(formatter.string(fromByteCount: Int64(size))))"
    }.joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
struct TimeZoneTool: Tool {
  let name = "convert_time"
  let description = "What a time in one place is in another."

  @Generable struct Arguments {
    @Guide(description: "Time as HH:mm, 24-hour.")
    var time: String
    @Guide(description: "IANA zone the time is in, e.g. Asia/Tokyo.")
    var from: String
    @Guide(description: "IANA zone to convert to, e.g. America/Los_Angeles.")
    var to: String
  }

  func call(arguments: Arguments) async throws -> String {
    guard let source = TimeZone(identifier: arguments.from),
      let target = TimeZone(identifier: arguments.to)
    else { return "unknown time zone (use IANA names like Asia/Tokyo)" }
    let parser = DateFormatter()
    parser.dateFormat = "HH:mm"
    parser.timeZone = source
    guard let date = parser.date(from: arguments.time) else {
      return "could not read the time \(arguments.time); use HH:mm"
    }
    let printer = DateFormatter()
    printer.dateFormat = "HH:mm"
    printer.timeZone = target
    let converted = printer.string(from: date)
    ArtifactBox.shared.post(
      .clocks(
        from: arguments.time, fromZone: arguments.from, to: converted, toZone: arguments.to))
    return "\(arguments.time) in \(arguments.from) is \(converted) in \(arguments.to)"
  }
}

// Tools that are several tools.
//
// A small model chains badly (the 1.2B mashes two calls into one; Apple FM
// chains but loses the argument on the way), and some jobs are always the same
// three steps anyway. So the steps move into the tool: one name the model can
// say, one call, and the app does the walking. Each of these reuses the single
// tools' bodies rather than reimplementing them, so a compound and its parts
// can never drift apart.
import Foundation
import FoundationModels
import UIKit

@available(iOS 27.0, *)
struct FocusSessionTool: Tool {
  let name = "start_focus_session"
  let description =
    "Start a focus session: clear pending notifications, dim the screen and set a timer."

  @Generable struct Arguments {
    // Minutes, not seconds: every model asked for seconds did the arithmetic
    // wrong at least once (150 s for 25 min, 600 s for an hour). Optional
    // with a default, because a required field is a question the model may
    // ask instead of calling.
    @Guide(description: "How long to focus, in minutes. Default 25.")
    var minutes: Int?
  }

  func call(arguments: Arguments) async throws -> String {
    let minutes = max(1, arguments.minutes ?? 25)
    let cleared = try await CancelNotificationsTool().call(arguments: NoArguments())
    let dimmed = try await BrightnessTool().call(arguments: .init(percent: 25))
    // The system timer is the nice ending; a notification is the one that
    // cannot fail. A compound that dies on its third step after doing two
    // visible things reads as broken, so the ending degrades instead.
    let ending: String
    do {
      ending = try await TimerTool().call(arguments: .init(seconds: minutes * 60, label: "Focus"))
    } catch {
      ending = try await NotificationTool().call(
        arguments: .init(
          title: "Focus session over", body: "\(minutes) minutes are up.",
          seconds: minutes * 60))
    }
    _ = try await HapticTool().call(arguments: .init(kind: "success"))
    return "focus session started for \(minutes) min — \(cleared), \(dimmed), \(ending)"
  }
}

@available(iOS 27.0, *)
struct MorningBriefingTool: Tool {
  let name = "morning_briefing"
  let description =
    "The morning rundown in one go: time, battery, today's calendar, open reminders, steps."

  func call(arguments: NoArguments) async throws -> String {
    // Each part is its own tool with its own permission and its own failure
    // string; a denied calendar shows up as one line, not a failed briefing.
    async let time = CurrentTimeTool().call(arguments: .init(timeZone: nil))
    async let battery = BatteryTool().call(arguments: NoArguments())
    async let events = ListEventsTool().call(arguments: .init(days: 1))
    async let reminders = ListRemindersTool().call(arguments: NoArguments())
    async let steps = StepsTool().call(arguments: NoArguments())
    return """
      time: \(try await time)
      battery: \(try await battery)
      today: \(try await events)
      reminders: \(try await reminders)
      steps: \(try await steps)
      """
  }
}

@available(iOS 27.0, *)
struct ShareLocationTool: Tool {
  let name = "copy_my_location"
  let description =
    "Put where the user is — place name, address and a map link — on the clipboard to paste anywhere."

  func call(arguments: NoArguments) async throws -> String {
    let location = try await LocationBox.shared.current()
    let place = try await PlaceNameTool().call(arguments: NoArguments())
    let link = String(
      format: "https://maps.apple.com/?ll=%.5f,%.5f", location.coordinate.latitude,
      location.coordinate.longitude)
    let text = "\(place)\n\(link)"
    _ = try await WriteClipboardTool().call(arguments: .init(text: text))
    return "copied to the clipboard: \(text)"
  }
}

@available(iOS 27.0, *)
struct PhotoTextToNoteTool: Tool {
  // The anaphora fix from the recipes, as a tool: "save that as a note" is a
  // reference a 1.2B cannot turn into an argument, so the tool reads the
  // photo itself and keeps the text — no argument to fill.
  let name = "save_photo_text_as_note"
  let description = "Read the text in the latest photo and save it as a note."

  func call(arguments: NoArguments) async throws -> String {
    let text = try await ReadPhotoTextTool().call(arguments: NoArguments())
    guard !text.hasPrefix("no photo"), !text.hasPrefix("no text") else { return text }
    _ = try await WriteNoteTool().call(arguments: .init(text: text))
    return "noted the photo's text: \(text.prefix(120))"
  }
}

// Every tool the demo offers, in one list.
//
// The model sees this list as a name + description + argument schema per tool,
// and picks by name. That is the whole contract: adding a capability to the
// agent is adding one struct here, not touching the loop.
import Foundation
import FoundationModels

/// Run a tool's body with a deadline. CoreLocation, CoreMotion and EventKit all
/// answer through callbacks that can simply never arrive — a denied permission,
/// a released manager, a sensor that is not there. Without this the whole agent
/// stops on one of them, which on a phone looks exactly like a slow model.
@available(iOS 27.0, *)
func withDeadline(
  _ seconds: Double = 8, _ what: String, _ body: @escaping @Sendable () async throws -> String
) async throws -> String {
  do {
    return try await firstToFinish(within: seconds, body)
  } catch is DeadlinePassed {
    return "\(what) did not answer within \(Int(seconds))s"
  }
}

struct DeadlinePassed: Error, CustomStringConvertible {
  let seconds: Double
  var description: String { "timed out after \(Int(seconds))s" }
}

/// Whichever finishes first: `body`, or a timer that throws `DeadlinePassed`.
///
/// Not a task group on purpose. A `withThrowingTaskGroup` race looks right
/// and is dead code against a real hang: the group awaits *every* child
/// before it returns, even after the timer child throws, and a child stuck
/// in a callback that never comes — or awaiting a detached task's `.value`,
/// which cancellation does not interrupt — never finishes. The bench watched
/// a hung engine for five minutes past a 180 s "timeout" this way. This
/// version resumes the caller the moment the timer fires and lets the hung
/// work stay hung on its own thread; the caller decides what to do about a
/// process that now has one wedged task in it.
@available(iOS 27.0, *)
func firstToFinish<T: Sendable>(
  within seconds: Double, _ body: @escaping @Sendable () async throws -> T
) async throws -> T {
  let gate = OnceGate<T>()
  return try await withCheckedThrowingContinuation { continuation in
    gate.arm(continuation)
    Task.detached {
      do { gate.finish(.success(try await body())) } catch { gate.finish(.failure(error)) }
    }
    Task.detached {
      try? await Task.sleep(for: .seconds(seconds))
      gate.finish(.failure(DeadlinePassed(seconds: seconds)))
    }
  }
}

/// Resumes a continuation exactly once, from whichever racer gets there first.
final class OnceGate<T: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private var continuation: CheckedContinuation<T, Error>?

  func arm(_ continuation: CheckedContinuation<T, Error>) {
    lock.lock()
    self.continuation = continuation
    lock.unlock()
  }

  func finish(_ result: Result<T, Error>) {
    lock.lock()
    let pending = continuation
    continuation = nil
    lock.unlock()
    pending?.resume(with: result)
  }
}

@available(iOS 27.0, *)
enum ToolBox {
  /// Tools that need no permission prompt. Start here when checking whether a
  /// model calls anything at all — a refusal dialog in the middle of a demo
  /// looks like the model failed when it did its part.
  static let ambient: [any FoundationModels.Tool] = [
    CurrentTimeTool(), DeviceInfoTool(), BatteryTool(), StorageTool(),
    PowerStateTool(), LocaleTool(), NetworkTool(), CalculateTool(),
    ReadBrightnessTool(), VolumeTool(), OrientationTool(), AudioRouteTool(),
    ScreenInfoTool(), AccessibilitySettingsTool(), TimeZoneTool(),
    ListDocumentsTool(), ReadNotesTool(), PendingNotificationsTool(),
    DetectLanguageTool(), SentimentTool(), TranslateTool(),
  ]

  /// Tools that change something the user can see, hear or feel.
  static let actions: [any FoundationModels.Tool] = [
    BrightnessTool(), TorchTool(), SpeakTool(), HapticTool(),
    ReadClipboardTool(), WriteClipboardTool(), OpenURLTool(), OpenMapsTool(),
    NotificationTool(),
    SystemSoundTool(), BadgeTool(), WriteNoteTool(), CancelNotificationsTool(),
    TimerTool(),
  ]

  /// Tools behind a permission prompt.
  static let personal: [any FoundationModels.Tool] = [
    LocationTool(), PlaceNameTool(), SearchPlacesTool(),
    HeadingTool(), AltitudeTool(), StepsTool(), AttitudeTool(), MotionActivityTool(),
    SoundLevelTool(), ReadPhotoTextTool(), ClassifyPhotoTool(), EditPhotoTool(),
    StepChartTool(),
    ListEventsTool(), CreateEventTool(), ListRemindersTool(), CreateReminderTool(),
    SearchContactsTool(), PhotoLibraryTool(),
  ]

  /// The photo-editing scenario pack. Session-stateful: edits stack, undo
  /// steps back, save writes out. The near-neighbor names (crop/resize/zoom,
  /// brightness/exposure/contrast) are a benchmark axis.
  static let photoEditing: [any FoundationModels.Tool] = [
    CropPhotoTool(), ResizePhotoTool(), ZoomPhotoTool(), RotatePhotoTool(), FlipPhotoTool(),
    BrightnessPhotoTool(), ExposurePhotoTool(), ContrastPhotoTool(),
    SaturationPhotoTool(), WarmthPhotoTool(), FilterPhotoTool(), BlurPhotoTool(),
    AutoEnhancePhotoTool(), CutOutSubjectTool(),
    UndoPhotoEditTool(), ResetPhotoEditsTool(), SavePhotoTool(),
  ]

  /// The stage cut of the photo pack: no undo_photo_edit. Measured, not
  /// cosmetic — every "undo everything / revert / reset" wording routed to
  /// the one-step undo on the 1.2B while it was present. With it gone,
  /// revert_to_original owns the going-back words.
  static let photoStage: [any FoundationModels.Tool] = photoEditing.filter {
    $0.name != "undo_photo_edit"
  }

  /// The focus scenario pack: one vague sentence fanning out into
  /// notifications, a timer and the screen. The discrimination axes are the
  /// `set_` prefix neighbors (timer/brightness/torch), the get/set brightness
  /// pair, and remind-vs-remember (schedule_notification vs write_note).
  static let focus: [any FoundationModels.Tool] = [
    TimerTool(), BrightnessTool(), ReadBrightnessTool(),
    NotificationTool(), CancelNotificationsTool(), PendingNotificationsTool(),
    WriteNoteTool(), ReadNotesTool(), TorchTool(), HapticTool(),
  ]

  /// The field-report scenario pack: a photographed gauge becoming a note and
  /// a next-morning reminder, fully offline. Discrimination axes: the
  /// read/identify photo pair, reminder-vs-calendar, and a date argument
  /// ("tomorrow") the model can only fill by asking the phone what today is.
  static let fieldReport: [any FoundationModels.Tool] = [
    ReadPhotoTextTool(), ClassifyPhotoTool(), PhotoLibraryTool(),
    WriteNoteTool(), ReadNotesTool(),
    CreateReminderTool(), ListRemindersTool(),
    CreateEventTool(), ListEventsTool(), CurrentTimeTool(),
  ]

  /// Quick packs for trying the rest of the phone on Apple's model. Not
  /// benchmarked yet; each becomes a scenario pack the day it earns cases.
  static let briefing: [any FoundationModels.Tool] = [
    CurrentTimeTool(), BatteryTool(), PowerStateTool(),
    ListEventsTool(), ListRemindersTool(), StepsTool(), StepChartTool(),
  ]
  static let sensors: [any FoundationModels.Tool] = [
    LocationTool(), PlaceNameTool(), HeadingTool(), MotionActivityTool(),
    SoundLevelTool(), AltitudeTool(), OrientationTool(), AttitudeTool(),
  ]
  static let handoff: [any FoundationModels.Tool] = [
    TorchTool(), SystemSoundTool(), HapticTool(), BadgeTool(),
    WriteClipboardTool(), ReadClipboardTool(), NotificationTool(), WriteNoteTool(),
    BrightnessTool(),
  ]

  /// Tools that are several tools (Tools/CompoundTools.swift). One name to
  /// say, one call, the app walks the steps — for the jobs that are always
  /// the same three steps, and for the models that cannot chain.
  static let compound: [any FoundationModels.Tool] = [
    FocusSessionTool(), MorningBriefingTool(), ShareLocationTool(), PhotoTextToNoteTool(),
  ]

  /// The chains pack: every beat wants two or three calls out of one
  /// sentence, on tools that have nothing to do with each other. Apple FM
  /// chains; the 1.2B stops after one; this is where that shows.
  static let chains: [any FoundationModels.Tool] = [
    TorchTool(), BatteryTool(), LocationTool(), SearchPlacesTool(), OpenMapsTool(),
    CurrentTimeTool(), ReadPhotoTextTool(), TranslateTool(), SoundLevelTool(),
    WriteNoteTool(),
  ]

  static let all: [any FoundationModels.Tool] =
    ambient + actions + personal + photoEditing + compound

  /// The set the scripted run uses.
  ///
  /// Not a size optimization — a correctness one. Routing across all 54 is past
  /// what a 1.2B can do: "read the text in my photo" went to `get_audio_route`,
  /// "what is my compass heading" to `get_orientation`. Six distinct jobs
  /// it can tell apart. The whole set stays in the tool sheet.
  static let demo: [any FoundationModels.Tool] = [
    LocationTool(), SearchPlacesTool(), ReadPhotoTextTool(), TranslateTool(),
    SpeakLastAnswerTool(), OpenMapsTool(),
  ]

  /// The system prompt. Kept short on purpose: it is prefilled on every turn,
  /// and on a 1–3B model running on the phone every sentence here is paid for
  /// twice — once in latency, once in the attention it takes from the question.
  // Positive imperative, no "you cannot" list: told it "cannot speak out
  // loud yourself", the 1.2B apologized that it can't speak instead of
  // calling the speak tool. Small models quote their limitations back.
  static let instructions = """
    You are running on the user's iPhone and can operate it through tools.
    Prefer a tool over guessing: you cannot know the time, the battery level or
    where the phone is without calling one. When a tool matches the request,
    call it instead of answering yourself. Call one tool at a time. When a
    tool has returned, answer the user in one short sentence using its result.
    """

  static func summary(for tools: [any FoundationModels.Tool]) -> String {
    tools.map(\.name).joined(separator: ", ")
  }
}

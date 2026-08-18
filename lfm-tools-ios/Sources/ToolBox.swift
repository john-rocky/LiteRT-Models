// Every tool the demo offers, in one list.
//
// The model sees this list as a name + description + argument schema per tool,
// and picks by name. That is the whole contract: adding a capability to the
// agent is adding one struct here, not touching the loop.
import FoundationModels

/// Run a tool's body with a deadline. CoreLocation, CoreMotion and EventKit all
/// answer through callbacks that can simply never arrive — a denied permission,
/// a released manager, a sensor that is not there. Without this the whole agent
/// stops on one of them, which on a phone looks exactly like a slow model.
@available(iOS 27.0, *)
func withDeadline(
  _ seconds: Double = 8, _ what: String, _ body: @escaping @Sendable () async throws -> String
) async throws -> String {
  try await withThrowingTaskGroup(of: String.self) { group in
    group.addTask { try await body() }
    group.addTask {
      try await Task.sleep(for: .seconds(seconds))
      return "\(what) did not answer within \(Int(seconds))s"
    }
    let first = try await group.next()!
    group.cancelAll()
    return first
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

  static let all: [any FoundationModels.Tool] = ambient + actions + personal + photoEditing

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

// Every tool the demo offers, in one list.
//
// The model sees this list as a name + description + argument schema per tool,
// and picks by name. That is the whole contract: adding a capability to the
// agent is adding one struct here, not touching the loop.
import FoundationModels

@available(iOS 27.0, *)
enum ToolBox {
  /// Tools that need no permission prompt. Start here when checking whether a
  /// model calls anything at all — a refusal dialog in the middle of a demo
  /// looks like the model failed when it did its part.
  static let ambient: [any Tool] = [
    CurrentTimeTool(), DeviceInfoTool(), BatteryTool(), StorageTool(),
    PowerStateTool(), LocaleTool(), NetworkTool(), CalculateTool(),
    ReadBrightnessTool(), VolumeTool(),
  ]

  /// Tools that change something the user can see, hear or feel.
  static let actions: [any Tool] = [
    BrightnessTool(), TorchTool(), SpeakTool(), HapticTool(),
    ReadClipboardTool(), WriteClipboardTool(), OpenURLTool(), NotificationTool(),
  ]

  /// Tools behind a permission prompt.
  static let personal: [any Tool] = [
    LocationTool(), PlaceNameTool(), SearchPlacesTool(),
    ListEventsTool(), CreateEventTool(), ListRemindersTool(), CreateReminderTool(),
    SearchContactsTool(), PhotoLibraryTool(),
  ]

  static let all: [any Tool] = ambient + actions + personal

  /// The system prompt. Kept short on purpose: it is prefilled on every turn,
  /// and on a 1–3B model running on the phone every sentence here is paid for
  /// twice — once in latency, once in the attention it takes from the question.
  static let instructions = """
    You are running on the user's iPhone and can operate it through tools.
    Prefer a tool over guessing: you cannot know the time, the battery level or
    where the phone is without calling one. Call one tool at a time. When a tool
    has returned, answer the user in one short sentence using its result.
    """

  static func summary(for tools: [any Tool]) -> String {
    tools.map(\.name).joined(separator: ", ")
  }
}

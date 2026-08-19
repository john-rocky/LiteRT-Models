// A tool with the real schema and no body.
//
// The benchmark has to show the model exactly what the app shows it — same
// name, same description, same argument schema, so routing and guided decoding
// are exercised unchanged — while calling it must move nothing in the world:
// no location prompt, no speech, no app switch to Maps mid-run. `call` returns
// a canned result and that is all. What the model actually called is read back
// from the session transcript afterwards, the same way the stage replay does.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
struct RecordingTool<Base: Tool>: Tool {
  let base: Base
  /// The result every call returns. Canned on purpose: a deterministic world
  /// is the only one where "the model chained search → maps correctly" is a
  /// checkable claim.
  let canned: String

  var name: String { base.name }
  var description: String { base.description }
  var parameters: GenerationSchema { base.parameters }

  func call(arguments: Base.Arguments) async throws -> String { canned }
}

@available(iOS 27.0, *)
enum BenchToolBox {
  /// The demo six, neutralized. Canned results describe one fixed world —
  /// a cafe called CAFE LA nearby, a coffee-menu photo — so multi-tool cases
  /// have something coherent to chain on.
  static let demo: [any FoundationModels.Tool] = [
    RecordingTool(base: LocationTool(), canned: "Chuo, Osaka (fix to ±25 m)"),
    RecordingTool(
      base: SearchPlacesTool(),
      canned: "- CAFE LA (140 m)\n- Blue Bottle Coffee (240 m)\n- Doutor (310 m)"),
    RecordingTool(
      base: ReadPhotoTextTool(), canned: "COFFEE MENU\nLatte ¥520\nAmericano ¥450"),
    RecordingTool(base: TranslateTool(), canned: "(translated text)"),
    RecordingTool(base: SpeakLastAnswerTool(), canned: "spoke the previous answer"),
    RecordingTool(base: OpenMapsTool(), canned: "opened it in Maps"),
  ]

  /// The photo-editing pack, neutralized. Every canned result claims success
  /// in the shape the real tool answers with — a result that reads wrong
  /// makes a model retry the call (Apple FM re-translated twice on a canned
  /// result that didn't look translated).
  static let photo: [any FoundationModels.Tool] = [
    RecordingTool(base: CropPhotoTool(), canned: "done: cropped — the result is on screen"),
    RecordingTool(base: ResizePhotoTool(), canned: "done: resized — the result is on screen"),
    RecordingTool(base: ZoomPhotoTool(), canned: "done: zoomed — the result is on screen"),
    RecordingTool(base: RotatePhotoTool(), canned: "done: rotated — the result is on screen"),
    RecordingTool(base: FlipPhotoTool(), canned: "done: flipped — the result is on screen"),
    RecordingTool(
      base: BrightnessPhotoTool(), canned: "done: brightness adjusted — the result is on screen"),
    RecordingTool(
      base: ExposurePhotoTool(), canned: "done: exposure adjusted — the result is on screen"),
    RecordingTool(
      base: ContrastPhotoTool(), canned: "done: contrast adjusted — the result is on screen"),
    RecordingTool(
      base: SaturationPhotoTool(), canned: "done: saturation adjusted — the result is on screen"),
    RecordingTool(
      base: WarmthPhotoTool(), canned: "done: warmth adjusted — the result is on screen"),
    RecordingTool(base: FilterPhotoTool(), canned: "done: look applied — the result is on screen"),
    RecordingTool(base: BlurPhotoTool(), canned: "done: blurred — the result is on screen"),
    RecordingTool(
      base: AutoEnhancePhotoTool(), canned: "done: auto-enhanced — the result is on screen"),
    RecordingTool(
      base: CutOutSubjectTool(),
      canned: "done: remove the background — the result is on screen"),
    RecordingTool(base: UndoPhotoEditTool(), canned: "undid the last edit"),
    RecordingTool(
      base: ResetPhotoEditsTool(),
      canned: "discarded all edits — back to the original photo"),
    RecordingTool(
      base: SavePhotoTool(), canned: "saved a copy with 2 edits to the photo library"),
  ]

  /// The focus pack, neutralized. One fixed world again: two notifications
  /// pending, one note remembered — so list/read cases have something real-
  /// shaped to answer from and cancel has something to claim it cancelled.
  static let focus: [any FoundationModels.Tool] = [
    RecordingTool(
      base: TimerTool(), canned: "timer set — it will ring in the system, not in this app"),
    RecordingTool(base: BrightnessTool(), canned: "brightness set"),
    RecordingTool(base: ReadBrightnessTool(), canned: "35%"),
    RecordingTool(base: NotificationTool(), canned: "notification scheduled"),
    RecordingTool(base: CancelNotificationsTool(), canned: "cancelled 2"),
    RecordingTool(
      base: PendingNotificationsTool(),
      canned: "- Stretch in 1800s\n- Stand up in 2700s"),
    RecordingTool(base: WriteNoteTool(), canned: "noted"),
    RecordingTool(
      base: ReadNotesTool(),
      canned: "2026-08-18T09:12:00Z  the wifi password is 4471"),
    RecordingTool(base: TorchTool(), canned: "torch on"),
    RecordingTool(base: HapticTool(), canned: "played the success haptic"),
  ]

  /// The field-report pack, neutralized — except the clock. A fixed world
  /// again (a gauge photo, one note, one open reminder), but get_current_time
  /// stays the real tool: `dateResolvesTo` scores against the device's date
  /// at run time, and a canned "today" would silently break every tomorrow
  /// case the day after it was written. Create results claim success without
  /// echoing specifics a mismatch could contradict.
  static let report: [any FoundationModels.Tool] = [
    RecordingTool(
      base: ReadPhotoTextTool(), canned: "TANK 3\nPRESSURE 82 PSI\nCHECKED 08:40"),
    RecordingTool(base: ClassifyPhotoTool(), canned: "gauge 92%, pipe 74%"),
    RecordingTool(
      base: PhotoLibraryTool(),
      canned: "1204 photos, 87 videos, most recent Aug 18, 2026 at 8:40"),
    RecordingTool(base: WriteNoteTool(), canned: "noted"),
    RecordingTool(
      base: ReadNotesTool(), canned: "2026-08-18T08:41:22Z  TANK 3 PRESSURE 82 PSI"),
    RecordingTool(base: CreateReminderTool(), canned: "added the reminder"),
    RecordingTool(base: ListRemindersTool(), canned: "- File the report"),
    RecordingTool(base: CreateEventTool(), canned: "created the event on the calendar"),
    RecordingTool(base: ListEventsTool(), canned: "- Aug 19 14:00 Site inspection"),
    CurrentTimeTool(),
  ]

  /// The video pack, neutralized. Results claim success in the shape the
  /// real tools answer with, against the one fixed timeline every video
  /// case's `state` describes (see the cases file): one 12.4 s landscape
  /// clip, playhead at 5 s. Nothing here renders.
  static let video: [any FoundationModels.Tool] = [
    RecordingTool(base: TrimClipTool(), canned: "trimmed — the timeline is on screen"),
    RecordingTool(base: SplitClipTool(), canned: "split — clip 1 is selected"),
    RecordingTool(base: SelectClipTool(), canned: "selected"),
    RecordingTool(base: DeleteClipTool(), canned: "deleted — 1 clip left"),
    RecordingTool(base: ClipSpeedTool(), canned: "speed set — the timeline is on screen"),
    RecordingTool(base: CropVideoTool(), canned: "cropped — the frame is on screen"),
    RecordingTool(base: AddCaptionTool(), canned: "caption added"),
    RecordingTool(base: AddFadeTool(), canned: "fade added (picture and sound)"),
    RecordingTool(base: StabilizeVideoTool(), canned: "stabilization on"),
    RecordingTool(base: VideoVolumeTool(), canned: "volume set"),
    RecordingTool(base: RevertVideoTool(), canned: "discarded all edits — back to the original video"),
    RecordingTool(base: ExportVideoTool(), canned: "exported to the photo library"),
  ]

  /// The store pack, neutralized. The canned world is the pack's own
  /// canned data, frozen at "6 products under 5 in stock, 5 orders awaiting
  /// payment"; results claim success in the shape the real tools answer.
  static let store: [any FoundationModels.Tool] = [
    RecordingTool(base: SearchProductsTool(), canned: "3 products selected (search):\nLinen Shirt — ¥6,800, stock 3, active\nOxford Shirt — ¥7,200, stock 14, active\nCamp Shirt — ¥6,200, stock 7, active"),
    RecordingTool(base: FilterProductsTool(), canned: "4 products selected (filter):\nField Jacket — ¥16,500, stock 4, draft\nCorduroy Pants — ¥9,800, stock 12, draft\nWool Cardigan — ¥13,800, stock 9, draft\nFlannel Shirt — ¥7,800, stock 15, draft"),
    RecordingTool(base: LowStockTool(), canned: "6 products selected (stock below 5):\nWool Beanie — ¥3,200, stock 0, active\nStraw Hat — ¥4,600, stock 1, active\nWide Chinos — ¥8,900, stock 2, active\nLinen Shirt — ¥6,800, stock 3, active\nWool Scarf — ¥5,400, stock 4, active\nField Jacket — ¥16,500, stock 4, draft"),
    RecordingTool(base: UpdatePriceTool(), canned: "prices changed on the selected products"),
    RecordingTool(base: SetPriceTool(), canned: "price set on the selected products"),
    RecordingTool(base: AddTagTool(), canned: "tagged the selected products"),
    RecordingTool(base: SetProductStatusTool(), canned: "status changed on the selected products"),
    RecordingTool(base: AdjustInventoryTool(), canned: "stock adjusted on the selected products"),
    RecordingTool(base: FilterOrdersTool(), canned: "5 orders selected (filter):\n#1020 Uma Reddy — ¥1,800, pending, unfulfilled, today\n#1018 Sam Doyle — ¥5,900, pending, unfulfilled, 1 d ago\n#1015 Olivia Park — ¥15,800, pending, unfulfilled, 2 d ago\n#1012 Leo Brandt — ¥5,400, pending, unfulfilled, 4 d ago\n#1008 Hana Kim — ¥8,900, pending, unfulfilled, 6 d ago"),
    RecordingTool(base: FulfillOrdersTool(), canned: "fulfilled 5 orders: #1010, #1013, #1016, #1017, #1019"),
    RecordingTool(base: SendInvoiceTool(), canned: "sent a payment reminder for 5 orders"),
    RecordingTool(base: RefundOrderTool(), canned: "refunded the order"),
    RecordingTool(base: OrderNoteTool(), canned: "note added to the selected orders"),
    RecordingTool(base: SalesSummaryTool(), canned: "last 7 days: 13 orders, ¥104,500 in sales, up 44% on the 7 days before"),
  ]

  /// `--toolset <name>` picks the pack a bench run offers the model.
  static func named(_ name: String) -> [any FoundationModels.Tool]? {
    switch name {
    case "demo": return demo
    case "photo": return photo
    case "focus": return focus
    case "report": return report
    case "video": return video
    case "store": return store
    default: return nil
    }
  }

  /// The instructions that travel with a pack — the stage's, so a bench
  /// case is the same message the demo sends.
  static func instructions(for name: String) -> String {
    switch name {
    case "video": return ToolBox.videoInstructions
    case "store": return ToolBox.storeInstructions
    default: return ToolBox.instructions
    }
  }
}

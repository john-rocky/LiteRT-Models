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
}

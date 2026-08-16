// The session, the transcript, and the model file behind them.
import Foundation
import FoundationModels
import LiteRTLM
import Observation

@available(iOS 27.0, *)
@MainActor
@Observable
final class ChatModel {
  struct Line: Identifiable {
    enum Kind { case user, assistant, tool, error }
    let id = UUID()
    let kind: Kind
    let text: String
    let detail: String?
  }

  enum Status: Equatable {
    case noModel
    case loading(String)
    case ready(String)
    case failed(String)
  }

  private(set) var status: Status = .noModel
  private(set) var lines: [Line] = []
  private(set) var thinking = false
  var backend: Backend = .cpu()
  var enabledGroups: Set<String> = ["ambient", "actions", "personal"]

  private var session: LanguageModelSession?
  private var model: LiteRTLanguageModel?

  var tools: [any Tool] {
    var tools: [any Tool] = []
    if enabledGroups.contains("ambient") { tools += ToolBox.ambient }
    if enabledGroups.contains("actions") { tools += ToolBox.actions }
    if enabledGroups.contains("personal") { tools += ToolBox.personal }
    return tools
  }

  // MARK: Model

  /// `.litertlm` bundles the user has dropped into the app's Documents folder
  /// (Files.app, or `xcrun devicectl`). The bundle is opened where it lies —
  /// copying a 1–2 GB file into a cache to read it is the difference between
  /// fitting on the device and not.
  static func availableModels() -> [URL] {
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    guard let root = documents.first,
      let files = try? FileManager.default.contentsOfDirectory(
        at: root, includingPropertiesForKeys: nil)
    else { return [] }
    return files.filter { $0.pathExtension == "litertlm" }.sorted { $0.path < $1.path }
  }

  func load(_ url: URL) async {
    status = .loading(url.lastPathComponent)
    do {
      let model = try LiteRTLanguageModel(modelPath: url.path, backend: backend)
      self.model = model
      startSession()
      status = .ready(url.lastPathComponent)
    } catch {
      status = .failed(error.localizedDescription)
    }
  }

  /// A session owns its transcript, so changing the tool set or clearing the
  /// history means a new session over the same (cached, already-loaded) engine.
  func startSession() {
    guard let model else { return }
    session = LanguageModelSession(
      model: model, tools: tools, instructions: ToolBox.instructions)
    lines = []
  }

  // MARK: Turn

  func send(_ text: String) async {
    guard let session else { return }
    let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty, !thinking else { return }
    lines.append(Line(kind: .user, text: prompt, detail: nil))
    thinking = true
    defer { thinking = false }

    let before = session.transcript.count
    do {
      let response = try await session.respond(to: prompt)
      // Tool calls are not part of the reply text — they are transcript entries
      // the session made on its own. Reading them back is the only way to show
      // what the model actually did.
      appendToolTrace(from: session.transcript, after: before)
      lines.append(Line(kind: .assistant, text: response.content, detail: nil))
    } catch {
      appendToolTrace(from: session.transcript, after: before)
      lines.append(Line(kind: .error, text: error.localizedDescription, detail: nil))
    }
  }

  private func appendToolTrace(from transcript: Transcript, after index: Int) {
    for entry in Array(transcript).dropFirst(index) {
      switch entry {
      case .toolCalls(let calls):
        for call in calls {
          lines.append(
            Line(kind: .tool, text: "called \(call.toolName)", detail: String(describing: call.arguments)))
        }
      case .toolOutput(let output):
        let text = output.segments.compactMap { segment -> String? in
          if case .text(let t) = segment { return t.content } else { return nil }
        }.joined()
        lines.append(Line(kind: .tool, text: "\(output.toolName) returned", detail: text))
      default:
        break
      }
    }
  }
}

// The session, the transcript, and the model file behind them.
import CoreGraphics
import Foundation
import FoundationModels
import LiteRTLM
import LiteRTLMFoundationModels
import Observation
import UIKit

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
    /// What the tools drew for this answer, shown under the bubble.
    var artifact: Artifact?
    /// The picture that went in with a user line, shown in its bubble.
    var image: UIImage?
  }

  /// A photo waiting to go in with the next message. iOS 27's prompt
  /// `Attachment` carries it: Apple's model reads it natively, and the LiteRT
  /// adapter turns it into image content for a vision bundle. The image is the
  /// vaguest input after speech — "what's this?" with no words at all.
  private(set) var attachedImage: CGImage?
  private(set) var attachedThumbnail: UIImage?

  func attach(_ image: CGImage) {
    attachedImage = image
    attachedThumbnail = UIImage(cgImage: image)
    // The attached picture is also "the photo" for the editing tools: "fix
    // this" edits what the model was shown, not whatever is newest.
    PhotoEditBox.shared.load(image)
  }

  func detachImage() {
    attachedImage = nil
    attachedThumbnail = nil
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
  // The engine's Metal path is the one LiteRT-LM ships for iOS; --gpu selects it
  // when the CPU path is what a failure has to be isolated from.
  var backend: Backend = CommandLine.arguments.contains("--gpu") ? .gpu : .cpu()
  var enabledGroups: Set<String> = ["ambient", "actions", "personal", "compound"]

  private var session: LanguageModelSession?
  private var model: LiteRTLanguageModel?
  /// Apple's on-device model instead of a LiteRT bundle. Same session, same
  /// tools, same transcript reading — the fastest way to try a new tool or
  /// wording is to ask the model that answers in a second.
  private var usingSystemModel = false

  var tools: [any FoundationModels.Tool] {
    // --no-tools isolates the engine from the tool path when a turn fails: a
    // plain session exercises neither the router nor constrained decoding.
    if CommandLine.arguments.contains("--no-tools") { return [] }
    if CommandLine.arguments.contains("--autorun") { return ToolBox.demo }
    var tools: [any FoundationModels.Tool] = []
    if enabledGroups.contains("ambient") { tools += ToolBox.ambient }
    if enabledGroups.contains("actions") { tools += ToolBox.actions }
    if enabledGroups.contains("personal") { tools += ToolBox.personal }
    if enabledGroups.contains("photo") { tools += ToolBox.photoEditing }
    if enabledGroups.contains("compound") { tools += ToolBox.compound }
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
    // Newest first: pushing a new bundle should be enough to switch to it.
    return files.filter { $0.pathExtension == "litertlm" }.sorted {
      let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
      let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
      return a > b
    }
  }

  func load(_ url: URL) async {
    status = .loading(url.lastPathComponent)
    do {
      // 27 tool descriptions and their argument schemas go into the system
      // prompt on every turn, and the adapter rebuilds the conversation from the
      // whole transcript each time. The convenience default of 2048 is spent
      // before the first question arrives.
      // A vision bundle (LFM2.5-VL-*) needs its encoder tower given a backend,
      // or the adapter reports no vision capability and image attachments are
      // dropped on the floor. Named by convention; the bundle does not say.
      let isVision = url.lastPathComponent.uppercased().contains("-VL-")
      let model = try LiteRTLanguageModel(
        modelPath: url.path, backend: backend, visionBackend: isVision ? .cpu() : nil,
        maxTokens: 4096, toolListStyle: .bare, thinkingTokenBudget: 32)
      self.model = model
      usingSystemModel = false
      startSession()
      status = .ready(url.lastPathComponent)
    } catch {
      status = .failed(error.localizedDescription)
    }
  }

  static let systemModelName = "Apple on-device"

  /// Apple's Foundation Model. No file, no load time; available whenever
  /// Apple Intelligence is on. `--model apple` picks it at launch.
  func loadSystemModel() {
    guard SystemLanguageModel.default.availability == .available else {
      status = .failed("Apple's on-device model is not available on this device")
      return
    }
    model = nil
    usingSystemModel = true
    startSession()
    status = .ready(Self.systemModelName)
  }

  /// A session owns its transcript, so changing the tool set or clearing the
  /// history means a new session over the same (cached, already-loaded) engine.
  func startSession() {
    if usingSystemModel {
      session = LanguageModelSession(tools: tools, instructions: ToolBox.instructions)
    } else {
      guard let model else { return }
      session = LanguageModelSession(
        model: model, tools: tools, instructions: ToolBox.instructions)
    }
    lines = []
  }

  // MARK: Turn

  func send(_ text: String) async {
    guard let session else { return }
    let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
    // A photo alone is a message: "what's this?" without the words.
    guard !prompt.isEmpty || attachedImage != nil, !thinking else { return }
    let image = attachedImage
    let thumbnail = attachedThumbnail
    detachImage()
    lines.append(
      Line(kind: .user, text: prompt, detail: nil, artifact: nil, image: thumbnail))
    _ = ArtifactBox.shared.take()  // drop anything left over from a failed turn
    live = ""
    thinking = true
    defer { thinking = false }

    let before = session.transcript.count
    do {
      // Off the main actor on purpose. The FM executor protocol declares
      // `respond` as `nonisolated(nonsending)`, so it runs on the *caller's*
      // executor — and this class is @MainActor. Left alone, the LiteRT-LM
      // engine's blocking work lands on the main thread and its callback pool
      // times out (DEADLINE_EXCEEDED), which surfaces as missing output
      // TensorBuffers. `LanguageModelSession` is @unchecked Sendable, so a
      // detached task is the supported way out.
      let response = try await Task.detached(priority: .userInitiated) {
        if let image {
          return try await session.respond(
            to: Prompt {
              prompt
              Attachment(image)
            })
        }
        return try await session.respond(to: prompt)
      }.value
      // Tool calls are not part of the reply text — they are transcript entries
      // the session made on its own. Reading them back is the only way to show
      // what the model actually did.
      appendToolTrace(from: session.transcript, after: before)
      // The card belongs to the answer, not to the tool line: it should appear
      // as the reply lands, the way something prepared while you waited does.
      lines.append(
        Line(
          kind: .assistant, text: response.content, detail: nil,
          artifact: ArtifactBox.shared.take()))
    } catch {
      appendToolTrace(from: session.transcript, after: before)
      lines.append(
        Line(
          kind: .error, text: error.localizedDescription, detail: nil,
          artifact: ArtifactBox.shared.take()))
    }
  }

  // MARK: Scripted run

  /// A fixed sequence, so a recording of the app is reproducible and so the
  /// demo does not depend on someone typing while the model is loading.
  static let script = [
    // Act 1 — what the phone knows about right now.
    "Where am I?",
    "What is my compass heading?",
    "Am I moving?",
    "How many steps have I walked today?",
    // Act 2 — out of the app and into the rest of the phone.
    "Find a coffee shop near me.",
    "Put coffee in my calendar at 3pm today for 30 minutes.",
    "Remind me in 40 seconds that the demo is running.",
    // Act 3 — read, understand, translate, speak.
    "Read the text in my latest photo.",
    // The photo is English so the OCR card reads at a glance; the translation
    // beat brings its own Japanese, so the output is still English.
    "Translate 'お腹が空きました。ラーメンが食べたい。' into English.",
    "Say the translation out loud.",
    // Act 4 — leave something behind.
    "Make a black and white copy of that photo.",
    "Chart my steps for the last 7 days.",
    // Last on purpose: this one sends the app to the background, and iOS stops
    // generating there — nothing after it would run.
    "Show the first coffee shop on the map.",
  ]

  private(set) var scriptedPrompt: String?

  /// What the model is emitting right now, before anything is parsed. On a
  /// phone a turn takes long enough that a spinner is indistinguishable from a
  /// hang; the raw tokens are the only honest progress indicator.
  private(set) var live = ""

  private(set) var lastRate: Double = 0

  func startTrace() {
    LiteRTFMTrace.onChunk = { [weak self] piece in
      Task { @MainActor in self?.live += piece }
    }
    LiteRTFMTrace.onRate = { [weak self] rate in
      Task { @MainActor in self?.lastRate = rate }
    }
  }

  func runScript() async {
    for prompt in Self.script {
      guard session != nil else { return }
      // Show the line in the composer for a beat before it is sent: on a video
      // it has to be obvious that a question went in before anything happened.
      scriptedPrompt = prompt
      try? await Task.sleep(for: .milliseconds(900))
      scriptedPrompt = nil
      await send(prompt)
      try? await Task.sleep(for: .milliseconds(600))
    }
  }

  private func appendToolTrace(from transcript: Transcript, after index: Int) {
    for entry in Array(transcript).dropFirst(index) {
      switch entry {
      case .toolCalls(let calls):
        for call in calls {
          lines.append(
            Line(
              kind: .tool, text: "called \(call.toolName)",
              detail: String(describing: call.arguments), artifact: nil))
        }
      case .toolOutput(let output):
        let text = output.segments.compactMap { segment -> String? in
          if case .text(let t) = segment { return t.content } else { return nil }
        }.joined()
        lines.append(
          Line(kind: .tool, text: "\(output.toolName) returned", detail: text, artifact: nil))
      default:
        break
      }
    }
  }
}

// Drives the stage: one beat at a time, each with a visible phase.
//
// Unlike the chat model this keeps no history on screen. What matters is the
// moment in front of you; the transcript still exists inside the session.
import Foundation
import FoundationModels
import LiteRTLM
import LiteRTLMFoundationModels
import Observation
import UIKit

@available(iOS 27.0, *)
@MainActor
@Observable
final class StageModel {
  enum Phase {
    case idle
    /// The instruction being typed into the composer, before it is sent.
    case typing
    case thinking
    case calling(name: String, arguments: String, returned: String?)
    case result(text: String, artifact: Artifact?)
  }

  /// Six beats, chosen for what a viewer can see happening rather than for
  /// coverage. Steps, tilt, arithmetic and calendar all work; none of them is a
  /// surprise, and a demo that shows everything shows nothing.
  // Four beats, not six: on the 1.2B the translate and speak beats fail the
  // same way across five prompt/description variants — the model translates
  // by itself and apologizes that it "can't speak out loud" — so the recorded
  // cut keeps only what the model actually does. The 2.6B runs all six.
  static let beats = [
    "Where am I?",
    "Find a coffee shop near me.",
    "Read the text in my latest photo.",
    // Last on purpose: this one sends the app to the background, and iOS stops
    // generating there — nothing after it would run.
    // Named, not referred to: the shop was found turns ago and the history
    // window does not reach that far, so "that coffee shop" sent the model
    // searching again instead of opening the map.
    "Open CAFE LA in Apple Maps.",
  ]

  /// The photo-editing cut: edits stacking on edits, a mistake talked back
  /// out of existence, and a save. Undo by voice is the surprise beat.
  // Runs against ToolBox.photoStage (no one-step undo): with undo present,
  // every "undo everything / revert / reset" wording routed to it on the
  // 1.2B, and "Reset it" routed to resize_photo (res- prefix). The mistake
  // beat now reverts the whole chain, visibly back to the untouched photo.
  static let photoBeats = [
    "Make the photo a bit brighter.",
    "A bit warmer, too.",
    "Crop it square.",
    "Undo everything — back to the original.",
    "Give it a sepia look.",
    "Remove the background.",
    "Save it.",
  ]

  /// The focus cut: one sentence at a time steering notifications, a timer
  /// and the screen itself. The dim beat is the visible one; the last beat is
  /// the compound — two tools out of one sentence, written in call order
  /// because the models that chain follow the sentence.
  static let focusBeats = [
    "Set a timer for 25 minutes.",
    "Remind me to stretch in half an hour.",
    "What notifications are coming up?",
    "Remember this: I stopped at page 128.",
    "Dim the screen — I need to focus.",
    "Silence all my notifications and set a one-hour focus timer.",
  ]

  /// `--scenario photo|focus` swaps the stage to that pack; default stays the
  /// coffee run. Beats and tools travel together, same as the bench.
  static var scenarioBeats: [String] {
    switch scenarioName {
    case "photo": return photoBeats
    case "focus": return focusBeats
    default: return beats
    }
  }

  static var scenarioName: String {
    guard let flag = CommandLine.arguments.firstIndex(of: "--scenario"),
      CommandLine.arguments.indices.contains(flag + 1)
    else { return "coffee" }
    return CommandLine.arguments[flag + 1].lowercased()
  }

  static var scenarioIsPhoto: Bool { scenarioName == "photo" }



  private(set) var question = ""
  /// What is in the composer right now, while the instruction is being typed.
  private(set) var typed = ""
  private(set) var live = ""
  private(set) var phase: Phase = .idle
  private(set) var rate: Double = 0
  private(set) var beatIndex = 0
  private(set) var toolCount = 0
  /// What the HUD claims. The demo runs on whichever backend is selected below;
  /// the tools, the cards and the stage are identical either way.
  private(set) var backendName = ""

  enum Backend { case system, liteRT }
  /// Apple's model by default: it is the one that answers in a second. Flip to
  /// `.liteRT` to run the same demo on LFM2.5 through LiteRT-LM.
  static let backend: Backend = .liteRT
  /// Bumped on every phase change so the view animates the swap rather than
  /// diffing an enum with associated values.
  private(set) var phaseID = 0

  /// The photo scenario's star: the edit chain's current image, on screen
  /// from the first frame to the last, updated after every edit.
  private(set) var stageImage: UIImage?
  private(set) var stageImageID = 0

  private func refreshStageImage() {
    guard Self.scenarioIsPhoto else { return }
    stageImage = PhotoEditBox.shared.currentRendered()
    stageImageID += 1
  }

  var beatCount: Int { Self.scenarioBeats.count }

  /// The bundle to run: `--model <substring>` pins one by filename
  /// (case-insensitive); otherwise the newest bundle wins. Newest-first bit a
  /// whole day of measurements when a pushed 2.6B silently became "the model".
  static func chosenModel() -> URL? {
    let models = ChatModel.availableModels()
    if let flag = CommandLine.arguments.firstIndex(of: "--model"),
      CommandLine.arguments.indices.contains(flag + 1)
    {
      let needle = CommandLine.arguments[flag + 1].lowercased()
      if let match = models.first(where: { $0.lastPathComponent.lowercased().contains(needle) }) {
        return match
      }
    }
    return models.first
  }

  private var session: LanguageModelSession?
  private let pending = TokenBuffer()
  private var pump: Task<Void, Never>?

  func start() async {
    guard session == nil else { return }
    // Tokens arrive faster than a screen can usefully change. They are collected
    // off the actor and published on a timer; one hop and one redraw per token
    // was competing with the generation itself.
    LiteRTFMTrace.onChunk = { [weak self] piece in
      self?.pending.append(piece)
      RunLog.stream(piece)
    }
    LiteRTFMTrace.onTiming = { line in RunLog.write("TIME \(line)") }
    LiteRTFMTrace.onRate = { [weak self] rate in
      Task { @MainActor in self?.rate = rate }
    }
    pump = Task { @MainActor [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .milliseconds(60))
        guard let self, let drained = self.pending.drain() else { continue }
        self.live += drained
      }
    }
    RunLog.startNewRun()
    let tools: [any FoundationModels.Tool]
    switch Self.scenarioName {
    case "photo": tools = ToolBox.photoStage
    case "focus": tools = ToolBox.focus
    default: tools = ToolBox.demo
    }
    toolCount = tools.count
    switch Self.backend {
    case .system:
      // Apple's on-device model. Same session, same tools, same cards — the
      // only thing that changes is who generates. It answers in about a second
      // where the LiteRT path spends ten on prefill alone.
      let model = SystemLanguageModel.default
      guard model.availability == .available else {
        question = "the system model is not available on this device"
        return
      }
      backendName = "Apple on-device"
      session = LanguageModelSession(tools: tools, instructions: ToolBox.instructions)
    case .liteRT:
      // Same process, same bundle, same backend as the beats that follow. The
      // standalone benchmark reported 252 tok/s of prefill where a turn in this
      // app measured about 46; running both here says whether that gap belongs
      // to the conversation path or to the environment it runs in.
      if let url = Self.chosenModel() {
        let bench = await Task.detached(priority: .userInitiated) { () -> String in
          do {
            let info = try await LiteRTLM.benchmark(
              modelPath: url.path, backend: .cpu(), prefillTokens: 256, decodeTokens: 64)
            return String(describing: info)
          } catch {
            return "failed: \(error)"
          }
        }.value
        RunLog.write("BENCH in-process \(bench.prefix(400))")
      }
      guard let url = Self.chosenModel() else {
        question = "no .litertlm bundle in the app container"
        return
      }
      do {
        // A big bundle plus a big KV is what iOS kills the app for: the 2.6B's
        // 1.55 GB of weights left nothing for a 4096-token cache and the
        // process went away during load.
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        // The KV is allocated at this size and every prefill step works against
        // it: the benchmark that reported 252 tok/s used 544, this app used
        // 4096 and measured about 50.
        let context = bytes > 1_000_000_000 ? 1024 : 2048
        // The bundle's own name, not a hardcoded label. `availableModels` is
        // newest-first, and a hardcoded "LFM2.5-1.2B" here spent a day fronting
        // for the 2.6B — every in-app number got attributed to the wrong model.
        backendName = url.deletingPathExtension().lastPathComponent
        // `.bare` writes the tool list in LFM2's trained format — no OpenAI
        // envelope — which is both what the model expects and ~180 fewer
        // prefilled characters across the demo's six tools.
        // The thinking budget is tight on purpose: LFM2.5's reasoning is
        // invisible (a metadata channel the runtime strips from the stream),
        // so on stage it is pure silence — up to 40s of it, sometimes ending
        // in an empty answer. 32 tokens caps the silence at about two seconds.
        let model = try LiteRTLanguageModel(
          modelPath: url.path, backend: .cpu(), maxTokens: context, toolListStyle: .bare,
          thinkingTokenBudget: 32)
        session = LanguageModelSession(
          model: model, tools: tools, instructions: ToolBox.instructions)
      } catch {
        question = "could not load the model: \(error.localizedDescription)"
        return
      }
    }
    RunLog.write("BACKEND \(backendName)")
    if Self.scenarioIsPhoto {
      // The photo is on stage before the first word is typed; a permission
      // prompt, if any, fires here rather than mid-beat.
      try? await PhotoEditBox.shared.preload()
      refreshStageImage()
    }
    await run()
  }

  private func run() async {
    guard let session else { return }
    for (index, prompt) in Self.scenarioBeats.enumerated() {
      RunLog.flushStream()  // beat N's tail was landing at the head of beat N+1
      RunLog.write("BEAT \(index + 1) \(prompt)")
      beatIndex = index
      question = ""
      typed = ""
      set(.typing)
      // Typed out rather than announced: on screen this has to read as somebody
      // instructing the phone, not as a chapter heading.
      for character in prompt {
        typed.append(character)
        try? await Task.sleep(for: .milliseconds(28))
      }
      try? await Task.sleep(for: .milliseconds(450))
      question = prompt
      typed = ""
      live = ""
      set(.thinking)
      _ = ArtifactBox.shared.take()

      let before = session.transcript.count
      let answer: String
      do {
        // Off the main actor. `respond` is `nonisolated(nonsending)`, so it runs
        // on the caller's executor — and this class is @MainActor. Left alone,
        // the model generates on the main thread while the same thread is being
        // asked to redraw for every token it produces.
        answer = try await Task.detached(priority: .userInitiated) {
          try await session.respond(to: prompt).content
        }.value
      } catch {
        RunLog.write("ERROR \(error)")
        answer = error.localizedDescription
      }

      // Replay what the session did, so the tool call gets its own moment on
      // screen instead of being something that already happened.
      let tail = Array(session.transcript).dropFirst(before)
      var pendingCall: (name: String, arguments: String)?
      for entry in tail {
        switch entry {
        case .toolCalls(let calls):
          if let call = calls.first {
            pendingCall = (call.toolName, Self.pretty(call.arguments))
            set(.calling(name: call.toolName, arguments: Self.pretty(call.arguments), returned: nil))
            try? await Task.sleep(for: .milliseconds(900))
          }
        case .toolOutput(let output):
          // What the tool gave back, on screen. The badge alone showed that
          // something was reached; this shows what it said.
          let returned = output.segments.compactMap { segment -> String? in
            if case .text(let text) = segment { return text.content } else { return nil }
          }.joined()
          RunLog.write("TOOL \(output.toolName) -> \(returned.prefix(300))")
          refreshStageImage()
          if let call = pendingCall {
            set(.calling(name: call.name, arguments: call.arguments, returned: returned))
            try? await Task.sleep(for: .milliseconds(1400))
          }
        default:
          break
        }
      }

      // Everything the model actually produced this beat, so a silent turn can
      // be told apart from a turn that produced something unusable.
      RunLog.write("STREAM[\(live.count)] \(live.replacingOccurrences(of: "\n", with: "⏎").prefix(400))")
      RunLog.write("ANSWER[\(answer.count)] \(answer.prefix(200))")
      LastAnswer.shared.set(answer)
      set(.result(text: answer, artifact: ArtifactBox.shared.take()))
      beatIndex = index + 1
      try? await Task.sleep(for: .seconds(3))
    }
  }

  private func set(_ next: Phase) {
    phase = next
    phaseID += 1
  }

  private static func pretty(_ arguments: GeneratedContent) -> String {
    let raw = String(describing: arguments)
    return raw.count > 90 ? String(raw.prefix(90)) + "…" : raw
  }
}


/// Collects stream chunks off the main actor so the UI can take them in batches.
final class TokenBuffer: @unchecked Sendable {
  private let lock = NSLock()
  private var text = ""

  func append(_ piece: String) {
    lock.lock()
    text += piece
    lock.unlock()
  }

  func drain() -> String? {
    lock.lock()
    defer { lock.unlock() }
    guard !text.isEmpty else { return nil }
    let out = text
    text = ""
    return out
  }
}


/// A run log in the app's Documents, pulled back with `devicectl device copy
/// from`. The console stream needs a session attached, and attaching one means
/// something can kill the app when that session ends — which is what happened
/// to a take.
enum RunLog {
  private static let lock = NSLock()

  /// A fresh name per run. Pulling a fixed path with `devicectl copy from`
  /// kept returning a cached, stale copy — which is how three "the log says
  /// nothing happened" reports got made about runs that had happened.
  nonisolated(unsafe) private static var name = "run.log"

  static func startNewRun() {
    lock.lock()
    name = "run-\(Int(Date().timeIntervalSince1970)).log"
    lock.unlock()
  }

  static var url: URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent(name)
  }

  /// Raw tokens as they arrive, coalesced into lines. The thinking is the most
  /// interesting thing on screen during a turn and it was going nowhere.
  nonisolated(unsafe) private static var streamBuffer = ""

  /// Drop whatever is half-collected, so one beat's tail does not open the next.
  static func flushStream() {
    lock.lock()
    streamBuffer = ""
    lock.unlock()
  }

  static func stream(_ piece: String) {
    lock.lock()
    streamBuffer += piece
    let flush = streamBuffer.count > 120 || piece.contains("\n")
    let out = flush ? streamBuffer : ""
    if flush { streamBuffer = "" }
    lock.unlock()
    guard flush, !out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
    write("THINK \(out.replacingOccurrences(of: "\n", with: "⏎"))")
  }

  static func write(_ line: String) {
    lock.lock()
    defer { lock.unlock() }
    let stamped = "\(Date().formatted(date: .omitted, time: .standard))  \(line)\n"
    if let handle = try? FileHandle(forWritingTo: url) {
      defer { try? handle.close() }
      try? handle.seekToEnd()
      try? handle.write(contentsOf: Data(stamped.utf8))
    } else {
      try? Data(stamped.utf8).write(to: url)
    }
  }

  static func reset() {
    lock.lock()
    defer { lock.unlock() }
    try? FileManager.default.removeItem(at: url)
  }
}

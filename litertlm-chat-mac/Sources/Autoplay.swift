import AppKit
import Foundation

/// The recorder: runs a script of conversations through the same actions the window's buttons
/// call (open the file, type into the field, Send, New conversation), checks each answer, and
/// writes a run JSON.
///
///   LiteRTLMChat --model <file.litertlm> --autoplay <script.json> --run-out <run.json> [--snapshot-dir <dir>]
///
/// Each conversation after the first starts with New conversation, so it gets a new engine.
/// Scripts are in `checks/`.
@MainActor
enum Autoplay {
  struct Script: Decodable {
    struct Turn: Decodable {
      var text: String
      /// A regular expression the answer has to contain (case-insensitive).
      var expect: String?
      /// Press Stop once the answer has this many pieces of text.
      var stopAfterChunks: Int?
    }

    struct Conversation: Decodable {
      var turns: [Turn]
    }

    var name: String
    /// 0 puts the whole question in the field at once.
    var typingCharactersPerSecond: Double?
    var pauseBeforeSendSeconds: Double?
    var pauseAfterAnswerSeconds: Double?
    /// Show the empty window this long before the file is opened (for recordings).
    var pauseBeforeOpenSeconds: Double?
    /// Keep the last screen this long before closing the engine (for recordings).
    var holdSecondsAtEnd: Double?
    var quitWhenDone: Bool?
    var conversations: [Conversation]
  }

  struct Check: Encodable {
    var conversation: Int
    var question: String
    var expect: String?
    var answer: String
    var matched: Bool?
    var degenerate: Bool
    var charRun: String?
  }

  struct Run: Encodable {
    var script: String
    var scriptPath: String
    var started: String
    var finished: String
    var app: [String: String]
    var machine: [String: String]
    var model: [String: String]
    var modelSummary: ModelSummary?
    var windowFrameFromTopLeft: [Double]?
    var conversations: [ChatModel.ConversationRecord]
    var checks: [Check]
    var verdict: [String: Int]
    var aborted: String?
    var abortedDetail: String?
    /// Conversations that ended before their script did (an answer stopped, an engine that stopped).
    var conversationEndings: [String]
    var snapshots: [String]
  }

  /// One wait (an engine build, an answer) longer than this ends the run.
  static let waitLimit: TimeInterval = 1800

  static func run(model: ChatModel, scriptPath: String, modelPath: String, outPath: String?) async {
    let started = Date()
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let script: Script
    do {
      script = try decoder.decode(Script.self, from: Data(contentsOf: URL(fileURLWithPath: scriptPath)))
    } catch {
      FileHandle.standardError.write(Data("cannot read the autoplay script \(scriptPath): \(error)\n".utf8))
      return
    }
    NSApp.activate(ignoringOtherApps: true)
    let snapshots = Snapshots(directory: Launch.value(after: "--snapshot-dir"))

    var checks: [Check] = []
    var endings: [String] = []
    var aborted: ChatModel.Problem?
    await pause(script.pauseBeforeOpenSeconds)
    model.open(URL(fileURLWithPath: modelPath))
    conversations: for (index, conversation) in script.conversations.enumerated() {
      let c = index + 1
      if index > 0 { model.newConversation() }
      var tookBuilding = false
      let ready = await waitUntil(model, { $0.phase == .ready || $0.isStopped }) {
        if !tookBuilding, case .building(let since) = model.phase, Date().timeIntervalSince(since) >= 5 {
          tookBuilding = true
          snapshots.take("c\(c)-building")
        }
      }
      guard ready else {
        aborted = ChatModel.Problem(title: "timed out waiting for the engine", detail: "", canRetry: false)
        break
      }
      if case .stopped(let problem) = model.phase {
        aborted = problem
        await pause(0.5)
        await snapshots.takeAndWait("c\(c)-stopped")
        break
      }
      await snapshots.takeAndWait("c\(c)-ready")
      for (t, turn) in conversation.turns.enumerated() {
        let n = t + 1
        await type(turn.text, into: model, perSecond: script.typingCharactersPerSecond ?? 0)
        await pause(script.pauseBeforeSendSeconds)
        model.send()
        var tookStreaming = false
        var pressedStop = false
        let answered = await waitUntil(model, { $0.phase != .answering }) {
          let chunks = model.turns.last?.chunks ?? 0
          if !tookStreaming, chunks >= 40 {
            tookStreaming = true
            snapshots.take("c\(c)-t\(n)-streaming")
          }
          if !pressedStop, let limit = turn.stopAfterChunks, chunks >= limit {
            pressedStop = true
            model.stopAnswer()
          }
        }
        guard answered else {
          aborted = ChatModel.Problem(title: "timed out waiting for an answer", detail: "", canRetry: false)
          break conversations
        }
        // A stopped answer closes its engine: wait for that to finish too.
        guard await waitUntil(model, { $0.phase == .ready || $0.isStopped }) else {
          aborted = ChatModel.Problem(title: "timed out waiting for the engine to settle", detail: "", canRetry: false)
          break conversations
        }
        await pause(0.5)
        await snapshots.takeAndWait("c\(c)-t\(n)-answer")
        let answer = model.turns.last?.answer ?? ""
        checks.append(
          Check(
            conversation: c, question: turn.text, expect: turn.expect, answer: answer,
            matched: turn.expect.map { AnswerCheck.matches(answer, $0) },
            degenerate: AnswerCheck.degenerate(answer), charRun: AnswerCheck.charRun(answer)))
        if case .stopped(let problem) = model.phase {
          // The conversation is over. A later conversation in the script starts on a new engine.
          endings.append("conversation \(c): \(problem.title)")
          if problem.canRetry && index + 1 < script.conversations.count { continue conversations }
          aborted = problem
          break conversations
        }
        await pause(script.pauseAfterAnswerSeconds)
      }
    }
    await pause(script.holdSecondsAtEnd)

    let summary = model.summary
    let frame = WindowConfigurator.frameFromTopLeft()
    // Close the last engine too, so its exit and peak memory are in the record.
    model.closeEngine()
    _ = await waitUntil(model, { !$0.engineRunning })

    let expected = checks.filter { $0.expect != nil }
    let run = Run(
      script: script.name, scriptPath: scriptPath,
      started: ISO8601DateFormatter().string(from: started),
      finished: ISO8601DateFormatter().string(from: Date()),
      app: [
        "version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "",
        "runtime": "LiteRT-LM 0.18.1 Swift package",
        "backend": "gpu",
        "cache_dir": ":nocache",
        "sampler": "not set by the app (the file's own)",
        "log_directory": model.logDirectory.path,
      ],
      machine: machine(),
      model: [
        "path": modelPath,
        "size_bytes": model.file.map { String($0.size) } ?? "",
        "inode": model.file.map { String($0.inode) } ?? "",
      ],
      modelSummary: summary, windowFrameFromTopLeft: frame, conversations: model.allRecords, checks: checks,
      verdict: [
        "answers": checks.count,
        "expected": expected.count,
        "matched": expected.filter { $0.matched == true }.count,
        "degenerate": checks.filter(\.degenerate).count,
      ],
      aborted: aborted?.title, abortedDetail: aborted?.detail, conversationEndings: endings,
      snapshots: snapshots.taken)
    if let outPath {
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
      do {
        try encoder.encode(run).write(to: URL(fileURLWithPath: outPath))
      } catch {
        FileHandle.standardError.write(Data("cannot write \(outPath): \(error)\n".utf8))
      }
    }
    if script.quitWhenDone == true {
      NSApp.terminate(nil)
    }
  }

  private static func type(_ text: String, into model: ChatModel, perSecond: Double) async {
    guard perSecond > 0 else {
      model.draft = text
      return
    }
    model.draft = ""
    for character in text {
      model.draft.append(character)
      try? await Task.sleep(nanoseconds: UInt64(1e9 / perSecond))
    }
  }

  private static func pause(_ seconds: Double?) async {
    guard let seconds, seconds > 0 else { return }
    try? await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
  }

  private static func waitUntil(
    _ model: ChatModel, _ condition: (ChatModel) -> Bool, onTick: () -> Void = {}
  ) async -> Bool {
    let deadline = Date().addingTimeInterval(waitLimit)
    while !condition(model) {
      if Date() > deadline { return false }
      onTick()
      try? await Task.sleep(nanoseconds: 100_000_000)
    }
    return true
  }

  private static func machine() -> [String: String] {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var name = [CChar](repeating: 0, count: max(size, 1))
    sysctlbyname("hw.model", &name, &size, nil, 0)
    return [
      "model": String(cString: name),
      "os": ProcessInfo.processInfo.operatingSystemVersionString,
      "physical_memory_bytes": String(ProcessInfo.processInfo.physicalMemory),
      "assumed_physical_memory_bytes": SystemMemory.physicalIsAssumed ? String(SystemMemory.physicalBytes) : "",
    ]
  }
}

/// Saves a picture of the window at named moments of a recorder run (`--snapshot-dir <dir>`) with
/// macOS's `screencapture`, the window only. The Screen Recording permission is the one of the app
/// that started this one (Terminal, for example).
@MainActor
final class Snapshots {
  private let directory: URL?
  private(set) var taken: [String] = []

  init(directory: String?) {
    self.directory = directory.map { URL(fileURLWithPath: $0) }
  }

  /// Starts a picture and returns at once (for moments while an answer streams).
  func take(_ name: String) {
    _ = start(name)
  }

  /// Takes a picture and returns once it is saved, so the next step cannot change the window first.
  func takeAndWait(_ name: String) async {
    guard let process = start(name) else { return }
    await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
      DispatchQueue.global().async {
        process.waitUntilExit()
        done.resume()
      }
    }
  }

  private func start(_ name: String) -> Process? {
    guard let directory, let window = NSApp.windows.first(where: { $0.isVisible }) else { return nil }
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    process.arguments = ["-x", "-o", "-l", String(window.windowNumber), directory.appendingPathComponent("\(name).png").path]
    guard (try? process.run()) != nil else { return nil }
    taken.append(name)
    return process
  }
}

/// The answer checks of the Qwen3.8-27B card's 8-question test (the same rules, in Swift).
enum AnswerCheck {
  static func matches(_ answer: String, _ pattern: String) -> Bool {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return false }
    return regex.firstMatch(in: answer, range: NSRange(answer.startIndex..., in: answer)) != nil
  }

  /// Empty, a run of one character, or one word making up more than half of 12 or more words.
  static func degenerate(_ text: String) -> Bool {
    let words = all(#"\w+"#, in: text.lowercased())
    if words.count >= 12 {
      let top = Dictionary(grouping: words, by: { $0 }).values.map(\.count).max() ?? 0
      if Double(top) / Double(words.count) > 0.5 { return true }
    }
    if charRun(text) != nil { return true }
    return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }

  /// 4+ of one symbol in a row (`!!!!`), 5+ of one letter in a row, or text with no letter or digit.
  static func charRun(_ text: String) -> String? {
    for pattern in [#"([^\w\s])\1{3,}"#, #"([^\W\d_])\1{4,}"#] {
      if let regex = try? NSRegularExpression(pattern: pattern),
        let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
        let range = Range(match.range, in: text)
      {
        return String(text[range].prefix(20))
      }
    }
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if !trimmed.isEmpty, text.range(of: #"[^\W_]"#, options: .regularExpression) == nil {
      return String(trimmed.prefix(20))
    }
    return nil
  }

  private static func all(_ pattern: String, in text: String) -> [String] {
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
    return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
      Range($0.range, in: text).map { String(text[$0]) }
    }
  }
}

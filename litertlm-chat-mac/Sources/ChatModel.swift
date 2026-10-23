import AppKit
import Foundation
import UniformTypeIdentifiers

/// Everything the window shows, and the actions its buttons call. The recorder (Autoplay.swift)
/// calls the same actions.
@MainActor
final class ChatModel: ObservableObject {
  struct ModelFile: Equatable {
    var url: URL
    var size: UInt64
    var inode: UInt64

    var name: String { url.lastPathComponent }

    /// "Qwen3.8-27B_mixed_int4.litertlm" → "Qwen3.8-27B".
    var displayName: String {
      let stem = url.deletingPathExtension().lastPathComponent
      return stem.split(separator: "_").first.map(String.init) ?? stem
    }
  }

  struct Problem: Equatable {
    var title: String
    var detail: String
    /// Whether building the engine again for the same file makes sense.
    var canRetry: Bool
  }

  enum Phase: Equatable {
    case empty
    case starting
    case building(since: Date)
    case ready
    case answering
    case closing
    case stopped(Problem)
  }

  enum TurnState: Equatable {
    case streaming, done, cancelled
    case failed(String)
  }

  struct Turn: Identifiable, Equatable {
    let id: Int
    let question: String
    let sentAt: Date
    var answer = ""
    var state = TurnState.streaming
    var firstChunkAt: Date?
    var chunks = 0
    /// The host's clock for the first and the latest piece of text.
    var firstChunkTime: Double?
    var lastChunkTime: Double?
    var stats: TurnStats?

    /// Text pieces per second since the first one, while the answer streams, on the host's
    /// clock. The runtime sends one piece per decode step that adds text, so this follows the
    /// decode rate; the runtime's own figure replaces it when the answer is done.
    var liveRate: Double? {
      guard chunks >= 2, let first = firstChunkTime, let last = lastChunkTime, last > first else { return nil }
      return Double(chunks - 1) / (last - first)
    }
  }

  struct Memory: Equatable {
    var current: UInt64
    var peak: UInt64
  }

  /// What the recorder writes per conversation.
  struct TurnRecord: Codable {
    var question: String
    var answer: String
    var state: String
    /// Send to first piece of text as the window saw it (the host's own figure is in `stats`).
    var windowTimeToFirstChunk: Double?
    var stats: TurnStats?
    /// Wall-clock times (seconds since 1970): sent, first piece of text, done.
    var sentAt: Double?
    var firstChunkAt: Double?
    var doneAt: Double?
  }

  struct ConversationRecord: Codable {
    var index: Int
    var hostPid: Int32
    var engineLog: String
    var spawnToReadySeconds: Double?
    var hostBuildSeconds: Double?
    var peakFootprintAtReadyBytes: UInt64?
    var maxSampledFootprintBytes: UInt64?
    var webgpuErrorLines = 0
    var ending: String?
    var problem: String?
    var turns: [TurnRecord] = []
    /// Wall-clock times (seconds since 1970): the engine process started, the engine was ready.
    var spawnedAt: Double?
    var readyAt: Double?
  }

  /// The engine's peak memory as a multiple of the file size, used to refuse a file that cannot
  /// fit. Qwen3.8-27B's 15.4 GB file peaked at 24.9 GB on the Mac GPU (1.62×).
  static let memoryPerFileByte = 1.65

  @Published private(set) var file: ModelFile?
  @Published private(set) var summary: ModelSummary?
  @Published private(set) var phase = Phase.empty
  @Published private(set) var turns: [Turn] = []
  @Published var draft = ""
  @Published private(set) var clock = Date()
  @Published private(set) var buildSeconds: Double?
  @Published private(set) var lastBuildSeconds: Double?
  @Published private(set) var memory: Memory?
  @Published private(set) var webGPUErrors = 0
  @Published private(set) var memoryNote: String?

  private(set) var records: [ConversationRecord] = []
  private var current: ConversationRecord?
  private var client: EngineClient?
  private var conversationNumber = 0
  private var nextTurnID = 1
  private var spawnedAt = Date()
  private var lastError: HostEvent?
  private var expectingEnd = false
  private var afterEnd: (() -> Void)?
  /// What to show once an engine that was closed on purpose has ended, when nothing follows.
  private var closingProblem: Problem?
  private var ticker: Timer?
  private var activity: NSObjectProtocol?
  private let defaults = UserDefaults.standard

  let logDirectory: URL
  private let runtimeLog: String?

  init() {
    if let custom = defaults.string(forKey: "logDirectory") {
      logDirectory = URL(fileURLWithPath: custom)
    } else {
      logDirectory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/LiteRTLMChat")
    }
    runtimeLog = defaults.string(forKey: "runtimeLog")
  }

  var lastModelURL: URL? {
    defaults.string(forKey: "lastModelPath").map { URL(fileURLWithPath: $0) }
  }

  var engineRunning: Bool { client != nil }

  var isStopped: Bool {
    if case .stopped = phase { return true }
    return false
  }

  var canStartNewConversation: Bool {
    guard file != nil else { return false }
    switch phase {
    case .ready, .answering: return true
    case .stopped(let problem): return problem.canRetry
    default: return false
    }
  }

  /// Whether the open file is one the engine can run (so far as the app knows).
  var fileIsUsable: Bool {
    guard file != nil else { return false }
    if case .stopped(let problem) = phase { return problem.canRetry }
    return true
  }

  var allRecords: [ConversationRecord] { records + (current.map { [$0] } ?? []) }

  // MARK: Actions

  func chooseFile() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [UTType(filenameExtension: "litertlm") ?? .data]
    panel.allowsMultipleSelection = false
    panel.canChooseDirectories = false
    panel.message = "Choose a LiteRT-LM model file (.litertlm)"
    if panel.runModal() == .OK, let url = panel.url {
      open(url)
    }
  }

  func open(_ url: URL) {
    if client != nil {
      stopEngine { [weak self] in self?.load(url) }
    } else {
      load(url)
    }
  }

  /// A fresh engine for a fresh conversation (LiteRT-LM issue 3165; EngineHost.swift).
  func newConversation() {
    guard file != nil else { return }
    if client != nil {
      stopEngine { [weak self] in self?.startEngine() }
    } else {
      startEngine()
    }
  }

  func send() {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard phase == .ready, !text.isEmpty, let client else { return }
    let turn = Turn(id: nextTurnID, question: text, sentAt: Date())
    nextTurnID += 1
    turns.append(turn)
    draft = ""
    phase = .answering
    client.send(HostCommand(cmd: .send, id: turn.id, text: text))
  }

  func ask(_ example: String) {
    draft = example
    send()
  }

  func stopAnswer() {
    client?.send(HostCommand(cmd: .cancel))
  }

  /// Ends the engine without starting another one (the recorder's last step).
  func closeEngine() {
    guard client != nil else { return }
    stopEngine(then: nil)
  }

  // MARK: Engine lifecycle

  private func load(_ url: URL) {
    file = nil
    summary = nil
    turns = []
    memoryNote = nil
    let path = url.path
    guard FileManager.default.fileExists(atPath: path) else {
      phase = .stopped(
        Problem(
          title: "File not found",
          detail: "Nothing is at \(path). The file may have been moved, renamed or deleted; open it from where it is now.",
          canRetry: false))
      return
    }
    guard url.pathExtension.lowercased() == "litertlm" else {
      phase = .stopped(
        Problem(
          title: "Not a .litertlm file",
          detail: "LiteRT-LM opens a model from a file whose name ends in .litertlm, and “\(url.lastPathComponent)” does not.",
          canRetry: false))
      return
    }
    let attributes = try? FileManager.default.attributesOfItem(atPath: path)
    let size = (attributes?[.size] as? NSNumber)?.uint64Value ?? 0
    let inode = (attributes?[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
    let model = ModelFile(url: url, size: size, inode: inode)
    file = model
    defaults.set(path, forKey: "lastModelPath")

    let needed = UInt64(Double(size) * Self.memoryPerFileByte)
    let physical = SystemMemory.physicalBytes
    if physical < needed {
      var detail =
        "This model needs about \(Format.gb(needed)) of memory while it runs, and this Mac has \(Format.ram(physical)). "
        + "The estimate is \(Self.memoryPerFileByte)× the file size: Qwen3.8-27B's 15.4 GB file used 24.9 GB on the GPU."
      if SystemMemory.physicalIsAssumed {
        detail += "\n\n(This Mac's memory size was set with -assumePhysicalMemoryGB.)"
      }
      phase = .stopped(Problem(title: "Not enough memory", detail: detail, canRetry: false))
      return
    }
    if let available = SystemMemory.availableBytes, available < needed {
      memoryNote =
        "macOS has \(Format.gb(available)) of memory free right now and the engine needs about \(Format.gb(needed)). "
        + "Quit other apps, or macOS will swap and the build will take longer."
    }
    startEngine()
  }

  private func startEngine() {
    guard let file else { return }
    conversationNumber += 1
    turns = []
    summary = nil
    buildSeconds = nil
    lastBuildSeconds = defaults.object(forKey: buildKey(file)) as? Double
    memory = nil
    webGPUErrors = 0
    lastError = nil
    expectingEnd = false
    spawnedAt = Date()
    phase = .starting
    do {
      let client = try EngineClient(
        modelPath: file.url.path, logDirectory: logDirectory, runtimeLog: runtimeLog,
        onEvent: { [weak self] event in
          MainActor.assumeIsolated { self?.handle(event) }
        },
        onWebGPUError: { [weak self] _ in
          MainActor.assumeIsolated { self?.noteWebGPUError() }
        },
        onEnd: { [weak self] ending in
          MainActor.assumeIsolated { self?.engineEnded(ending) }
        })
      self.client = client
      current = ConversationRecord(index: conversationNumber, hostPid: client.pid, engineLog: client.logURL.path)
      current?.spawnedAt = spawnedAt.timeIntervalSince1970
      // No App Nap while an engine lives: the window keeps up with the answer even when it is
      // not in front.
      activity = activity ?? ProcessInfo.processInfo.beginActivity(
        options: .userInitiatedAllowingIdleSystemSleep, reason: "A LiteRT-LM engine is running")
    } catch {
      phase = .stopped(
        Problem(title: "Could not start the engine", detail: error.localizedDescription, canRetry: true))
      return
    }
    startTicker()
  }

  private func stopEngine(then next: (() -> Void)?, problem: Problem? = nil) {
    afterEnd = next
    closingProblem = problem
    expectingEnd = true
    phase = .closing
    client?.stop()
  }

  private func handle(_ event: HostEvent) {
    switch event.event {
    case .modelInfo:
      summary = event.model
    case .building:
      phase = .building(since: spawnedAt)
    case .ready:
      buildSeconds = event.buildSeconds
      if let file, let seconds = event.buildSeconds {
        defaults.set(seconds, forKey: buildKey(file))
      }
      current?.hostBuildSeconds = event.buildSeconds
      current?.spawnToReadySeconds = Date().timeIntervalSince(spawnedAt)
      current?.peakFootprintAtReadyBytes = event.peakFootprintBytes
      current?.readyAt = Date().timeIntervalSince1970
      phase = .ready
    case .chunk:
      guard let index = turns.lastIndex(where: { $0.id == event.id }) else { return }
      if turns[index].firstChunkAt == nil { turns[index].firstChunkAt = Date() }
      if let t = event.t {
        if turns[index].firstChunkTime == nil { turns[index].firstChunkTime = t }
        turns[index].lastChunkTime = t
      }
      turns[index].chunks += 1
      turns[index].answer += event.text ?? ""
    case .turnDone:
      guard let index = turns.lastIndex(where: { $0.id == event.id }) else { return }
      turns[index].stats = event.stats
      if event.cancelled == true {
        turns[index].state = .cancelled
        record(turns[index])
        // LiteRT-LM 0.18.1 does not go on after a cancel: "Reusing the session after calling
        // CancelProcess() is neither recommended nor supported" (runtime/core/session_advanced.h),
        // and the next message fails at once with "Task cancelled". So a stopped answer ends the
        // conversation, and its engine is closed to give the memory back.
        stopEngine(
          then: nil,
          problem: Problem(
            title: "Answer stopped",
            detail:
              "LiteRT-LM 0.18.1 cannot go on with a conversation after an answer is stopped, so this conversation "
              + "has ended and its engine is closed. Start a new conversation to ask again; its engine is built anew.",
            canRetry: true))
        return
      } else if let message = event.message {
        turns[index].state = .failed(message)
      } else {
        turns[index].state = .done
      }
      record(turns[index])
      if phase == .answering { phase = .ready }
    case .error:
      lastError = event
      if event.stage == .generate, let index = turns.lastIndex(where: { $0.id == event.id }) {
        turns[index].state = .failed(event.message ?? "The engine could not answer.")
        record(turns[index])
        if phase == .answering { phase = .ready }
      }
    }
  }

  private func noteWebGPUError() {
    webGPUErrors += 1
    current?.webgpuErrorLines += 1
  }

  private func engineEnded(_ ending: EngineClient.Ending) {
    stopTicker()
    if let activity {
      ProcessInfo.processInfo.endActivity(activity)
      self.activity = nil
    }
    let logPath = client?.logURL.path ?? ""
    client = nil
    current?.ending = ending.signaled ? "signal \(ending.status)" : "exit \(ending.status)"
    for index in turns.indices where turns[index].state == .streaming {
      turns[index].state = .failed("The engine stopped before the answer was finished.")
      record(turns[index])
    }

    if expectingEnd {
      expectingEnd = false
      finishRecord(problem: nil)
      let problem = closingProblem
      closingProblem = nil
      if let next = afterEnd {
        afterEnd = nil
        next()
      } else {
        phase = .stopped(
          problem
            ?? Problem(title: "The engine is closed", detail: "Start a new conversation to build it again.", canRetry: true))
      }
      return
    }

    let problem: Problem
    let logLine = "\n\nEngine log: \(logPath)"
    if let error = lastError, error.stage == .file {
      problem = Problem(title: "Not a model this app can run", detail: error.message ?? "", canRetry: false)
    } else if let error = lastError, error.stage == .engine {
      problem = Problem(
        title: "The engine could not be built", detail: (error.message ?? "") + logLine, canRetry: true)
    } else if ending.signaled {
      var detail = "The engine process was ended by signal \(ending.status) (\(Self.signalName(ending.status)))."
      if ending.status == SIGKILL {
        detail +=
          " A process ends this way when it is force-quit, or when macOS stops it because the Mac ran out of memory."
          + " Quit other apps and start a new conversation."
      }
      problem = Problem(title: "The engine stopped", detail: detail + logLine, canRetry: true)
    } else {
      problem = Problem(
        title: "The engine stopped",
        detail: "The engine process exited with status \(ending.status)." + logLine, canRetry: true)
    }
    finishRecord(problem: problem.title)
    phase = .stopped(problem)
  }

  private func record(_ turn: Turn) {
    let state: String
    switch turn.state {
    case .streaming: state = "streaming"
    case .done: state = "done"
    case .cancelled: state = "cancelled"
    case .failed(let message): state = "failed: \(message)"
    }
    current?.turns.append(
      TurnRecord(
        question: turn.question, answer: turn.answer, state: state,
        windowTimeToFirstChunk: turn.firstChunkAt.map { $0.timeIntervalSince(turn.sentAt) },
        stats: turn.stats, sentAt: turn.sentAt.timeIntervalSince1970,
        firstChunkAt: turn.firstChunkAt?.timeIntervalSince1970, doneAt: Date().timeIntervalSince1970))
  }

  private func finishRecord(problem: String?) {
    guard var record = current else { return }
    record.problem = problem
    records.append(record)
    current = nil
  }

  private func startTicker() {
    ticker?.invalidate()
    ticker = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
      MainActor.assumeIsolated { self?.tick() }
    }
  }

  private func stopTicker() {
    ticker?.invalidate()
    ticker = nil
  }

  private func tick() {
    clock = Date()
    // A process that is ending reports 0.
    guard let client, let footprint = SystemMemory.footprint(pid: client.pid), footprint.current > 0 else { return }
    memory = Memory(current: footprint.current, peak: footprint.peak)
    let sampled = max(current?.maxSampledFootprintBytes ?? 0, footprint.current)
    current?.maxSampledFootprintBytes = sampled
  }

  private func buildKey(_ file: ModelFile) -> String {
    "buildSeconds.\(file.name).\(file.size)"
  }

  private static func signalName(_ signal: Int32) -> String {
    switch signal {
    case SIGKILL: return "SIGKILL"
    case SIGTERM: return "SIGTERM"
    case SIGABRT: return "SIGABRT"
    case SIGSEGV: return "SIGSEGV"
    case SIGBUS: return "SIGBUS"
    case SIGILL: return "SIGILL"
    case SIGTRAP: return "SIGTRAP"
    default: return "signal \(signal)"
    }
  }
}

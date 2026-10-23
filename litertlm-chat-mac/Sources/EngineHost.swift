import Darwin
import Foundation
import LiteRTLM

#if canImport(CLiteRTLM)
  import CLiteRTLM
#endif

/// The headless half of the app. For every conversation the window starts
/// `<this executable> --engine-host --model <path>` and talks to it in JSON lines (Protocol.swift).
///
/// Why one process per conversation:
/// - LiteRT-LM issue 3165: a new Conversation on an Engine that already ran one can continue from
///   the earlier conversation's recurrent state, and Qwen3.5-family hybrids (Qwen3.8-27B among them)
///   carry such a state. So every conversation gets a fresh Engine.
/// - On the Mac GPU (WebGPU), processes that built four Engines one after another logged WebGPU
///   validation errors in 2 of 44 engines, both times in the third engine of the process (Qwen3.8-27B
///   and variants of it); 130 processes that built one Engine each logged none.
/// - An engine that runs the Mac out of memory takes this process down, not the window.
enum EngineHost {
  static let flag = "--engine-host"

  static func main(arguments: [String]) -> Never {
    // Stdout carries the protocol. File descriptor 1 is pointed at stderr so that anything else
    // printed to stdout lands in the engine log instead of in the middle of an event.
    let protocolFD = dup(STDOUT_FILENO)
    dup2(STDERR_FILENO, STDOUT_FILENO)
    let options = HostOptions(arguments: arguments)
    setRuntimeLogLevel(options.runtimeLog)
    // No App Nap for the engine: it is the work the user is waiting for.
    let activity = ProcessInfo.processInfo.beginActivity(
      options: .userInitiatedAllowingIdleSystemSleep, reason: "LiteRT-LM engine")
    let host = HostSession(options: options, events: EventWriter(fd: protocolFD))
    Thread.detachNewThread { host.readCommands() }
    Task.detached { await host.start() }
    withExtendedLifetime(activity) { dispatchMain() }
  }

  /// `--runtime-log verbose|info|warning|error` sets LiteRT-LM's minimum log level.
  private static func setRuntimeLogLevel(_ name: String?) {
    #if canImport(CLiteRTLM)
      let levels: [String: LiteRtLmLogSeverity] = [
        "verbose": kLiteRtLmLogSeverityVerbose, "info": kLiteRtLmLogSeverityInfo,
        "warning": kLiteRtLmLogSeverityWarning, "error": kLiteRtLmLogSeverityError,
      ]
      if let name, let level = levels[name] {
        litert_lm_set_min_log_level(level)
      }
    #endif
  }
}

struct HostOptions {
  var modelPath = ""
  var maxTokens: Int?
  var runtimeLog: String?

  init(arguments: [String]) {
    var i = 0
    while i < arguments.count {
      let next = i + 1 < arguments.count ? arguments[i + 1] : nil
      switch arguments[i] {
      case "--model":
        modelPath = next ?? ""
        i += 1
      case "--max-tokens":
        maxTokens = next.flatMap { Int($0) }
        i += 1
      case "--runtime-log":
        runtimeLog = next
        i += 1
      default:
        break
      }
      i += 1
    }
  }
}

/// Writes one event per line to the protocol pipe. Several threads send events.
final class EventWriter: @unchecked Sendable {
  private let fd: Int32
  private let lock = NSLock()
  private let encoder = JSONEncoder()

  init(fd: Int32) {
    self.fd = fd
  }

  func send(_ event: HostEvent) {
    lock.lock()
    defer { lock.unlock() }
    guard var data = try? encoder.encode(event) else { return }
    data.append(0x0A)
    data.withUnsafeBytes { raw in
      var offset = 0
      while offset < raw.count {
        let n = write(fd, raw.baseAddress! + offset, raw.count - offset)
        if n < 0 && errno == EINTR { continue }
        if n <= 0 { return }
        offset += n
      }
    }
  }
}

final class HostSession: @unchecked Sendable {
  private let options: HostOptions
  private let events: EventWriter
  private let lock = NSLock()
  private var engine: Engine?
  private var conversation: Conversation?
  private var busy = false
  private var cancelRequested = false

  init(options: HostOptions, events: EventWriter) {
    self.options = options
    self.events = events
  }

  func start() async {
    let path = options.modelPath
    log("model \(path)")

    // 1. The file's header: fast, and where a file that is not a LiteRT-LM bundle fails.
    let summary: ModelSummary
    do {
      let info = try ModelInfo(throwingModelPath: path)
      let type: String
      switch info.modelType() {
      case .llm: type = "llm"
      case .embedding: type = "embedding"
      case .unknown: type = "unknown"
      }
      summary = ModelSummary(
        modelType: type, maxContextTokens: info.maxContextTokens(),
        supportsThinking: info.llm?.supportsThinking() ?? false,
        minRuntimeVersion: info.minRuntimeVersion)
    } catch {
      fail(.file, "LiteRT-LM could not read this file: \(Self.describe(error))", HostExit.badFile)
    }
    log(
      "model type \(summary.modelType), context \(summary.maxContextTokens) tokens, "
        + "thinking \(summary.supportsThinking), min runtime \(summary.minRuntimeVersion ?? "-")")
    events.send(HostEvent(event: .modelInfo, model: summary))
    guard summary.modelType == "llm" else {
      fail(.file, "This file holds an \(summary.modelType) model, not a text-generation model.", HostExit.badFile)
    }

    // 2. The engine: minutes for a 27B model on the GPU.
    events.send(HostEvent(event: .building))
    let started = DispatchTime.now().uptimeNanoseconds
    do {
      ExperimentalFlags.optIntoExperimentalAPIs()
      ExperimentalFlags.enableBenchmark = true
      // ":nocache": LiteRT-LM would otherwise write its GPU weight and program caches (about 16 GB
      // for Qwen3.8-27B) next to the model, and on this file they did not make the build faster.
      let config = try EngineConfig(
        modelPath: path, backend: .gpu, maxNumTokens: options.maxTokens, cacheDir: ":nocache")
      let engine = Engine(engineConfig: config)
      try await engine.initialize()
      let conversation = try await engine.createConversation()
      keep(engine, conversation)
    } catch {
      fail(.engine, Self.describe(error), HostExit.engineFailed)
    }
    let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
    let peak = SystemMemory.footprint(pid: getpid())?.peak
    log(String(format: "engine ready in %.1f s, peak footprint %llu bytes", seconds, peak ?? 0))
    events.send(HostEvent(event: .ready, buildSeconds: seconds, peakFootprintBytes: peak))
  }

  func send(id: Int, text: String) {
    lock.lock()
    let conversation = busy ? nil : self.conversation
    if conversation != nil {
      busy = true
      cancelRequested = false
    }
    lock.unlock()
    guard let conversation else {
      events.send(
        HostEvent(event: .error, id: id, stage: .generate, message: "The engine is not ready or still answering."))
      return
    }
    log("turn \(id): \(text.count) characters")
    Task.detached { [self] in
      var stats = TurnStats()
      var failure: String?
      let sent = DispatchTime.now().uptimeNanoseconds
      var first: UInt64?
      var last: UInt64?
      do {
        for try await message in conversation.sendMessageStream(Message(text)) {
          let piece = Self.text(of: message)
          if piece.isEmpty { continue }
          let now = DispatchTime.now().uptimeNanoseconds
          if first == nil { first = now }
          last = now
          stats.chunks += 1
          events.send(HostEvent(event: .chunk, id: id, text: piece, t: Double(now - sent) / 1e9))
        }
      } catch {
        if !isCancelRequested() { failure = Self.describe(error) }
      }
      if let first { stats.timeToFirstChunk = Double(first - sent) / 1e9 }
      if let first, let last { stats.streamSeconds = Double(last - first) / 1e9 }
      if let info = try? conversation.getBenchmarkInfo() {
        stats.prefillTokens = info.lastPrefillTokenCount
        stats.prefillTokensPerSecond = info.lastPrefillTokensPerSecond
        stats.decodeTokens = info.lastDecodeTokenCount
        stats.decodeTokensPerSecond = info.lastDecodeTokensPerSecond
        stats.runtimeTimeToFirstToken = info.timeToFirstTokenInSecond
      }
      stats.contextTokens = try? conversation.getTokenCount()
      if let memory = SystemMemory.footprint(pid: getpid()) {
        stats.footprintBytes = memory.current
        stats.peakFootprintBytes = memory.peak
      }
      let cancelled = isCancelRequested()
      log(
        "turn \(id) \(cancelled ? "cancelled" : failure == nil ? "done" : "failed"): "
          + "chunks \(stats.chunks), ttft \(stats.timeToFirstChunk ?? -1) s, "
          + "prefill \(stats.prefillTokens ?? -1) tokens at \(stats.prefillTokensPerSecond ?? -1) tok/s, "
          + "decode \(stats.decodeTokens ?? -1) tokens at \(stats.decodeTokensPerSecond ?? -1) tok/s, "
          + "context \(stats.contextTokens ?? -1), peak footprint \(stats.peakFootprintBytes ?? 0) bytes"
          + (failure.map { ", error: \($0)" } ?? ""))
      events.send(HostEvent(event: .turnDone, id: id, message: failure, stats: stats, cancelled: cancelled))
      setIdle()
    }
  }

  private func keep(_ engine: Engine, _ conversation: Conversation) {
    lock.lock()
    defer { lock.unlock() }
    self.engine = engine
    self.conversation = conversation
  }

  private func setIdle() {
    lock.lock()
    defer { lock.unlock() }
    busy = false
  }

  func cancel() {
    lock.lock()
    let conversation = busy ? self.conversation : nil
    if conversation != nil { cancelRequested = true }
    lock.unlock()
    guard let conversation else { return }
    log("cancel")
    try? conversation.cancel()
  }

  /// Runs on its own thread for the life of the process.
  func readCommands() {
    let decoder = JSONDecoder()
    while let line = readLine(strippingNewline: true) {
      guard let data = line.data(using: .utf8),
        let command = try? decoder.decode(HostCommand.self, from: data)
      else { continue }
      switch command.cmd {
      case .send: send(id: command.id ?? 0, text: command.text ?? "")
      case .cancel: cancel()
      case .quit: finish("quit")
      }
    }
    finish("the window closed the pipe")
  }

  private func isCancelRequested() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelRequested
  }

  private func finish(_ why: String) -> Never {
    log("exit (\(why)), peak footprint \(SystemMemory.footprint(pid: getpid())?.peak ?? 0) bytes")
    // _exit: no atexit handlers or C++ static destructors while an engine thread may still run.
    _exit(0)
  }

  private func fail(_ stage: HostEvent.Stage, _ message: String, _ code: Int32) -> Never {
    log("error (\(stage.rawValue)): \(message)")
    events.send(HostEvent(event: .error, stage: stage, message: message))
    _exit(code)
  }

  private func log(_ line: String) {
    FileHandle.standardError.write(Data("[host] \(line)\n".utf8))
  }

  static func text(of message: Message) -> String {
    var text = ""
    for content in message.contents {
      if case .text(let piece) = content { text += piece }
    }
    return text
  }

  static func describe(_ error: Error) -> String {
    (error as? LocalizedError)?.errorDescription ?? "\(error)"
  }
}

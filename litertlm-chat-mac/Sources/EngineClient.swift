import Darwin
import Foundation

/// The window's side of one engine host process: starts it, sends commands, relays its events,
/// writes its stderr to a log file, and reports how it ended. Callbacks arrive on the main queue.
final class EngineClient {
  struct Ending {
    var status: Int32
    var signaled: Bool
    var stderrTail: [String]
  }

  let logURL: URL
  var pid: Int32 { process.processIdentifier }

  private let process = Process()
  private let input: FileHandle
  private let queue = DispatchQueue(label: "engine-client")
  private let log: FileHandle?
  private var stdoutBuffer = Data()
  private var stderrBuffer = Data()
  private var tail: [String] = []
  private var stdoutClosed = false
  private var stderrClosed = false
  private var exitStatus: (status: Int32, signaled: Bool)?
  private var reported = false

  private let onEvent: (HostEvent) -> Void
  private let onWebGPUError: (String) -> Void
  private let onEnd: (Ending) -> Void

  /// - Parameters:
  ///   - onWebGPUError: called for each stderr line with `Validation error` or `Shape mismatch`.
  ///     On LiteRT-LM 0.18.1 a WebGPU validation error does not stop the engine (the process goes
  ///     on and exits 0) but the answers that follow can be wrong.
  init(
    modelPath: String, logDirectory: URL, runtimeLog: String?,
    onEvent: @escaping (HostEvent) -> Void,
    onWebGPUError: @escaping (String) -> Void,
    onEnd: @escaping (Ending) -> Void
  ) throws {
    self.onEvent = onEvent
    self.onWebGPUError = onWebGPUError
    self.onEnd = onEnd

    try FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
    let stamp = DateFormatter()
    stamp.dateFormat = "yyyyMMdd-HHmmss-SSS"
    logURL = logDirectory.appendingPathComponent("engine-\(stamp.string(from: Date())).log")
    FileManager.default.createFile(atPath: logURL.path, contents: nil)
    log = try? FileHandle(forWritingTo: logURL)

    var arguments = [EngineHost.flag, "--model", modelPath]
    if let runtimeLog { arguments += ["--runtime-log", runtimeLog] }
    process.executableURL = Bundle.main.executableURL
    process.arguments = arguments
    let stdinPipe = Pipe()
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardInput = stdinPipe
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    input = stdinPipe.fileHandleForWriting

    stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      if data.isEmpty { handle.readabilityHandler = nil }
      self?.queue.async { self?.consumeStdout(data) }
    }
    stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
      let data = handle.availableData
      if data.isEmpty { handle.readabilityHandler = nil }
      self?.queue.async { self?.consumeStderr(data) }
    }
    process.terminationHandler = { [weak self] process in
      let signaled = process.terminationReason == .uncaughtSignal
      self?.queue.async {
        self?.exitStatus = (process.terminationStatus, signaled)
        self?.reportIfFinished()
      }
      // A process that left a child holding the pipes open would never send EOF.
      self?.queue.asyncAfter(deadline: .now() + 3) {
        self?.stdoutClosed = true
        self?.stderrClosed = true
        self?.reportIfFinished()
      }
    }

    try process.run()
    let header = "[app] engine host pid \(pid): \(arguments.joined(separator: " "))\n"
    try? log?.write(contentsOf: Data(header.utf8))
  }

  var isRunning: Bool { process.isRunning }

  func send(_ command: HostCommand) {
    guard var data = try? JSONEncoder().encode(command) else { return }
    data.append(0x0A)
    // The app ignores SIGPIPE, so a host that is already gone makes this throw instead of
    // ending the app.
    try? input.write(contentsOf: data)
  }

  /// Asks the host to end (it exits when its stdin closes) and kills it if it has not ended
  /// after `grace` seconds, for example while it is stuck inside a GPU call.
  func stop(grace: TimeInterval = 5) {
    try? input.close()
    let pid = self.pid
    DispatchQueue.global().asyncAfter(deadline: .now() + grace) { [weak self] in
      guard let self, self.process.isRunning else { return }
      kill(pid, SIGKILL)
    }
  }

  private func consumeStdout(_ data: Data) {
    if data.isEmpty {
      stdoutClosed = true
      reportIfFinished()
      return
    }
    stdoutBuffer.append(data)
    while let newline = stdoutBuffer.firstIndex(of: 0x0A) {
      let line = stdoutBuffer[stdoutBuffer.startIndex..<newline]
      stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newline)
      guard let event = try? JSONDecoder().decode(HostEvent.self, from: line) else { continue }
      DispatchQueue.main.async { self.onEvent(event) }
    }
  }

  private func consumeStderr(_ data: Data) {
    if data.isEmpty {
      stderrClosed = true
      reportIfFinished()
      return
    }
    try? log?.write(contentsOf: data)
    stderrBuffer.append(data)
    while let newline = stderrBuffer.firstIndex(of: 0x0A) {
      let line = String(decoding: stderrBuffer[stderrBuffer.startIndex..<newline], as: UTF8.self)
      stderrBuffer.removeSubrange(stderrBuffer.startIndex...newline)
      tail.append(line)
      if tail.count > 40 { tail.removeFirst(tail.count - 40) }
      if line.contains("Validation error") || line.contains("Shape mismatch") {
        DispatchQueue.main.async { self.onWebGPUError(line) }
      }
    }
  }

  private func reportIfFinished() {
    guard !reported, stdoutClosed, stderrClosed, let exit = exitStatus else { return }
    reported = true
    let note = "[app] engine host \(exit.signaled ? "killed by signal" : "exited with status") \(exit.status)\n"
    try? log?.write(contentsOf: Data(note.utf8))
    try? log?.close()
    let ending = Ending(status: exit.status, signaled: exit.signaled, stderrTail: tail)
    DispatchQueue.main.async { self.onEnd(ending) }
  }
}

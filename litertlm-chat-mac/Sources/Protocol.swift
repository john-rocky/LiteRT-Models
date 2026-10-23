import Foundation

// The window and the engine host talk in JSON lines: commands on the host's stdin, events on its
// stdout. The host's stderr is the engine log (the runtime's own messages plus `[host]` lines).

struct HostCommand: Codable {
  enum Kind: String, Codable {
    case send, cancel, quit
  }

  var cmd: Kind
  var id: Int?
  var text: String?
}

struct ModelSummary: Codable, Equatable {
  /// "llm", "embedding" or "unknown", as the runtime reads the file's header.
  var modelType: String
  var maxContextTokens: Int
  var supportsThinking: Bool
  var minRuntimeVersion: String?
}

struct TurnStats: Codable, Equatable {
  /// Text pieces the stream delivered (the runtime sends one per decode step that adds text).
  var chunks = 0
  /// Send to the first piece of text, measured in the host: the time to first token a user sees.
  var timeToFirstChunk: Double?
  /// First to last piece of text.
  var streamSeconds: Double?
  /// The runtime's own counters for this turn (LiteRT-LM BenchmarkInfo, last prefill / decode turn).
  var prefillTokens: Int?
  var prefillTokensPerSecond: Double?
  var decodeTokens: Int?
  var decodeTokensPerSecond: Double?
  /// BenchmarkInfo's time to first token. In LiteRT-LM 0.18.1 it is computed from the
  /// conversation's FIRST prefill and decode turns, so it repeats turn 1's value on later turns;
  /// kept for the record, not shown.
  var runtimeTimeToFirstToken: Double?
  /// Tokens in the conversation's context after this turn.
  var contextTokens: Int?
  /// The host's physical memory footprint: now, and the most it has ever used.
  var footprintBytes: UInt64?
  var peakFootprintBytes: UInt64?
}

struct HostEvent: Codable {
  enum Kind: String, Codable {
    case modelInfo = "model_info"
    case building
    case ready
    case chunk
    case turnDone = "turn_done"
    case error
  }

  enum Stage: String, Codable {
    case file, engine, generate
  }

  var event: Kind
  var id: Int?
  var text: String?
  var stage: Stage?
  var message: String?
  var model: ModelSummary?
  var buildSeconds: Double?
  var peakFootprintBytes: UInt64?
  var stats: TurnStats?
  var cancelled: Bool?
  /// For a chunk: seconds since the host received the message, on the host's clock. The window's
  /// live rate uses these times, so a busy or throttled window does not change it.
  var t: Double?
}

/// Exit codes of the host, so the window can say what went wrong after the process is gone.
enum HostExit {
  static let badFile: Int32 = 3
  static let engineFailed: Int32 = 4
}

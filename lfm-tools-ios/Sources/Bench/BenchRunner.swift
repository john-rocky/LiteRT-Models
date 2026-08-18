// `--toolbench` turns the app into a tool-calling benchmark: every case in
// Documents/toolbench-cases.json against one model, results as JSONL back
// into Documents.
//
// One model per launch, chosen with `--model` (`apple` for the system model,
// any filename substring for a LiteRT bundle, newest bundle otherwise). The
// engines do not reliably give memory back mid-process; relaunching per model
// is how the numbers stay attributable. The Mac side loops — see
// edge-agent-lab/ios/bench/run-device.sh.
import Foundation
import FoundationModels
import LiteRTLM
import LiteRTLMFoundationModels

@available(iOS 27.0, *)
enum BenchRunner {
  private enum Chosen {
    case apple
    case liteRT(URL)
  }

  // On the main actor like the stage: model choice and session creation live
  // there, and everything slow is detached anyway.
  @MainActor
  static func run() async {
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    // Per-run filename. `devicectl copy from` returns stale content for a
    // path it has copied before; a fixed name here cost days elsewhere.
    let out = JSONLWriter(
      url: documents.appendingPathComponent(
        "toolbench-\(Int(Date().timeIntervalSince1970)).jsonl"))
    // The Mac script waits for this sentinel in the file *listing* — pulling a
    // growing file repeatedly returns devicectl's stale cached copy; a fresh
    // name that only appears when the run is over dodges that. Deferred so an
    // error exit still releases the waiting script.
    defer {
      try? Data("done\n".utf8).write(
        to: out.url.deletingPathExtension().appendingPathExtension("done"))
    }

    let cases: [BenchCase]
    do {
      let data = try Data(contentsOf: documents.appendingPathComponent("toolbench-cases.json"))
      cases = try JSONDecoder().decode([BenchCase].self, from: data)
    } catch {
      out.write(["type": "error", "what": "toolbench-cases.json: \(error)"])
      return
    }

    // Which scenario pack the model sees. The cases file and the tool set
    // travel together — pushing photo cases with the demo tools measures
    // nothing.
    var toolsetName = "demo"
    if let flag = CommandLine.arguments.firstIndex(of: "--toolset"),
      CommandLine.arguments.indices.contains(flag + 1)
    {
      toolsetName = CommandLine.arguments[flag + 1].lowercased()
    }
    guard let tools = BenchToolBox.named(toolsetName) else {
      out.write(["type": "error", "what": "unknown toolset \(toolsetName)"])
      return
    }

    let chosen: Chosen
    if CommandLine.arguments.contains("--model"),
      let flag = CommandLine.arguments.firstIndex(of: "--model"),
      CommandLine.arguments.indices.contains(flag + 1),
      CommandLine.arguments[flag + 1].lowercased() == "apple"
    {
      chosen = .apple
    } else if let url = StageModel.chosenModel() {
      chosen = .liteRT(url)
    } else {
      out.write(["type": "error", "what": "no model: no .litertlm in Documents, --model apple not given"])
      return
    }

    let modelName: String
    var liteRTModel: LiteRTLanguageModel?
    switch chosen {
    case .apple:
      guard SystemLanguageModel.default.availability == .available else {
        out.write(["type": "error", "what": "system model unavailable"])
        return
      }
      modelName = "apple-fm"
    case .liteRT(let url):
      do {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        // Same sizing as the stage demo: a 2.6B plus a 2048-token KV is what
        // iOS kills the process for.
        let context = bytes > 1_000_000_000 ? 1024 : 2048
        liteRTModel = try LiteRTLanguageModel(
          modelPath: url.path, backend: .cpu(), maxTokens: context, toolListStyle: .bare,
          thinkingTokenBudget: 32)
        modelName = url.deletingPathExtension().lastPathComponent
      } catch {
        out.write(["type": "error", "what": "model load: \(error)"])
        return
      }
    }
    // The run's own date, so `dateResolvesTo` can be re-scored offline:
    // "tomorrow" only means something relative to the day the run happened.
    let dayFormatter = DateFormatter()
    dayFormatter.dateFormat = "yyyy-MM-dd"
    out.write([
      "type": "run", "model": modelName, "cases": cases.count, "toolset": toolsetName,
      "date": dayFormatter.string(from: Date()),
    ])

    var passed = 0
    var failed = 0
    for benchCase in cases {
      if benchCase.image != nil {
        out.write(["type": "skip", "case": benchCase.id, "why": "image cases need the VLM stage"])
        continue
      }
      // A fresh session per case: no history, no carried KV, every case pays
      // the same prefill. Cross-turn behavior is a different benchmark.
      let session: LanguageModelSession
      if let model = liteRTModel {
        session = LanguageModelSession(
          model: model, tools: tools, instructions: ToolBox.instructions)
      } else {
        session = LanguageModelSession(tools: tools, instructions: ToolBox.instructions)
      }

      // Written before the respond, so a hang is attributable to its case —
      // one engine freeze left a 30-minute silence with nothing to say where.
      out.write(["type": "start", "case": benchCase.id])
      let started = Date()
      var answer = ""
      var errorText: String?
      do {
        // Detached for the same reason as the stage: `respond` runs on the
        // caller's executor. Raced against a deadline — one transient engine
        // hang has been observed, and it must cost one case, not the run.
        answer = try await withTimeout(180) {
          try await Task.detached(priority: .userInitiated) {
            try await session.respond(to: benchCase.input).content
          }.value
        }
      } catch let timeout as Timeout {
        // The engine is hung, not slow: reading the transcript now blocks on
        // the same engine lock the generation is stuck behind — that is how
        // the first hang ate 27 minutes without even writing an error line.
        // Write what is known, then stop this model's run; every later case
        // would create a session against the same wedged engine.
        out.write([
          "case": benchCase.id, "lang": benchCase.lang, "model": modelName,
          "input": benchCase.input, "error": String(describing: timeout),
          "pass": false, "selectionPass": false, "argsPass": false,
          "ms": Int(Date().timeIntervalSince(started) * 1000),
        ])
        out.write(["type": "abort", "why": "engine hang; remaining cases not run"])
        failed += 1
        break
      } catch {
        errorText = String(describing: error)
      }
      let ms = Int(Date().timeIntervalSince(started) * 1000)

      // What actually got called, from the transcript — the same source the
      // stage replay trusts.
      var calls: [(tool: String, args: [String: Any], raw: String)] = []
      for entry in session.transcript {
        guard case .toolCalls(let toolCalls) = entry else { continue }
        for call in toolCalls {
          let raw = String(describing: call.arguments)
          let args =
            raw.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) }
            as? [String: Any] ?? [:]
          calls.append((call.toolName, args, raw))
        }
      }

      let called = calls.map(\.tool)
      let expected = benchCase.expected.map(\.tool)
      let selectionPass = called == expected
      var argsPass = true
      for (index, want) in benchCase.expected.enumerated() {
        guard let matchers = want.args, !matchers.isEmpty else { continue }
        // Align by position when the order matched, by name otherwise.
        let call =
          index < calls.count && calls[index].tool == want.tool
          ? calls[index] : calls.first { $0.tool == want.tool }
        guard let call else {
          argsPass = false
          continue
        }
        for (key, matcher) in matchers where !matcher.matches(call.args[key]) {
          argsPass = false
        }
      }
      let pass = selectionPass && argsPass && errorText == nil
      if pass { passed += 1 } else { failed += 1 }

      var line: [String: Any] = [
        "case": benchCase.id, "lang": benchCase.lang, "model": modelName,
        "input": benchCase.input,
        "expected": expected, "called": called,
        "calls": calls.map { ["tool": $0.tool, "args": $0.raw] },
        "selectionPass": selectionPass, "argsPass": argsPass, "pass": pass,
        "ms": ms, "answer": String(answer.prefix(200)),
      ]
      if let errorText { line["error"] = errorText }
      out.write(line)
      print("TOOLBENCH \(benchCase.id) \(pass ? "PASS" : "FAIL") \(ms)ms \(called)")
    }
    out.write([
      "type": "summary", "model": modelName, "passed": passed, "failed": failed,
      "total": passed + failed,
    ])
    print("TOOLBENCH done \(passed)/\(passed + failed)")
  }

  struct Timeout: Error, CustomStringConvertible {
    let seconds: Double
    var description: String { "timed out after \(Int(seconds))s" }
  }

  /// Unlike `withDeadline`, timing out is an error here, not a result string —
  /// a timeout must land in the JSONL as `error`, not as the model's answer.
  private static func withTimeout(
    _ seconds: Double, _ body: @escaping @Sendable () async throws -> String
  ) async throws -> String {
    try await withThrowingTaskGroup(of: String.self) { group in
      group.addTask { try await body() }
      group.addTask {
        try await Task.sleep(for: .seconds(seconds))
        throw Timeout(seconds: seconds)
      }
      let first = try await group.next()!
      group.cancelAll()
      return first
    }
  }
}

/// One JSON object per line, appended as it happens — a died run keeps every
/// case it finished.
struct JSONLWriter {
  let url: URL

  func write(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
      let line = String(data: data, encoding: .utf8)
    else { return }
    if let handle = try? FileHandle(forWritingTo: url) {
      defer { try? handle.close() }
      try? handle.seekToEnd()
      try? handle.write(contentsOf: Data((line + "\n").utf8))
    } else {
      try? Data((line + "\n").utf8).write(to: url)
    }
  }
}

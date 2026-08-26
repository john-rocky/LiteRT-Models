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
#if canImport(LiteRTLM)
  import LiteRTLM
#endif
#if canImport(LiteRTLMFoundationModels)
  import LiteRTLMFoundationModels
#endif
import UIKit

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
    let documents = AppFiles.documents
    // Per-run filename. `devicectl copy from` returns stale content for a
    // path it has copied before; a fixed name here cost days elsewhere.
    let out = JSONLWriter(
      url: documents.appendingPathComponent(
        "toolbench-\(Int(Date().timeIntervalSince1970)).jsonl"))
    // A backgrounded app generates nothing and its timers stop with it — two
    // runs froze ~20 s in with no error line, exactly what an auto-lock or a
    // stray tap on the home screen looks like from the Mac. Keep the screen
    // awake, and if the app is sent to the background anyway, say so in the
    // log rather than leaving a silence to diagnose.
    UIApplication.shared.isIdleTimerDisabled = true
    let backgrounded = NotificationCenter.default.addObserver(
      forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main
    ) { _ in
      out.write(["type": "background", "at": ISO8601DateFormatter().string(from: Date())])
    }
    defer { NotificationCenter.default.removeObserver(backgrounded) }
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
    guard var tools = BenchToolBox.named(toolsetName) else {
      out.write(["type": "error", "what": "unknown toolset \(toolsetName)"])
      return
    }
    // `--only a,b,c` cuts the toolset down to the named tools, list order
    // preserved — the ladder runs (evaluation program #1) vary nothing but
    // the count. A name the toolset lacks is an error, not a silent shrink:
    // a subset without a case's correct tool measures nothing.
    if let flag = CommandLine.arguments.firstIndex(of: "--only"),
      CommandLine.arguments.indices.contains(flag + 1)
    {
      let want = CommandLine.arguments[flag + 1].split(separator: ",").map(String.init)
      let have = Set(tools.map(\.name))
      if let missing = want.first(where: { !have.contains($0) }) {
        out.write(["type": "error", "what": "--only names \(missing), not in toolset \(toolsetName)"])
        return
      }
      let keep = Set(want)
      tools = tools.filter { keep.contains($0.name) }
    }
    // `--instructions <pack>` pins the instructions independently of the
    // tool list — the cross-domain runs grow the list while each pack's
    // cases keep their own measured instructions, so tool count is the
    // only variable (the business wing's evaluation program).
    var instructionsName = toolsetName
    if let flag = CommandLine.arguments.firstIndex(of: "--instructions"),
      CommandLine.arguments.indices.contains(flag + 1)
    {
      instructionsName = CommandLine.arguments[flag + 1].lowercased()
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
    #if canImport(LiteRTLM)
      var liteRTModel: LiteRTLanguageModel?
    #endif
    switch chosen {
    case .apple:
      guard SystemLanguageModel.default.availability == .available else {
        out.write(["type": "error", "what": "system model unavailable"])
        return
      }
      modelName = "apple-fm"
    case .liteRT(let url):
      #if !canImport(LiteRTLM)
        _ = url
        out.write(["type": "error", "what": "LiteRT is not in this build; run with --model apple"])
        return
      #else
      do {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        // Same sizing as the stage demo: a 2.6B plus a 2048-token KV is what
        // iOS kills the process for.
        let context = bytes > 1_000_000_000 ? 1024 : 2048
        liteRTModel = try LiteRTLanguageModel(
          modelPath: url.path, backend: .cpu(),
          visionBackend: url.lastPathComponent.uppercased().contains("-VL-") ? .cpu() : nil,
          maxTokens: context, toolListStyle: .bare, thinkingTokenBudget: 32)
        modelName = url.deletingPathExtension().lastPathComponent
      } catch {
        out.write(["type": "error", "what": "model load: \(error)"])
        return
      }
      #endif
    }
    // The run's own date, so `dateResolvesTo` can be re-scored offline:
    // "tomorrow" only means something relative to the day the run happened.
    let dayFormatter = DateFormatter()
    dayFormatter.dateFormat = "yyyy-MM-dd"
    // The tool-list identity (evaluation program #5): the names on the run
    // line, the count on every case line — cross-config analysis needs to
    // know what list a row was measured against.
    out.write([
      "type": "run", "model": modelName, "cases": cases.count, "toolset": toolsetName,
      "tools": tools.count, "toolNames": tools.map(\.name).joined(separator: ","),
      "date": dayFormatter.string(from: Date()),
    ])

    var passed = 0
    var failed = 0
    for benchCase in cases {
      // An image case names a fixture pushed next to the cases file; it goes
      // into the prompt as an attachment and becomes "the photo" for the
      // tools, exactly as on the stage. A missing fixture is a skip, not a
      // fail — the model never saw the case.
      var attached: CGImage?
      if let fixture = benchCase.image {
        let url = documents.appendingPathComponent("toolbench-fixtures/\(fixture)")
        guard let data = try? Data(contentsOf: url), let image = UIImage(data: data)?.cgImage
        else {
          out.write(["type": "skip", "case": benchCase.id, "why": "fixture \(fixture) not found"])
          continue
        }
        attached = image
        PhotoEditBox.shared.load(image, label: SeenPhoto.singleLabel)
      }
      // A fresh session per case: no history, no carried KV, every case pays
      // the same prefill. Cross-turn behavior is a different benchmark.
      let session: LanguageModelSession
      let instructions = BenchToolBox.instructions(for: instructionsName)
      #if canImport(LiteRTLM)
        if let model = liteRTModel {
          session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
        } else {
          session = LanguageModelSession(tools: tools, instructions: instructions)
        }
      #else
        session = LanguageModelSession(tools: tools, instructions: instructions)
      #endif

      TranscriptBox.shared.attach(session)
      // A loop case runs its own multi-turn protocol (goal-driven polish);
      // an engine hang inside it aborts the run exactly like the
      // single-turn path below.
      if benchCase.loop == true {
        guard let fixture = attached else {
          out.write(["type": "skip", "case": benchCase.id, "why": "loop case without image"])
          continue
        }
        let verdict = await runLoopCase(
          benchCase, session: session, fixture: fixture, out: out,
          model: modelName, toolset: toolsetName, toolCount: tools.count)
        if verdict == .pass { passed += 1 } else { failed += 1 }
        if verdict == .hang {
          out.write(["type": "abort", "why": "engine hang; remaining cases not run"])
          break
        }
        continue
      }
      // The fakes' selection, from this case's state line — so a bulk call
      // with nothing selected gets the real app's refusal, not a lie.
      BenchSelection.shared.prime(from: benchCase.state)
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
        // A state case opens with the app's state, exactly as the stage does.
        let input = benchCase.state.map { AppState.compose(state: $0, request: benchCase.input) }
          ?? benchCase.input
        let attached = attached  // a let for the Sendable closure
        answer = try await firstToFinish(within: 180) {
          try await Task.detached(priority: .userInitiated) {
            if let attached {
              return try await session.respond(
                to: Prompt {
                  input
                  Attachment(attached).label(SeenPhoto.singleLabel)
                }
              ).content
            }
            return try await session.respond(to: input).content
          }.value
        }
      } catch let timeout as DeadlinePassed {
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
      // An ask-back case passes when the model routed the question — one
      // ask_user call and nothing else — or called nothing and asked in
      // prose ("?" is the test: a question without a question mark reads as
      // a statement on stage).
      let asked = answer.contains("?") || answer.contains("?")
      let askPass = called == ["ask_user"] || (called.isEmpty && asked)
      let answerPass: Bool
      if let keywords = benchCase.answerContains, !keywords.isEmpty {
        answerPass = keywords.contains { answer.range(of: $0, options: .caseInsensitive) != nil }
      } else {
        answerPass = true
      }
      let pass =
        benchCase.expectAsk == true
        ? (askPass && errorText == nil)
        : (selectionPass && argsPass && answerPass && errorText == nil)
      if pass { passed += 1 } else { failed += 1 }

      var line: [String: Any] = [
        "case": benchCase.id, "lang": benchCase.lang, "model": modelName,
        "toolset": toolsetName, "tools": tools.count,
        "input": benchCase.input,
        "expected": expected, "called": called,
        "calls": calls.map { ["tool": $0.tool, "args": $0.raw] },
        "selectionPass": selectionPass, "argsPass": argsPass, "pass": pass,
        "ms": ms, "answer": String(answer.prefix(200)),
      ]
      if benchCase.expectAsk == true { line["expectAsk"] = true; line["asked"] = asked }
      if benchCase.answerContains != nil { line["answerPass"] = answerPass }
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

  private enum LoopVerdict { case pass, fail, hang }

  /// The goal-driven loop: perceive → judge → act → perceive the result →
  /// judge again. Round one is the case's input (usually the silent beat)
  /// with the fixture attached; every later round attaches the photo as the
  /// model's edits left it, behind the loop reprompt. A round with no tool
  /// call is the stop — the score's `stopPass`; `maxRounds` is the cap a
  /// run that will not stop hits. `needs`/`avoid` are scored over every op
  /// of every round; oscillation (one tool pulled both ways) is recorded,
  /// not gated — the fixture's `avoid` already names the wrong direction.
  @MainActor
  private static func runLoopCase(
    _ benchCase: BenchCase, session: LanguageModelSession, fixture: CGImage,
    out: JSONLWriter, model: String, toolset: String, toolCount: Int
  ) async -> LoopVerdict {
    out.write(["type": "start", "case": benchCase.id])
    let started = Date()
    let maxRounds = benchCase.maxRounds ?? 4
    var opsPerRound: [[(tool: String, args: [String: Any], raw: String)]] = []
    var roundMs: [Int] = []
    var answer = ""
    var errorText: String?
    var stopped = false
    var seenEntries = 0

    for round in 1...maxRounds {
      let prompt = round == 1 ? benchCase.input : ToolBox.loopReprompt(lang: benchCase.lang)
      let image = round == 1 ? fixture : (PhotoEditBox.shared.currentCGImage() ?? fixture)
      let roundStarted = Date()
      do {
        answer = try await firstToFinish(within: 180) {
          try await Task.detached(priority: .userInitiated) {
            // The silent beat sends the attachment alone, like the stage.
            if prompt.isEmpty {
              return try await session.respond(
                to: Prompt { Attachment(image).label(SeenPhoto.singleLabel) }
              ).content
            }
            return try await session.respond(
              to: Prompt {
                prompt
                Attachment(image).label(SeenPhoto.singleLabel)
              }
            ).content
          }.value
        }
      } catch let timeout as DeadlinePassed {
        out.write([
          "case": benchCase.id, "lang": benchCase.lang, "model": model,
          "input": benchCase.input, "loop": true, "rounds": round,
          "error": String(describing: timeout), "pass": false,
          "ms": Int(Date().timeIntervalSince(started) * 1000),
        ])
        return .hang
      } catch {
        errorText = String(describing: error)
      }
      roundMs.append(Int(Date().timeIntervalSince(roundStarted) * 1000))

      // The calls this round made: the transcript past last round's mark.
      let entries = Array(session.transcript)
      var calls: [(tool: String, args: [String: Any], raw: String)] = []
      for entry in entries[seenEntries...] {
        guard case .toolCalls(let toolCalls) = entry else { continue }
        for call in toolCalls {
          let raw = String(describing: call.arguments)
          let args =
            raw.data(using: .utf8).flatMap { try? JSONSerialization.jsonObject(with: $0) }
            as? [String: Any] ?? [:]
          calls.append((call.toolName, args, raw))
        }
      }
      seenEntries = entries.count
      opsPerRound.append(calls)
      if errorText != nil { break }
      if calls.isEmpty {
        stopped = true
        break
      }
    }

    let ops = opsPerRound.flatMap { $0 }
    func hits(_ axis: OpAxis, _ call: (tool: String, args: [String: Any], raw: String)) -> Bool {
      guard call.tool == axis.tool else { return false }
      guard let direction = axis.direction else { return true }
      return (call.args["direction"] as? String)?.lowercased() == direction.lowercased()
    }
    let needs = benchCase.needs ?? []
    let avoid = benchCase.avoid ?? []
    // Empty `needs` is the already-good fixture: the correct run is no op
    // at all, exactly `expected: []`'s meaning on the single-turn side.
    let needsPass =
      needs.isEmpty
      ? ops.isEmpty
      : ops.contains { call in needs.contains { hits($0, call) } }
    let avoidPass = !ops.contains { call in avoid.contains { hits($0, call) } }
    let pass = needsPass && avoidPass && stopped && errorText == nil

    let opposites = [
      "brighter": "darker", "darker": "brighter", "up": "down", "down": "up",
      "more": "less", "less": "more", "warmer": "cooler", "cooler": "warmer",
      "more_vivid": "more_muted", "more_muted": "more_vivid",
    ]
    var oscillated = false
    for (index, call) in ops.enumerated() {
      guard let direction = (call.args["direction"] as? String)?.lowercased() else { continue }
      for later in ops[(index + 1)...]
      where later.tool == call.tool
        && (later.args["direction"] as? String)?.lowercased() == opposites[direction] {
        oscillated = true
      }
    }

    var line: [String: Any] = [
      "case": benchCase.id, "lang": benchCase.lang, "model": model,
      "toolset": toolset, "tools": toolCount, "input": benchCase.input,
      "loop": true, "rounds": opsPerRound.count, "stopped": stopped,
      "oscillated": oscillated,
      "ops": opsPerRound.map { round in round.map { ["tool": $0.tool, "args": $0.raw] } },
      "needsPass": needsPass, "avoidPass": avoidPass, "stopPass": stopped,
      "pass": pass, "msRounds": roundMs,
      "ms": Int(Date().timeIntervalSince(started) * 1000),
      "answer": String(answer.prefix(200)),
    ]
    if let errorText { line["error"] = errorText }
    out.write(line)
    print(
      "TOOLBENCH \(benchCase.id) \(pass ? "PASS" : "FAIL") rounds=\(opsPerRound.count) stopped=\(stopped) \(ops.map(\.tool))"
    )
    return pass ? .pass : .fail
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

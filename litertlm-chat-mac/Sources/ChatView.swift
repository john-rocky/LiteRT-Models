import SwiftUI

struct ChatView: View {
  @ObservedObject var model: ChatModel

  var body: some View {
    VStack(spacing: 0) {
      HeaderBar(model: model)
      Divider()
      if model.webGPUErrors > 0 {
        Banner(
          text:
            "WebGPU reported an error in this engine (\(model.webGPUErrors) log line\(model.webGPUErrors == 1 ? "" : "s")). "
            + "Answers from here on may be wrong: start a new conversation.",
          color: .red)
      }
      if let note = model.memoryNote {
        Banner(text: note, color: .orange)
      }
      if model.phase == .empty {
        WelcomeView(model: model)
      } else {
        TranscriptView(model: model)
      }
      Divider()
      Composer(model: model)
    }
    .frame(minWidth: 720, minHeight: 520)
  }
}

// MARK: Header

private struct HeaderBar: View {
  @ObservedObject var model: ChatModel

  var body: some View {
    HStack(spacing: 12) {
      Button {
        model.chooseFile()
      } label: {
        Label("Open…", systemImage: "folder")
      }
      if let file = model.file {
        VStack(alignment: .leading, spacing: 2) {
          Text(file.name)
            .font(.headline)
            .lineLimit(1)
            .truncationMode(.middle)
          Text("\(Format.gb(file.size)) · GPU · LiteRT-LM 0.18.1")
            .font(.subheadline)
            .foregroundStyle(.secondary)
        }
      } else {
        Text("No model open")
          .foregroundStyle(.secondary)
      }
      Spacer()
      VStack(alignment: .trailing, spacing: 2) {
        status
        if let memory = model.memory, model.engineRunning, model.phase != .closing {
          Text("Engine memory \(Format.gb(memory.current)) · peak \(Format.gb(memory.peak))")
            .font(.subheadline.monospacedDigit())
            .foregroundStyle(.secondary)
        }
      }
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 10)
  }

  @ViewBuilder private var status: some View {
    switch model.phase {
    case .empty:
      EmptyView()
    case .starting:
      Text("Starting…")
    case .building(let since):
      Text("Building the GPU engine · \(Format.clock(model.clock.timeIntervalSince(since)))")
        .monospacedDigit()
    case .ready, .answering:
      if let seconds = model.buildSeconds {
        Text("Engine built in \(Format.clock(seconds))")
          .monospacedDigit()
      } else {
        Text("Ready")
      }
    case .closing:
      Text("Closing the engine…")
    case .stopped(let problem):
      Text(problem.title)
        .foregroundStyle(.red)
    }
  }
}

private struct Banner: View {
  let text: String
  let color: Color

  var body: some View {
    HStack(alignment: .top, spacing: 8) {
      Image(systemName: "exclamationmark.triangle.fill")
      Text(text)
        .fixedSize(horizontal: false, vertical: true)
      Spacer(minLength: 0)
    }
    .font(.callout)
    .foregroundStyle(color)
    .padding(.horizontal, 16)
    .padding(.vertical, 8)
    .background(color.opacity(0.1))
  }
}

// MARK: Welcome

private struct WelcomeView: View {
  @ObservedObject var model: ChatModel

  var body: some View {
    VStack(spacing: 18) {
      Spacer()
      Text("Chat with a large model on this Mac")
        .font(.title)
      Text("Open a LiteRT-LM model file. The app runs it on the Mac's GPU with Google's LiteRT-LM runtime: no server, no Python.")
        .multilineTextAlignment(.center)
        .foregroundStyle(.secondary)
        .frame(maxWidth: 520)
      HStack(spacing: 12) {
        Button("Open a .litertlm file…") { model.chooseFile() }
          .controlSize(.large)
          .keyboardShortcut(.defaultAction)
        if let last = model.lastModelURL {
          Button("Open \(last.lastPathComponent) again") { model.open(last) }
            .controlSize(.large)
        }
      }
      VStack(alignment: .leading, spacing: 6) {
        Text("Qwen3.8-27B (15.4 GB, needs a Mac with 32 GB of memory or more):")
          .font(.callout)
          .foregroundStyle(.secondary)
        Text("hf download litert-community/Qwen3.8-27B Qwen3.8-27B_mixed_int4.litertlm --local-dir ~/models")
          .font(.system(.callout, design: .monospaced))
          .textSelection(.enabled)
          .padding(8)
          .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.12)))
      }
      Spacer()
    }
    .padding(24)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}

// MARK: Transcript

private struct TranscriptView: View {
  @ObservedObject var model: ChatModel

  var body: some View {
    if model.turns.isEmpty {
      // Nothing asked yet: the build clock, the examples or a problem, in the middle of the pane.
      VStack {
        Spacer()
        StatusCard(model: model)
          .frame(maxWidth: 760)
        Spacer()
      }
      .padding(20)
      .frame(maxWidth: .infinity, maxHeight: .infinity)
    } else {
      conversation
    }
  }

  private var conversation: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 22) {
          ForEach(model.turns) { turn in
            TurnView(turn: turn, speaker: model.file?.displayName ?? "Model")
          }
          StatusCard(model: model)
          Color.clear
            .frame(height: 1)
            .id("bottom")
        }
        .padding(20)
      }
      .defaultScrollAnchor(.bottom)
      .onChange(of: model.turns.last?.answer) {
        proxy.scrollTo("bottom", anchor: .bottom)
      }
      .onChange(of: model.turns.count) {
        proxy.scrollTo("bottom", anchor: .bottom)
      }
    }
  }
}

private struct TurnView: View {
  let turn: ChatModel.Turn
  let speaker: String

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      VStack(alignment: .leading, spacing: 4) {
        Text("You")
          .font(.caption.weight(.semibold))
          .foregroundStyle(.secondary)
        Text(turn.question)
          .font(.system(size: 15))
          .textSelection(.enabled)
          .padding(10)
          .background(RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.12)))
      }
      VStack(alignment: .leading, spacing: 6) {
        Text(speaker)
          .font(.caption.weight(.semibold))
          .foregroundStyle(.secondary)
        if turn.answer.isEmpty && turn.state == .streaming {
          ProgressView()
            .controlSize(.small)
        } else {
          MarkdownText(text: turn.answer)
        }
        footer
      }
    }
  }

  @ViewBuilder private var footer: some View {
    switch turn.state {
    case .streaming:
      if let rate = turn.liveRate, turn.chunks >= 8 {
        Text(String(format: "%.1f tok/s · %d tokens", rate, turn.chunks))
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
      } else if turn.chunks > 0 {
        Text("\(turn.chunks) tokens")
          .font(.caption.monospacedDigit())
          .foregroundStyle(.secondary)
      }
    case .done, .cancelled:
      HStack(spacing: 6) {
        if turn.state == .cancelled {
          Text("Stopped ·")
        }
        Text(Self.summary(of: turn))
      }
      .font(.caption.monospacedDigit())
      .foregroundStyle(.secondary)
      .textSelection(.enabled)
    case .failed(let message):
      Text(message)
        .font(.caption)
        .foregroundStyle(.red)
        .textSelection(.enabled)
    }
  }

  /// A rate over a handful of tokens is mostly the fixed cost of a call (a 25-token prompt
  /// prefills at under 20 tok/s on Qwen3.8-27B, a 1,024-token one at about 175), so a rate is
  /// shown only from these counts on; the engine log has every count and rate.
  static let minPrefillTokensForRate = 64
  static let minDecodeTokensForRate = 16

  static func summary(of turn: ChatModel.Turn) -> String {
    guard let stats = turn.stats else { return "" }
    var parts: [String] = []
    if let ttft = stats.timeToFirstChunk {
      parts.append(String(format: "first token %.2f s", ttft))
    }
    if let tokens = stats.prefillTokens, tokens > 0 {
      if let rate = stats.prefillTokensPerSecond, tokens >= minPrefillTokensForRate {
        parts.append(String(format: "prefill %d tokens at %.1f tok/s", tokens, rate))
      } else {
        parts.append("prefill \(tokens) tokens")
      }
    }
    if let tokens = stats.decodeTokens, tokens > 0 {
      if let rate = stats.decodeTokensPerSecond, tokens >= minDecodeTokensForRate {
        parts.append(String(format: "decode %d tokens at %.1f tok/s", tokens, rate))
      } else {
        parts.append("decode \(tokens) tokens")
      }
    }
    return parts.joined(separator: " · ")
  }
}

private struct StatusCard: View {
  @ObservedObject var model: ChatModel

  static let examples = [
    "What is 17 + 25? Answer briefly.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think it through step by step.",
    "Write a Python function that returns the n-th prime number, with a short docstring and one example call.",
    "Explain how a hash table handles collisions, then show a short Python example.",
  ]

  var body: some View {
    switch model.phase {
    case .starting, .closing:
      HStack(spacing: 10) {
        ProgressView().controlSize(.small)
        Text(model.phase == .closing ? "Closing the engine…" : "Reading the file…")
          .foregroundStyle(.secondary)
      }
    case .building(let since):
      buildingCard(since: since)
    case .ready where model.turns.isEmpty:
      VStack(alignment: .leading, spacing: 10) {
        Text("Ready. Ask anything, or try one:")
          .foregroundStyle(.secondary)
        ForEach(Self.examples, id: \.self) { example in
          Button {
            model.ask(example)
          } label: {
            Text(example)
              .multilineTextAlignment(.leading)
              .frame(maxWidth: 560, alignment: .leading)
          }
          .buttonStyle(.bordered)
        }
      }
    case .stopped(let problem):
      problemCard(problem)
    default:
      EmptyView()
    }
  }

  private func buildingCard(since: Date) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(alignment: .firstTextBaseline, spacing: 12) {
        ProgressView().controlSize(.small)
        Text(Format.clock(model.clock.timeIntervalSince(since)))
          .font(.system(size: 34, weight: .medium, design: .rounded).monospacedDigit())
        Text("Building the GPU engine")
          .font(.title3)
      }
      Text(buildNote)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(RoundedRectangle(cornerRadius: 10).fill(Color.secondary.opacity(0.08)))
  }

  private var buildNote: String {
    var note =
      "LiteRT-LM prepares the model for the GPU once per conversation. "
      + "A 27B model takes a few minutes; you can type your first question meanwhile."
    if let last = model.lastBuildSeconds {
      note += " Last build of this file: \(Format.clock(last))."
    }
    return note
  }

  private func problemCard(_ problem: ChatModel.Problem) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      Label(problem.title, systemImage: "exclamationmark.octagon.fill")
        .font(.title3)
        .foregroundStyle(.red)
      Text(problem.detail)
        .textSelection(.enabled)
        .fixedSize(horizontal: false, vertical: true)
      HStack {
        Button("Open another file…") { model.chooseFile() }
        if problem.canRetry {
          Button("New conversation") { model.newConversation() }
        }
      }
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(RoundedRectangle(cornerRadius: 10).fill(Color.red.opacity(0.07)))
  }
}

// MARK: Composer

private struct Composer: View {
  @ObservedObject var model: ChatModel
  @FocusState private var focused: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack(alignment: .bottom, spacing: 10) {
        TextField(placeholder, text: $model.draft, axis: .vertical)
          .textFieldStyle(.plain)
          .font(.system(size: 15))
          .lineLimit(1...6)
          .focused($focused)
          .onSubmit { model.send() }
          .padding(10)
          .background(RoundedRectangle(cornerRadius: 8).strokeBorder(Color.secondary.opacity(0.35)))
          .disabled(!model.fileIsUsable)
        if model.phase == .answering {
          Button("Stop") { model.stopAnswer() }
            .controlSize(.large)
            .keyboardShortcut(".", modifiers: .command)
        } else {
          Button("Send") { model.send() }
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(model.phase != .ready || model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
      }
      HStack {
        Button("New conversation") { model.newConversation() }
          .disabled(!model.canStartNewConversation)
        Text("A new conversation builds a new engine, so nothing carries over from this one.")
          .font(.caption)
          .foregroundStyle(.secondary)
        Spacer()
      }
    }
    .padding(14)
    .onAppear { focused = true }
  }

  private var placeholder: String {
    if model.fileIsUsable, let name = model.file?.displayName { return "Ask \(name) anything" }
    return "Open a model to start"
  }
}

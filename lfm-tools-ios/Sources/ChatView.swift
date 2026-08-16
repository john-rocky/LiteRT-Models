import FoundationModels
import LiteRTLM
import SwiftUI

@available(iOS 27.0, *)
@main
struct LFMToolsApp: App {
  var body: some Scene {
    WindowGroup { ChatView() }
  }
}

@available(iOS 27.0, *)
struct ChatView: View {
  @State private var chat = ChatModel()
  @State private var input = ""
  @State private var showingTools = false

  var body: some View {
    NavigationStack {
      VStack(spacing: 0) {
        statusBar
        transcript
        composer
      }
      .navigationTitle("LFM Tools")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) {
          Button { showingTools = true } label: { Image(systemName: "wrench.and.screwdriver") }
        }
      }
      .sheet(isPresented: $showingTools) { ToolSheet(chat: chat) }
    }
  }

  // MARK: Pieces

  @ViewBuilder private var statusBar: some View {
    switch chat.status {
    case .ready(let name):
      label("\(name) · \(chat.tools.count) tools", systemImage: "checkmark.circle", tint: .green)
    case .loading(let name):
      label("loading \(name)…", systemImage: "hourglass", tint: .orange)
    case .failed(let why):
      label(why, systemImage: "exclamationmark.triangle", tint: .red)
    case .noModel:
      ModelPicker(chat: chat)
    }
  }

  private func label(_ text: String, systemImage: String, tint: Color) -> some View {
    HStack(spacing: 6) {
      Image(systemName: systemImage)
      Text(text).lineLimit(1)
      Spacer()
    }
    .font(.caption)
    .foregroundStyle(tint)
    .padding(.horizontal)
    .padding(.vertical, 6)
    .background(.bar)
  }

  private var transcript: some View {
    ScrollViewReader { proxy in
      ScrollView {
        LazyVStack(alignment: .leading, spacing: 10) {
          ForEach(chat.lines) { line in
            LineView(line: line).id(line.id)
          }
          if chat.thinking {
            HStack(spacing: 8) {
              ProgressView()
              Text("thinking on-device…").font(.caption).foregroundStyle(.secondary)
            }
            .id("thinking")
          }
        }
        .padding()
      }
      .onChange(of: chat.lines.count) {
        withAnimation { proxy.scrollTo(chat.lines.last?.id, anchor: .bottom) }
      }
    }
  }

  private var composer: some View {
    HStack(spacing: 8) {
      TextField("Ask it to do something", text: $input, axis: .vertical)
        .textFieldStyle(.roundedBorder)
        .lineLimit(1...4)
        .disabled(chat.thinking)
        .onSubmit(send)
      Button(action: send) { Image(systemName: "arrow.up.circle.fill").font(.title2) }
        .disabled(input.isEmpty || chat.thinking)
    }
    .padding()
    .background(.bar)
  }

  private func send() {
    let text = input
    input = ""
    Task { await chat.send(text) }
  }
}

@available(iOS 27.0, *)
private struct LineView: View {
  let line: ChatModel.Line

  var body: some View {
    switch line.kind {
    case .user:
      HStack {
        Spacer(minLength: 40)
        Text(line.text)
          .padding(10)
          .background(Color.accentColor.opacity(0.15), in: .rect(cornerRadius: 12))
      }
    case .assistant:
      Text(line.text)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.12), in: .rect(cornerRadius: 12))
    case .tool:
      // The point of the demo: what the model actually reached for, and what
      // came back. Shown inline rather than in a log nobody opens.
      VStack(alignment: .leading, spacing: 2) {
        HStack(spacing: 6) {
          Image(systemName: "wrench.adjustable")
          Text(line.text).fontWeight(.medium)
        }
        if let detail = line.detail, !detail.isEmpty {
          Text(detail).font(.caption).foregroundStyle(.secondary)
        }
      }
      .font(.caption)
      .padding(8)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color.green.opacity(0.10), in: .rect(cornerRadius: 8))
    case .error:
      Text(line.text)
        .font(.caption)
        .foregroundStyle(.red)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
  }
}

@available(iOS 27.0, *)
private struct ModelPicker: View {
  let chat: ChatModel
  @State private var models = ChatModel.availableModels()

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if models.isEmpty {
        Text("No .litertlm in the app's Documents folder.")
          .font(.callout)
        Text(
          "Copy one in with Files.app or "
            + "`xcrun devicectl device copy to --domain-type appDataContainer`."
        )
        .font(.caption).foregroundStyle(.secondary)
        Button("Look again") { models = ChatModel.availableModels() }
      } else {
        Picker("Backend", selection: Binding(get: { chat.backend }, set: { chat.backend = $0 })) {
          Text("CPU").tag(Backend.cpu())
          Text("GPU").tag(Backend.gpu)
        }
        .pickerStyle(.segmented)
        ForEach(models, id: \.self) { url in
          Button(url.lastPathComponent) { Task { await chat.load(url) } }
        }
      }
    }
    .padding()
  }
}

@available(iOS 27.0, *)
private struct ToolSheet: View {
  let chat: ChatModel
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    NavigationStack {
      List {
        Section {
          toggle("ambient", "No permission needed", ToolBox.ambient)
          toggle("actions", "Changes something", ToolBox.actions)
          toggle("personal", "Asks permission", ToolBox.personal)
        } footer: {
          Text("Changing the set starts a new conversation; the model is kept loaded.")
        }
        ForEach(chat.tools, id: \.name) { tool in
          VStack(alignment: .leading, spacing: 2) {
            Text(tool.name).font(.body.monospaced())
            Text(tool.description).font(.caption).foregroundStyle(.secondary)
          }
        }
      }
      .navigationTitle("\(chat.tools.count) tools")
      .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
    }
  }

  private func toggle(_ key: String, _ title: String, _ tools: [any Tool]) -> some View {
    Toggle(
      isOn: Binding(
        get: { chat.enabledGroups.contains(key) },
        set: { on in
          if on { chat.enabledGroups.insert(key) } else { chat.enabledGroups.remove(key) }
          chat.startSession()
        })
    ) {
      VStack(alignment: .leading) {
        Text(title)
        Text(ToolBox.summary(for: tools)).font(.caption2).foregroundStyle(.secondary)
      }
    }
  }
}

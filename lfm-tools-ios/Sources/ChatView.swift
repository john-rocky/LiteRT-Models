import FoundationModels
import LiteRTLM
import Photos
import PhotosUI
import LiteRTLMFoundationModels
import SwiftUI

@available(iOS 27.0, *)
@main
struct LFMToolsApp: App {
  var body: some Scene {
    WindowGroup {
      // --autorun is the recorded demo: one beat at a time, full screen. The
      // chat stays for driving it by hand.
      if CommandLine.arguments.contains("--autorun") {
        StageView()
      } else {
        ChatView()  // also the host for --export-video
      }
    }
  }
}

@available(iOS 27.0, *)
struct ChatView: View {
  @State private var chat = ChatModel()
  @State private var voice = VoiceInput()
  @State private var input = ""
  @State private var showingTools = false
  @State private var pickedPhoto: PhotosPickerItem?

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
        ToolbarItem(placement: .topBarLeading) {
          Button { Task { await chat.runScript() } } label: { Image(systemName: "play.circle") }
            .disabled(chat.thinking)
        }
      }
      .sheet(isPresented: $showingTools) { ToolSheet(chat: chat) }
      .task { await autostart() }
    }
  }

  /// With one model on the device there is nothing to choose, so load it. With
  /// `--autorun` (how the demo is recorded) the script starts as soon as it is
  /// ready — no taps, so the recording is the app working rather than someone
  /// operating it.
  /// `--bench` turns the app into a measuring harness: every bundle in
  /// Documents, on both backends, with the same token counts as the desktop CLI
  /// so the numbers line up with it.
  private func runBenchmarks() async {
    for url in ChatModel.availableModels() {
      for backend in [Backend.cpu(), .gpu] {
        let label = "\(url.lastPathComponent) \(backend.rawValue)"
        do {
          let info = try await benchmark(
            modelPath: url.path, backend: backend, prefillTokens: 256, decodeTokens: 256,
            cacheDir: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?.path)
          print(
            "BENCH \(label) prefill=\(String(format: "%.2f", info.lastPrefillTokensPerSecond)) "
              + "decode=\(String(format: "%.2f", info.lastDecodeTokensPerSecond)) "
              + "init=\(String(format: "%.2f", info.initTimeInSecond))s "
              + "ttft=\(String(format: "%.3f", info.timeToFirstTokenInSecond))s")
        } catch {
          print("BENCH \(label) FAILED: \(error)")
        }
      }
    }
    print("BENCH done")
  }

  /// `--export-video` copies the newest video out of the photo library into the
  /// app's Documents, where `devicectl device copy from` can reach it. The
  /// screen recording itself has to be started from Control Center — neither
  /// devicectl nor AVFoundation can record this device — but getting the file
  /// back to the Mac should not also be manual.
  private func exportNewestVideo() async {
    let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    guard status == .authorized || status == .limited else {
      print("EXPORT no photo permission")
      return
    }
    let options = PHFetchOptions()
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = 1
    guard let asset = PHAsset.fetchAssets(with: .video, options: options).firstObject else {
      print("EXPORT no video in the library")
      return
    }
    let resources = PHAssetResource.assetResources(for: asset)
    guard let resource = resources.first(where: { $0.type == .video }) ?? resources.first else {
      print("EXPORT no resource")
      return
    }
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let destination = documents.appendingPathComponent("screen-recording.mov")
    try? FileManager.default.removeItem(at: destination)
    let options2 = PHAssetResourceRequestOptions()
    options2.isNetworkAccessAllowed = true
    do {
      try await PHAssetResourceManager.default().writeData(
        for: resource, toFile: destination, options: options2)
      let size = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
      print("EXPORT wrote screen-recording.mov (\(size) bytes)")
    } catch {
      print("EXPORT failed: \(error)")
    }
  }

  private func autostart() async {
    if CommandLine.arguments.contains("--export-video") {
      await exportNewestVideo()
      return
    }
    if CommandLine.arguments.contains("--toolbench") {
      await BenchRunner.run()
      return
    }
    if CommandLine.arguments.contains("--photocheck") {
      await PhotoCheck.run()
      return
    }
    if CommandLine.arguments.contains("--bench") {
      await runBenchmarks()
      return
    }
    chat.startTrace()
    guard case .noModel = chat.status else { return }
    // `--model apple` skips the bundle entirely: Apple's model, every tool,
    // the mic — the setup for trying things out, not measuring them.
    if let flag = CommandLine.arguments.firstIndex(of: "--model"),
      CommandLine.arguments.indices.contains(flag + 1),
      CommandLine.arguments[flag + 1].lowercased() == "apple"
    {
      chat.loadSystemModel()
      return
    }
    guard let newest = ChatModel.availableModels().first else { return }
    await chat.load(newest)
    if CommandLine.arguments.contains("--autorun") {
      await chat.runScript()
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
            VStack(alignment: .leading, spacing: 6) {
              HStack(spacing: 8) {
                ProgressView()
                Text(
                  chat.lastRate > 0
                    ? "generating on-device · \(Int(chat.lastRate)) chars/s"
                    : "generating on-device…"
                )
                .font(.caption).foregroundStyle(.secondary)
              }
              if !chat.live.isEmpty {
                // Only the tail: a router pass can run to hundreds of
                // characters, and what matters on screen is what it is
                // producing now.
                Text(String(chat.live.suffix(320)))
                  .font(.system(.footnote, design: .monospaced))
                  .foregroundStyle(.green)
                  .frame(maxWidth: .infinity, alignment: .leading)
              }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .id("thinking")
          }
        }
        .padding()
      }
      .onChange(of: chat.lines.count) {
        withAnimation { proxy.scrollTo(chat.lines.last?.id, anchor: .bottom) }
      }
      // Tokens arrive faster than lines do; without this the green text runs
      // off the bottom of the screen and the run looks stalled.
      .onChange(of: chat.live) {
        proxy.scrollTo("thinking", anchor: .bottom)
      }
    }
  }

  private var composer: some View {
    VStack(alignment: .leading, spacing: 4) {
      if let problem = voice.problem {
        Text(problem).font(.caption).foregroundStyle(.red)
      }
      if let thumbnail = chat.attachedThumbnail {
        // The photo about to go in, with a way out. A message can be the
        // photo alone.
        HStack(spacing: 6) {
          Image(uiImage: thumbnail)
            .resizable().scaledToFill()
            .frame(width: 44, height: 44)
            .clipShape(.rect(cornerRadius: 8))
          Text("photo attached").font(.caption).foregroundStyle(.secondary)
          Button { chat.detachImage() } label: { Image(systemName: "xmark.circle.fill") }
            .foregroundStyle(.secondary)
        }
      }
      HStack(spacing: 8) {
        Menu {
          Button {
            Task { if let image = try? await PhotoBox.latestImage() { chat.attach(image) } }
          } label: {
            Label("Latest photo", systemImage: "photo.on.rectangle")
          }
          // Wrapped in a Menu item via the picker's own label; the picker
          // sheet opens on tap.
          PhotosPicker(selection: $pickedPhoto, matching: .images) {
            Label("Choose a photo…", systemImage: "photo.stack")
          }
        } label: {
          Image(systemName: "paperclip").font(.title2)
        }
        .disabled(chat.thinking)
        TextField(
          voice.listening ? "Listening…" : "Ask it to do something",
          text: composerText, axis: .vertical
        )
        .textFieldStyle(.roundedBorder)
        .lineLimit(1...4)
        .disabled(chat.thinking || voice.listening)
        .onSubmit(send)
        Button(action: toggleVoice) {
          Image(systemName: voice.listening ? "mic.fill" : "mic")
            .font(.title2)
            .foregroundStyle(voice.listening ? Color.red : Color.accentColor)
            .symbolEffect(.pulse, isActive: voice.listening)
        }
        .disabled(chat.thinking)
        Button(action: send) { Image(systemName: "arrow.up.circle.fill").font(.title2) }
          .disabled(
            (input.isEmpty && chat.attachedThumbnail == nil) || chat.thinking || voice.listening)
      }
    }
    .padding()
    .background(.bar)
    // The words appear in the composer as they are spoken — the recognizer
    // editing itself mid-sentence is the visible half of voice input.
    .onChange(of: voice.heard) {
      if voice.listening { input = voice.heard }
    }
    .onChange(of: pickedPhoto) {
      guard let item = pickedPhoto else { return }
      Task {
        if let data = try? await item.loadTransferable(type: Data.self),
          let image = UIImage(data: data)?.cgImage
        {
          chat.attach(image)
        }
        pickedPhoto = nil
      }
    }
  }

  /// One button, two states: start a take, or finish it and leave the final
  /// transcript in the composer for the same send path typing uses.
  private func toggleVoice() {
    Task {
      if voice.listening {
        input = await voice.stop()
      } else {
        await voice.start()
      }
    }
  }

  /// While the script is running the composer shows the line about to be sent,
  /// so a recording shows a question going in rather than answers appearing.
  private var composerText: Binding<String> {
    if let scripted = chat.scriptedPrompt {
      return .constant(scripted)
    }
    return $input
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
        VStack(alignment: .trailing, spacing: 6) {
          if let image = line.image {
            Image(uiImage: image)
              .resizable().scaledToFit()
              .frame(maxHeight: 180)
              .clipShape(.rect(cornerRadius: 10))
          }
          if !line.text.isEmpty { Text(line.text) }
        }
        .padding(10)
        .background(Color.accentColor.opacity(0.15), in: .rect(cornerRadius: 12))
      }
    case .assistant:
      VStack(alignment: .leading, spacing: 8) {
        Text(line.text)
        if let artifact = line.artifact {
          ArtifactView(artifact: artifact)
            .transition(.asymmetric(
              insertion: .scale(scale: 0.92, anchor: .top).combined(with: .opacity),
              removal: .opacity))
        }
      }
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
      // Always first: no file to find, answers in a second, and every tool
      // in the sheet works with it. The bundles below are the same session
      // on open weights.
      Button {
        chat.loadSystemModel()
      } label: {
        Label(ChatModel.systemModelName, systemImage: "apple.logo")
      }
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
        Section("Model") {
          // Switching who answers, mid-session. Same tools either way; the
          // point of the demo is that the session does not care.
          Button {
            chat.loadSystemModel()
            dismiss()
          } label: {
            Label(ChatModel.systemModelName, systemImage: "apple.logo")
          }
          ForEach(ChatModel.availableModels(), id: \.self) { url in
            Button(url.lastPathComponent) {
              Task { await chat.load(url) }
              dismiss()
            }
          }
        }
        Section {
          toggle("ambient", "No permission needed", ToolBox.ambient)
          toggle("actions", "Changes something", ToolBox.actions)
          toggle("personal", "Asks permission", ToolBox.personal)
          toggle("photo", "Photo editing", ToolBox.photoEditing)
          toggle("compound", "One call, several things", ToolBox.compound)
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

  private func toggle(_ key: String, _ title: String, _ tools: [any FoundationModels.Tool]) -> some View {
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

import AppKit
import SwiftUI

struct ChatApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
  @StateObject private var model = ChatModel()

  var body: some Scene {
    // One window: every window would need its own 25 GB engine.
    Window("LiteRT-LM Chat", id: "chat") {
      ChatView(model: model)
        .background(WindowConfigurator())
        .task { await Launch.run(model: model) }
    }
    .commands {
      CommandGroup(replacing: .newItem) {
        Button("Open Model…") { model.chooseFile() }
          .keyboardShortcut("o")
        Button("New Conversation") { model.newConversation() }
          .keyboardShortcut("n")
          .disabled(!model.canStartNewConversation)
      }
    }
  }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    true
  }
}

/// Command-line options:
///   --model <path>        open this file at launch
///   --autoplay <script>   run a recorder script (Autoplay.swift) on --model
///   --run-out <path>      where the recorder writes its run JSON
///   --snapshot-dir <dir>  where the recorder saves pictures of the window
/// and user defaults given as `-key value`:
///   -logDirectory <dir>   where engine logs go (default ~/Library/Logs/LiteRTLMChat)
///   -runtimeLog info      LiteRT-LM's log level in the engine log (verbose, info, warning, error)
///   -windowFrame "x,y,w,h"  window position and size in points from the top-left of the main screen
///   -windowFloating YES   keep the window above other windows (for screen recordings)
///   -assumePhysicalMemoryGB <n>  pretend the Mac has n GB of memory
@MainActor
enum Launch {
  private static var handled = false

  static func run(model: ChatModel) async {
    guard !handled else { return }
    handled = true
    let modelPath = value(after: "--model")
    if let script = value(after: "--autoplay") {
      guard let modelPath else {
        FileHandle.standardError.write(Data("--autoplay needs --model <path>\n".utf8))
        return
      }
      await Autoplay.run(
        model: model, scriptPath: script, modelPath: modelPath, outPath: value(after: "--run-out"))
    } else if let modelPath {
      model.open(URL(fileURLWithPath: modelPath))
    }
  }

  static func value(after flag: String) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else { return nil }
    return arguments[index + 1]
  }
}

/// Applies `-windowFrame` and `-windowFloating` to the window once it exists.
struct WindowConfigurator: NSViewRepresentable {
  func makeNSView(context: Context) -> NSView {
    let view = NSView()
    DispatchQueue.main.async {
      guard let window = view.window else { return }
      let defaults = UserDefaults.standard
      if let spec = defaults.string(forKey: "windowFrame"), let screen = NSScreen.screens.first {
        let numbers = spec.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        if numbers.count == 4 {
          let frame = NSRect(
            x: numbers[0], y: screen.frame.maxY - numbers[1] - numbers[3], width: numbers[2], height: numbers[3])
          window.setFrame(frame, display: true)
        }
      }
      if defaults.bool(forKey: "windowFloating") {
        window.level = .floating
      }
    }
    return view
  }

  func updateNSView(_ nsView: NSView, context: Context) {}

  /// The window's frame in points from the top-left of the main screen (what
  /// `screencapture -R` takes).
  @MainActor static func frameFromTopLeft() -> [Double]? {
    guard let window = NSApp.windows.first(where: { $0.isVisible }), let screen = NSScreen.screens.first
    else { return nil }
    let frame = window.frame
    return [frame.minX, screen.frame.maxY - frame.maxY, frame.width, frame.height]
  }
}

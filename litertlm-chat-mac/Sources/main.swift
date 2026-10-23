import Foundation

// One binary, two modes: the chat window, and the headless engine host that the window
// starts once per conversation (EngineHost.swift says why).
if CommandLine.arguments.contains(EngineHost.flag) {
  EngineHost.main(arguments: CommandLine.arguments)
} else {
  // Writing to an engine host that has already ended must not end the app.
  signal(SIGPIPE, SIG_IGN)
  MainActor.assumeIsolated {
    ChatApp.main()
  }
}

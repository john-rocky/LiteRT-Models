// Where the app keeps its files: cases, run logs, saved documents, exports.
import Foundation

/// One place to answer "where do files go", because the honest answer differs
/// by platform. On iOS it is the sandbox Documents folder — visible in
/// Files.app, reachable with `devicectl device copy`. On the Mac (Catalyst,
/// deliberately unsandboxed) the real ~/Documents is TCC-protected: the first
/// touch pops a consent dialog, and a bench driven over SSH has nobody in
/// front of the screen to click it. ~/Library/Application Support is not
/// TCC-protected, so the Mac build lives there and the scripts read and write
/// it freely.
@available(iOS 27.0, *)
enum AppFiles {
  static let documents: URL = {
    #if targetEnvironment(macCatalyst)
      let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("LFMTools", isDirectory: true)
      try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
      return base
    #else
      return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    #endif
  }()
}

// Asking, undoing, confirming: the three tools every app has and no pack had.
//
// ask_user makes the ask-back routable: instead of hoping the model asks in
// prose when a required detail is missing (measured: it invents, especially
// in Japanese), asking becomes a call the bench can score. undo_last is the
// menu item every real app puts first; each pack's box keeps a small history
// of value snapshots and steps back. The confirm pattern lives on the
// destructive tools themselves (refund, checkout, delete): a `confirm`
// argument the model must leave false until the user has said yes — whether
// it does is a measurement, not an assumption.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
struct AskUserTool: Tool {
  let name = "ask_user"
  let description =
    "Ask the user for a detail the request is missing. Use this instead of inventing a value — the words of a caption or note, which record is meant, an amount."
  @Generable struct Arguments {
    @Guide(description: "The question, short and specific.") var question: String
  }
  func call(arguments: Arguments) async throws -> String {
    "asked the user: \"\(arguments.question)\" — their answer will come in the next message"
  }
}

/// One undo tool, aimed at whichever pack's box is on stage. Every box keeps
/// its state as a value; undo pops the last snapshot.
@available(iOS 27.0, *)
struct UndoLastTool: Tool {
  enum Target: Sendable {
    case video, store, audio, docs, shopping, money, inbox, crm, pm, library
  }
  let target: Target
  let name = "undo_last"
  // "do not reverse it by hand" is load-bearing: asked to undo, the model
  // re-set four faders to their old values and removed the effect itself
  // (Mac, 2026-08-19) — the primitives instinct again.
  let description = "Undo the last change. Use this for \"undo that\" — do not reverse the change by hand."

  func call(arguments: NoArguments) async throws -> String {
    switch target {
    case .video: return VideoEditBox.shared.undoLast()
    case .store: return StoreBox.shared.undoLast()
    case .audio: return AudioBox.shared.undoLast()
    case .docs: return DocBox.shared.undoLast()
    case .shopping: return ShopBox.shared.undoLast()
    case .money: return MoneyBox.shared.undoLast()
    case .inbox: return InboxBox.shared.undoLast()
    case .crm: return CrmBox.shared.undoLast()
    case .pm: return IssueBox.shared.undoLast()
    case .library: return PhotoLibraryBox.shared.undoLast()
    }
  }
}

/// A tiny history of value snapshots — enough for "undo that", not a redo
/// stack. Boxes own one per undoable state value.
@available(iOS 27.0, *)
final class UndoStack<State: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private var stack: [(State, String)] = []

  /// Remember the state as it is now, with a word for what is about to
  /// change ("fade", "prices").
  func push(_ state: State, _ what: String) {
    lock.lock()
    stack.append((state, what))
    if stack.count > 20 { stack.removeFirst() }
    lock.unlock()
  }

  func pop() -> (State, String)? {
    lock.lock()
    defer { lock.unlock() }
    return stack.popLast()
  }

  /// What the last change was, for the state line — "undo that" needs a
  /// referent the model can read, or it re-derives one (and got it wrong:
  /// 「今のを取り消して」 became three deletes on the Mac, 2026-08-19).
  func peekWhat() -> String? {
    lock.lock()
    defer { lock.unlock() }
    return stack.last?.1
  }
}

// The inbox pack: a mail app's triage (the Spark type), said out loud.
//
// Fifteen canned messages, none real. The finders select (unread, a sender,
// the newsletters), the actions clear them out (archive, snooze, flag), and
// the one composer writes a reply draft from a gist — the app never sends.
// "Archive the newsletters" is the chain the pack is for: find them, then
// act on what was found.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class InboxBox: @unchecked Sendable {
  static let shared = InboxBox()

  struct Message: Sendable {
    let id: Int
    let from: String
    let subject: String
    let snippet: String
    let daysAgo: Int
    var unread: Bool
    var flagged = false
    var archived = false
    var snoozedUntil: String?
    let newsletter: Bool
  }

  private let lock = NSLock()
  private var messages: [Message] = InboxData.messages
  private var selection: [Int] = []
  private var selectionHow = ""
  private var drafts: [(to: String, body: String)] = []
  private var pendingConfirmation: String?
  private let history = UndoStack<([Message], [Int], String)>()

  private func pushHistory(_ what: String) {
    history.push((messages, selection, selectionHow), what)
  }

  func undoLast() -> String {
    guard let (snap, what) = history.pop() else { return "nothing to undo" }
    sync { (messages, selection, selectionHow) = snap }
    post()
    return "undid the last change (\(what))"
  }

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  // MARK: The state the model reads

  func describe() -> String {
    let (rows, selection, how) = sync { (messages, selection, selectionHow) }
    let inbox = rows.filter { !$0.archived && $0.snoozedUntil == nil }
    let unread = inbox.filter(\.unread).count
    var line = "Inbox: \(inbox.count) messages, \(unread) unread."
    if let pending = sync({ pendingConfirmation }) { line += " Awaiting confirmation: \(pending)." }
    let listed = inbox.sorted { $0.daysAgo < $1.daysAgo }.prefix(6).map {
      "#\($0.id) \($0.from) — \"\($0.subject)\"\($0.unread ? " (unread)" : "")\($0.flagged ? " ⚑" : "")"
    }
    line += " Newest: " + listed.joined(separator: "; ") + "."
    if selection.isEmpty {
      line += " Selection: none — list, search or filter first, then act."
    } else {
      let selected = rows.filter { selection.contains($0.id) }
      line += " Selection: \(selected.count) from \(how): "
        + selected.prefix(5).map { "#\($0.id) \($0.from)" }.joined(separator: "; ")
        + (selected.count > 5 ? ", and \(selected.count - 5) more" : "") + "."
    }
    return line
  }

  // MARK: The panel

  func snapshot() -> TablePanel {
    let (rows, selection, how) = sync { (messages, selection, selectionHow) }
    let inbox = rows.filter { !$0.archived && $0.snoozedUntil == nil }.sorted { $0.daysAgo < $1.daysAgo }
    let shown = selection.isEmpty ? inbox : rows.filter { selection.contains($0.id) }
    let unread = inbox.filter(\.unread).count
    return TablePanel(
      title: selection.isEmpty ? "Inbox" : "\(shown.count) selected — \(how)",
      columns: ["", "From", "Subject", "When"],
      rows: shown.map {
        [($0.unread ? "●" : "") + ($0.flagged ? "⚑" : ""), $0.from, $0.subject, MoneyBox.when($0.daysAgo)]
      },
      totalRows: shown.count,
      overview: "\(inbox.count) in inbox · \(unread) unread")
  }

  private func post() {
    let panel = snapshot()
    ArtifactBox.shared.post(.table(
      title: panel.title, columns: panel.columns, rows: Array(panel.rows.prefix(8))))
  }

  // MARK: Finding (each replaces the selection)

  private func select(_ rows: [Message], how: String) -> String {
    sync {
      selection = rows.map(\.id)
      selectionHow = how
    }
    post()
    guard !rows.isEmpty else { return "no messages match \(how)" }
    let listed = rows.prefix(8).map {
      "#\($0.id) \($0.from) — \"\($0.subject)\" (\(MoneyBox.when($0.daysAgo))\($0.unread ? ", unread" : ""))"
    }
    return "\(rows.count) message\(rows.count == 1 ? "" : "s") (\(how)):\n" + listed.joined(separator: "\n")
      + (rows.count > 8 ? "\n… and \(rows.count - 8) more" : "")
  }

  func list(filter: String) -> String {
    let inbox = sync { messages }.filter { !$0.archived && $0.snoozedUntil == nil }
      .sorted { $0.daysAgo < $1.daysAgo }
    switch filter.lowercased() {
    case "unread": return select(inbox.filter(\.unread), how: "unread")
    case "flagged": return select(inbox.filter(\.flagged), how: "flagged")
    case "newsletters": return select(inbox.filter(\.newsletter), how: "newsletters")
    case "read": return select(inbox.filter { !$0.unread }, how: "already read")
    default: return select(inbox, how: "all")
    }
  }

  func search(_ query: String) -> String {
    let needle = query.lowercased()
    let rows = sync { messages }.filter { !$0.archived && $0.snoozedUntil == nil }
      .filter {
        $0.from.lowercased().contains(needle) || $0.subject.lowercased().contains(needle)
          || $0.snippet.lowercased().contains(needle)
      }
      .sorted { $0.daysAgo < $1.daysAgo }
    return select(rows, how: "search \"\(query)\"")
  }

  func read(_ id: Int) -> String {
    let message: Message? = sync {
      guard let index = messages.firstIndex(where: { $0.id == id }) else { return nil }
      messages[index].unread = false
      return messages[index]
    }
    guard let message else { return "there is no message #\(id)" }
    post()
    return "#\(message.id) from \(message.from), \(MoneyBox.when(message.daysAgo)) — \"\(message.subject)\": \(message.snippet)"
  }

  // MARK: Acting on the selection

  private func onSelection(_ what: String, _ change: (inout Message) -> Void) -> String {
    let ids = sync { selection }
    guard !ids.isEmpty else { return "nothing is selected — list, search or filter first" }
    sync {
      pushHistory(what)
      for index in messages.indices where ids.contains(messages[index].id) {
        change(&messages[index])
      }
      selection = []
      selectionHow = ""
    }
    post()
    return "\(what) \(ids.count) message\(ids.count == 1 ? "" : "s")"
  }

  func archive() -> String { onSelection("archived") { $0.archived = true } }
  func markRead() -> String { onSelection("marked read") { $0.unread = false } }
  func flag() -> String { onSelection("flagged") { $0.flagged = true } }

  func snooze(until when: String) -> String {
    let label = ["tonight", "tomorrow", "next_week"].contains(when.lowercased())
      ? when.lowercased() : "tomorrow"
    return onSelection("snoozed until \(label.replacingOccurrences(of: "_", with: " ")):") {
      $0.snoozedUntil = label
    }
  }

  // MARK: Writing

  func draftReply(to id: Int, gist: String) -> String {
    let message = sync { messages.first { $0.id == id } }
    guard let message else { return "there is no message #\(id)" }
    let firstName = message.from.split(separator: " ").first.map(String.init) ?? message.from
    let body = "Hi \(firstName),\n\n\(gist)\n\nBest,\nDaisuke"
    sync { drafts.append((to: message.from, body: body)) }
    ArtifactBox.shared.post(.note(text: "Draft to \(message.from):\n\(body)"))
    return "draft saved, replying to \(message.from) re \"\(message.subject)\":\n\(body)"
  }

  /// Delete is the destructive one (archive keeps the mail): behind the
  /// confirm pattern.
  func delete(_ id: Int, confirmed: Bool) -> String {
    let message = sync { messages.first { $0.id == id } }
    guard let message else { return "there is no message #\(id)" }
    guard confirmed else {
      sync { pendingConfirmation = "delete message #\(id) from \(message.from)" }
      return "deleting #\(id) from \(message.from) (\"\(message.subject)\") is permanent — ask the user to confirm, then call delete_message again with confirm true"
    }
    sync {
      pendingConfirmation = nil
      pushHistory("delete")
      messages.removeAll { $0.id == id }
    }
    post()
    return "deleted the message from \(message.from)"
  }

  func unsubscribe(_ id: Int) -> String {
    let message: Message? = sync {
      guard let index = messages.firstIndex(where: { $0.id == id }) else { return nil }
      pushHistory("unsubscribe")
      messages[index].archived = true
      return messages[index]
    }
    guard let message else { return "there is no message #\(id)" }
    guard message.newsletter else {
      return "#\(id) from \(message.from) is not a newsletter — archived it instead"
    }
    post()
    return "unsubscribed from \(message.from) and archived the message"
  }
}

/// Fifteen messages, frozen; none is real. Three newsletters (the sweep), an
/// invoice (the snooze), a question from a colleague (the reply), a message
/// from the landlord (the flag).
enum InboxData {
  static let messages: [InboxBox.Message] = [
    .init(id: 1, from: "Hana Kim", subject: "Report deadline", snippet: "Could you send the Q3 report this week?", daysAgo: 0, unread: true, newsletter: false),
    .init(id: 2, from: "Kanda Goods", subject: "Invoice #4471", snippet: "Your invoice for August is attached. Payment due Friday.", daysAgo: 0, unread: true, newsletter: false),
    .init(id: 3, from: "TechWeekly", subject: "This week in AI", snippet: "The five stories everyone is talking about.", daysAgo: 0, unread: true, newsletter: true),
    .init(id: 4, from: "Sakura Property", subject: "Building maintenance notice", snippet: "Water will be off Tuesday 9:00–12:00.", daysAgo: 1, unread: true, newsletter: false),
    .init(id: 5, from: "Deal Digest", subject: "Weekend sale picks", snippet: "20% off everything you didn't need.", daysAgo: 1, unread: true, newsletter: true),
    .init(id: 6, from: "Goro Ito", subject: "Lunch next week?", snippet: "Are you free Tuesday or Wednesday?", daysAgo: 1, unread: false, newsletter: false),
    .init(id: 7, from: "Runner's Letter", subject: "Your monthly training tips", snippet: "Building your base the easy way.", daysAgo: 2, unread: true, newsletter: true),
    .init(id: 8, from: "Aoi Tanaka", subject: "Re: photos from Saturday", snippet: "These came out great, thank you!", daysAgo: 2, unread: false, newsletter: false),
    .init(id: 9, from: "City Office", subject: "Tax payment reminder", snippet: "The second installment is due at the end of the month.", daysAgo: 3, unread: true, newsletter: false),
    .init(id: 10, from: "Ben Carter", subject: "Slides for Thursday", snippet: "Draft attached — comments welcome.", daysAgo: 3, unread: false, newsletter: false),
    .init(id: 11, from: "Emi Sato", subject: "Book club", snippet: "Next pick is out — meeting on the 30th.", daysAgo: 4, unread: false, newsletter: false),
    .init(id: 12, from: "Cloud Notes", subject: "Your plan renews soon", snippet: "Your annual plan renews on Sep 1.", daysAgo: 4, unread: true, newsletter: false),
    .init(id: 13, from: "Mio Hayashi", subject: "Thanks!", snippet: "The parcel arrived safely.", daysAgo: 5, unread: false, newsletter: false),
    .init(id: 14, from: "Farah Khan", subject: "Intro: Kenji from Osaka", snippet: "You two should talk about edge ML.", daysAgo: 6, unread: false, newsletter: false),
    .init(id: 15, from: "Daniel Ross", subject: "Draft agenda", snippet: "For the sync on Friday — anything to add?", daysAgo: 6, unread: false, newsletter: false),
  ]
}

// MARK: - Tools (the mail app's gestures, in its words)

@available(iOS 27.0, *)
struct ListInboxTool: Tool {
  let name = "list_inbox"
  let description = "List inbox messages; the list becomes the selection."
  @Generable struct Arguments {
    @Guide(description: "Which slice.", .anyOf(["all", "unread", "read", "flagged", "newsletters"]))
    var filter: String
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.list(filter: arguments.filter)
  }
}

@available(iOS 27.0, *)
struct SearchMailTool: Tool {
  let name = "search_mail"
  let description = "Find messages by sender, subject or words in them; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Who or what to look for, e.g. Kanda Goods, invoice.") var query: String
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.search(arguments.query)
  }
}

@available(iOS 27.0, *)
struct ReadMessageTool: Tool {
  let name = "read_message"
  let description = "Open one message, by its number."
  @Generable struct Arguments {
    @Guide(description: "The message number, e.g. 2. The inbox list is in the state.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.read(arguments.number)
  }
}

@available(iOS 27.0, *)
struct ArchiveTool: Tool {
  let name = "archive_selection"
  let description = "Archive the selected messages — out of the inbox, not deleted."
  func call(arguments: NoArguments) async throws -> String {
    InboxBox.shared.archive()
  }
}

@available(iOS 27.0, *)
struct MarkReadTool: Tool {
  let name = "mark_read"
  let description = "Mark the selected messages as read."
  func call(arguments: NoArguments) async throws -> String {
    InboxBox.shared.markRead()
  }
}

@available(iOS 27.0, *)
struct FlagMailTool: Tool {
  let name = "flag_selection"
  let description = "Flag the selected messages."
  func call(arguments: NoArguments) async throws -> String {
    InboxBox.shared.flag()
  }
}

@available(iOS 27.0, *)
struct SnoozeTool: Tool {
  let name = "snooze_selection"
  let description = "Hide the selected messages until later; they come back by themselves."
  @Generable struct Arguments {
    @Guide(description: "When they return.", .anyOf(["tonight", "tomorrow", "next_week"])) var until: String
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.snooze(until: arguments.until)
  }
}

@available(iOS 27.0, *)
struct DraftReplyTool: Tool {
  let name = "draft_reply"
  let description = "Write a reply draft to one message. Nothing is sent."
  @Generable struct Arguments {
    @Guide(description: "The message number to reply to.") var number: Int
    @Guide(description: "What the reply should say, in a sentence.") var gist: String
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.draftReply(to: arguments.number, gist: arguments.gist)
  }
}

@available(iOS 27.0, *)
struct DeleteMessageTool: Tool {
  let name = "delete_message"
  let description = "Delete a message permanently. Archive keeps it; this does not."
  @Generable struct Arguments {
    @Guide(description: "The message number.") var number: Int
    @Guide(description: "Pass false unless the user has already said yes to deleting this exact message; false shows them what would be deleted.")
    var confirm: Bool
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.delete(arguments.number, confirmed: arguments.confirm)
  }
}

@available(iOS 27.0, *)
struct UnsubscribeTool: Tool {
  let name = "unsubscribe"
  let description = "Unsubscribe from a newsletter and archive it, by message number."
  @Generable struct Arguments {
    @Guide(description: "The newsletter's message number.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    InboxBox.shared.unsubscribe(arguments.number)
  }
}

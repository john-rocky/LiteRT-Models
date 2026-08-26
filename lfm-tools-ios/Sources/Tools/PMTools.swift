// The PM pack: a Jira board's menu, said out loud, over canned data.
//
// The second business-wing pack (docs/business-packs.md), chosen for P0
// because its tool list is dense with deliberate similar tools: four
// change_* verbs (assign / status / priority / due date) that differ only
// in which column they touch, and close_issue sitting beside
// change_issue_status — the bench's central question, planted on purpose.
//
// Same architecture the CRM pack measured: search_issues makes the
// selection (what "this one" resolves through), every action takes its
// issue as an id argument, and the calendar is frozen — the state pins
// today's date so "Friday" and "next Monday" resolve to absolute dates a
// case can score.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class IssueBox: @unchecked Sendable {
  static let shared = IssueBox()

  struct Issue: Sendable {
    let id: String  // APP-3: the handle the user points with, shown in every row
    let project: String
    var title: String
    var priority: String
    var status: String
    var assignee: String?  // nil = unassigned
    let creator: String
    var due: String  // YYYY-MM-DD, absolute — the calendar is frozen
  }

  private let lock = NSLock()
  private var issues: [Issue] = PmData.issues
  private var comments: [String: [String]] = [:]
  private var selection: [String] = []
  private var selectionHow = ""
  private let history = UndoStack<([Issue], [String: [String]], [String], String)>()

  private func pushHistory(_ what: String) {
    history.push((issues, comments, selection, selectionHow), what)
  }

  func undoLast() -> String {
    guard let (snap, what) = history.pop() else { return "nothing to undo" }
    sync { (issues, comments, selection, selectionHow) = snap }
    post()
    return "undid the last change (\(what))"
  }

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  // MARK: The state the model reads

  /// Counts and handles, never rows — the CRM pack's shape: full rows in
  /// the state let search questions be answered without the finder, and
  /// the finder is what the pack measures. The handles a person points
  /// with — projects, assignees, today's date — are all here; selected
  /// rows carry everything.
  func describe() -> String {
    let (rows, selection, how) = sync { (issues, self.selection, self.selectionHow) }
    let byStatus = PmData.statuses.compactMap { status -> String? in
      let count = rows.filter { $0.status == status }.count
      return count > 0 ? "\(count) \(status)" : nil
    }
    let p1 = rows.filter { $0.priority == "P1" }.count
    let unassigned = rows.filter { $0.assignee == nil }.count
    // The weekday rides with the date: "due by Friday" and "next Monday"
    // are handles the user points with, and r15 measured the model blind
    // without it — "Friday" became due_date_from 2026-08-25, "next
    // Monday" 2026-08-28 (a Friday), in both languages.
    var line = "Today: \(PmData.today) (\(PmData.todayWeekday)). Board: \(rows.count) issues across \(PmData.projects.joined(separator: ", ")) — \(byStatus.joined(separator: ", ")); \(p1) P1."
    line += " Assignees: \(PmData.assignees.joined(separator: ", ")); \(unassigned) unassigned."
    if let last = history.peekWhat() { line += " Last change: \(last)." }
    if selection.isEmpty {
      line += " Selection: none."
    } else {
      let selected = selection.compactMap { id in rows.first { $0.id == id } }
      let names = selected.prefix(6).map(Self.oneLine).joined(separator: "; ")
      let more = selected.count > 6 ? ", and \(selected.count - 6) more" : ""
      line += " Selection: \(selected.count) issue\(selected.count == 1 ? "" : "s") from \(how): \(names)\(more)."
    }
    return line
  }

  /// Every handle a person grabs an issue by, in one selection row.
  /// Internal, not private: the bench's PmEcho renders the same rows.
  static func oneLine(_ i: Issue) -> String {
    "\(i.id) \(i.project) — \"\(i.title)\", \(i.priority), \(i.status), \(i.assignee ?? "unassigned"), due \(i.due)"
  }

  // MARK: The panel

  func snapshot() -> TablePanel {
    let (rows, selection, how) = sync { (issues, self.selection, self.selectionHow) }
    let shown = selection.isEmpty ? rows : selection.compactMap { id in rows.first { $0.id == id } }
    let p1 = rows.filter { $0.priority == "P1" }.count
    let todo = rows.filter { $0.status == "todo" }.count
    return TablePanel(
      title: selection.isEmpty ? "Board" : "\(shown.count) selected — \(how)",
      columns: ["ID", "Title", "Pri", "Status", "Assignee", "Due"],
      rows: shown.map { [$0.id, $0.title, $0.priority, $0.status, $0.assignee ?? "—", $0.due] },
      totalRows: shown.count,
      overview: "\(rows.count) issues · \(p1) P1 · \(todo) todo")
  }

  private func post() {
    let panel = snapshot()
    ArtifactBox.shared.post(.table(
      title: panel.title, columns: panel.columns, rows: Array(panel.rows.prefix(8))))
  }

  // MARK: Finding (replaces the selection)

  func searchIssues(
    project: String?, status: String?, priority: String?, assignee: String?,
    creator: String?, keyword: String?, dueFrom: String?, dueTo: String?
  ) -> String {
    var how: [String] = []
    var rows = sync { issues }
    if let project, !project.isEmpty {
      let want = project.lowercased()
      guard let canonical = PmData.projects.first(where: { $0.lowercased() == want }) else {
        return "no project called \(project) — the projects are \(PmData.projects.joined(separator: ", "))"
      }
      rows = rows.filter { $0.project == canonical }
      how.append("project \(canonical)")
    }
    if let status, !status.isEmpty {
      let want = status.lowercased()
      guard PmData.statuses.contains(want) else {
        return "unknown status; the statuses are \(PmData.statuses.joined(separator: ", "))"
      }
      rows = rows.filter { $0.status == want }
      how.append("status \(want)")
    }
    if let priority, !priority.isEmpty {
      let want = priority.uppercased()
      guard PmData.priorities.contains(want) else {
        return "unknown priority; the priorities are \(PmData.priorities.joined(separator: ", "))"
      }
      rows = rows.filter { $0.priority == want }
      how.append("priority \(want)")
    }
    if let assignee, !assignee.isEmpty {
      let want = assignee.lowercased()
      if want == "unassigned" || want == "none" || want == "nobody" {
        rows = rows.filter { $0.assignee == nil }
        how.append("unassigned")
      } else {
        guard let canonical = PmData.person(assignee) else {
          return "nobody called \(assignee) here — the assignees are \(PmData.assignees.joined(separator: ", "))"
        }
        rows = rows.filter { $0.assignee == canonical }
        how.append("assignee \(canonical)")
      }
    }
    if let creator, !creator.isEmpty {
      guard let canonical = PmData.person(creator) else {
        return "nobody called \(creator) here"
      }
      rows = rows.filter { $0.creator == canonical }
      how.append("creator \(canonical)")
    }
    if let keyword, !keyword.isEmpty {
      let needle = PmData.romaji(keyword)
      rows = rows.filter { $0.title.lowercased().contains(needle) }
      how.append("keyword \"\(keyword)\"")
    }
    // ISO dates compare correctly as strings; the calendar is frozen.
    if let dueFrom, !dueFrom.isEmpty {
      rows = rows.filter { $0.due >= dueFrom }
      how.append("due from \(dueFrom)")
    }
    if let dueTo, !dueTo.isEmpty {
      rows = rows.filter { $0.due <= dueTo }
      how.append("due by \(dueTo)")
    }
    let sorted = rows.sorted { $0.due < $1.due }
    let caption = how.isEmpty ? "all issues" : how.joined(separator: ", ")
    sync {
      selection = sorted.map(\.id)
      selectionHow = caption
    }
    post()
    guard !sorted.isEmpty else { return "no issues match \(caption)" }
    return "\(sorted.count) issue\(sorted.count == 1 ? "" : "s") (\(caption)):\n"
      + sorted.prefix(8).map(Self.oneLine).joined(separator: "\n")
      + (sorted.count > 8 ? "\n… and \(sorted.count - 8) more" : "")
  }

  // MARK: Reading one issue

  func getIssue(_ id: String) -> String {
    let want = id.uppercased()
    guard let issue = sync({ issues }).first(where: { $0.id == want }) else {
      return "there is no issue \(id) — ids look like APP-3"
    }
    var text = Self.oneLine(issue) + "\ncreated by \(issue.creator)"
    if let kept = sync({ comments[want] }), !kept.isEmpty {
      text += "\ncomments: " + kept.joined(separator: "; ")
    }
    return text
  }

  // MARK: Acting on one issue, by id

  private func changeIssue(_ id: String, what: String, _ change: (inout Issue) -> Void) -> String? {
    let want = id.uppercased()
    return sync {
      guard let index = issues.firstIndex(where: { $0.id == want }) else { return nil }
      pushHistory(what)
      change(&issues[index])
      return Self.oneLine(issues[index])
    }
  }

  private static let noSuchIssue = " — ids look like APP-3"

  func assign(id: String, assignee: String) -> String {
    guard let canonical = PmData.person(assignee), PmData.assignees.contains(canonical) else {
      return "nobody called \(assignee) here — the assignees are \(PmData.assignees.joined(separator: ", "))"
    }
    guard let row = changeIssue(id, what: "assignee → \(canonical) on \(id.uppercased())", { $0.assignee = canonical }) else {
      return "there is no issue \(id)" + Self.noSuchIssue
    }
    post()
    return "assigned: \(row)"
  }

  func changeStatus(id: String, status: String) -> String {
    let want = status.lowercased()
    guard PmData.statuses.contains(want) else {
      return "unknown status; the statuses are \(PmData.statuses.joined(separator: ", "))"
    }
    guard let row = changeIssue(id, what: "status → \(want) on \(id.uppercased())", { $0.status = want }) else {
      return "there is no issue \(id)" + Self.noSuchIssue
    }
    post()
    return "moved to \(want): \(row)"
  }

  func changePriority(id: String, priority: String) -> String {
    let want = priority.uppercased()
    guard PmData.priorities.contains(want) else {
      return "unknown priority; the priorities are \(PmData.priorities.joined(separator: ", "))"
    }
    guard let row = changeIssue(id, what: "priority → \(want) on \(id.uppercased())", { $0.priority = want }) else {
      return "there is no issue \(id)" + Self.noSuchIssue
    }
    post()
    return "priority changed: \(row)"
  }

  func changeDue(id: String, due: String) -> String {
    guard let row = changeIssue(id, what: "due → \(due) on \(id.uppercased())", { $0.due = due }) else {
      return "there is no issue \(id)" + Self.noSuchIssue
    }
    post()
    return "due date changed: \(row)"
  }

  func close(id: String) -> String {
    guard let row = changeIssue(id, what: "closed \(id.uppercased())", { $0.status = "done" }) else {
      return "there is no issue \(id)" + Self.noSuchIssue
    }
    post()
    return "closed: \(row)"
  }

  func addComment(id: String, text: String) -> String {
    let want = id.uppercased()
    let title: String? = sync { issues.first { $0.id == want }?.title }
    guard let title else { return "there is no issue \(id)" + Self.noSuchIssue }
    sync {
      pushHistory("comment on \(want)")
      comments[want, default: []].append(text)
    }
    return "comment added to \(want) (\(title)): \"\(text)\""
  }

  func createIssue(
    project: String, title: String, description: String?, priority: String?, assignee: String?
  ) -> String {
    let want = project.lowercased()
    guard let canonicalProject = PmData.projects.first(where: { $0.lowercased() == want }) else {
      return "no project called \(project) — the projects are \(PmData.projects.joined(separator: ", "))"
    }
    var canonicalAssignee: String?
    if let assignee, !assignee.isEmpty {
      guard let person = PmData.person(assignee), PmData.assignees.contains(person) else {
        return "nobody called \(assignee) here — the assignees are \(PmData.assignees.joined(separator: ", "))"
      }
      canonicalAssignee = person
    }
    let pri = (priority?.isEmpty == false ? priority!.uppercased() : "P2")
    guard PmData.priorities.contains(pri) else {
      return "unknown priority; the priorities are \(PmData.priorities.joined(separator: ", "))"
    }
    let issue: Issue = sync {
      pushHistory("created issue")
      let next = Issue(
        id: "APP-\(issues.count + 1)", project: canonicalProject, title: title,
        priority: pri, status: "todo", assignee: canonicalAssignee, creator: "Ito",
        due: PmData.defaultDue)
      issues.append(next)
      if let description, !description.isEmpty { comments[next.id] = [description] }
      return next
    }
    post()
    return "created: \(Self.oneLine(issue))"
  }
}

/// One team's board, frozen — calendar included: every due date is absolute
/// and `today` never moves, so "due by Friday" means the same rows on any
/// day the bench runs. Exactly one iOS P1 sits untouched (the spec's
/// headline search has one right answer), one issue is unassigned (assign
/// has a natural target), Tanaka holds three (the assignee search has a
/// list), and the due dates straddle this week, next week and next month.
@available(iOS 27.0, *)
enum PmData {
  /// The frozen "now" — 2026-08-20 is a Thursday: "Friday" is 08-21,
  /// "next Monday" is 08-24. The weekday is part of the state line.
  static let today = "2026-08-20"
  static let todayWeekday = "Thursday"
  /// Where a created issue's due lands when the request names none:
  /// a week out from the frozen today.
  static let defaultDue = "2026-08-27"

  static let projects = ["iOS", "Web", "Backend"]
  static let statuses = ["todo", "in_progress", "in_review", "done"]
  static let priorities = ["P1", "P2", "P3"]
  static let assignees = ["Tanaka", "Sato", "Suzuki"]
  static let people = ["Tanaka", "Sato", "Suzuki", "Ito"]

  /// The words the Japanese beats say, mapped to how the canned rows spell
  /// them — the money pack's kana lesson, again.
  static let kana: [String: String] = [
    "田中": "tanaka", "タナカ": "tanaka", "佐藤": "sato", "サトウ": "sato",
    "鈴木": "suzuki", "スズキ": "suzuki", "伊藤": "ito", "イトウ": "ito",
    "ログイン": "login", "クラッシュ": "crash", "ダークモード": "dark mode",
    "通知": "notification", "検索": "search", "決済": "payment", "支払": "payment",
    "バックアップ": "backup", "アップロード": "upload",
  ]

  static func romaji(_ text: String) -> String {
    let needle = text.lowercased().trimmingCharacters(in: .whitespaces)
    for (kana, latin) in kana where needle.contains(kana.lowercased()) { return latin }
    return needle
  }

  /// A person however the user said them — Suzuki, suzuki, 鈴木 —
  /// canonical, or nil for a name that is nobody here.
  static func person(_ said: String) -> String? {
    let needle = romaji(said)
    return people.first { needle.contains($0.lowercased()) || $0.lowercased().contains(needle) }
  }

  static let issues: [IssueBox.Issue] = [
    .init(id: "APP-1", project: "iOS", title: "Login crash on cold start", priority: "P1", status: "in_progress", assignee: "Tanaka", creator: "Ito", due: "2026-08-21"),
    .init(id: "APP-2", project: "iOS", title: "Dark mode colors wrong in settings", priority: "P2", status: "in_review", assignee: "Sato", creator: "Ito", due: "2026-08-24"),
    .init(id: "APP-3", project: "iOS", title: "Push notifications not received on reinstall", priority: "P1", status: "todo", assignee: nil, creator: "Tanaka", due: "2026-08-28"),
    .init(id: "APP-4", project: "Web", title: "Checkout button misaligned on mobile", priority: "P3", status: "todo", assignee: "Suzuki", creator: "Ito", due: "2026-09-04"),
    .init(id: "APP-5", project: "Web", title: "Search results page loads slowly", priority: "P2", status: "in_progress", assignee: "Sato", creator: "Sato", due: "2026-08-27"),
    .init(id: "APP-6", project: "Backend", title: "Order export times out over 10k rows", priority: "P1", status: "todo", assignee: "Suzuki", creator: "Ito", due: "2026-08-22"),
    .init(id: "APP-7", project: "iOS", title: "Add haptic feedback to pull-to-refresh", priority: "P3", status: "todo", assignee: "Tanaka", creator: "Sato", due: "2026-09-10"),
    .init(id: "APP-8", project: "Web", title: "Update privacy policy footer link", priority: "P3", status: "done", assignee: "Sato", creator: "Ito", due: "2026-08-18"),
    .init(id: "APP-9", project: "Backend", title: "Rotate API keys for payment provider", priority: "P2", status: "in_progress", assignee: "Suzuki", creator: "Ito", due: "2026-08-26"),
    .init(id: "APP-10", project: "iOS", title: "Voice input cuts off after 30 seconds", priority: "P2", status: "todo", assignee: "Tanaka", creator: "Ito", due: "2026-08-31"),
    .init(id: "APP-11", project: "Web", title: "Image uploads fail over 5 MB", priority: "P2", status: "in_progress", assignee: "Sato", creator: "Suzuki", due: "2026-08-25"),
    .init(id: "APP-12", project: "Backend", title: "Nightly backup job silently skips failures", priority: "P2", status: "in_review", assignee: "Suzuki", creator: "Ito", due: "2026-09-01"),
  ]
}

// MARK: - Tools (the board's menu, in its words)

@available(iOS 27.0, *)
struct SearchIssuesTool: Tool {
  let name = "search_issues"
  let description =
    "Find issues by any mix of project, status, priority, assignee, creator, title keyword and due-date range; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "The project: iOS, Web or Backend.") var project: String?
    @Guide(description: "One status: todo, in_progress, in_review or done.") var status: String?
    @Guide(description: "One priority: P1, P2 or P3.") var priority: String?
    @Guide(description: "The assignee's name — one of the assignees the state lists — or unassigned.")
    var assignee: String?
    @Guide(description: "Who created the issue.") var creator: String?
    @Guide(description: "Words from the issue's title.") var keyword: String?
    // Renamed from due_date_from/to (r17): "due by Friday" landed in the
    // *from* bound three runs straight, in both languages — the argument's
    // name is part of the contract, and due_by is the sentence's own word.
    @Guide(description: "Only issues due on or after this date, YYYY-MM-DD.") var due_from: String?
    @Guide(description: "Only issues due on or before this date, YYYY-MM-DD — \"by Friday\" goes here.") var due_by: String?
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.searchIssues(
      project: arguments.project, status: arguments.status, priority: arguments.priority,
      assignee: arguments.assignee, creator: arguments.creator, keyword: arguments.keyword,
      dueFrom: arguments.due_from, dueTo: arguments.due_by)
  }
}

@available(iOS 27.0, *)
struct GetIssueTool: Tool {
  let name = "get_issue"
  let description = "One issue's full details — creator and comments included."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first.")
    var id: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.getIssue(arguments.id)
  }
}

@available(iOS 27.0, *)
struct CreateIssueTool: Tool {
  let name = "create_issue"
  let description = "Open a new issue on the board."
  @Generable struct Arguments {
    @Guide(description: "The project it belongs to.", .anyOf(PmData.projects)) var project: String
    @Guide(description: "The issue's title, in the user's words.") var title: String
    @Guide(description: "More detail, when the user gave any.") var description: String?
    @Guide(description: "P1, P2 or P3; leave it out for P2.") var priority: String?
    @Guide(description: "Who to assign it to; leave it out for unassigned.") var assignee: String?
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.createIssue(
      project: arguments.project, title: arguments.title, description: arguments.description,
      priority: arguments.priority, assignee: arguments.assignee)
  }
}

@available(iOS 27.0, *)
struct AssignIssueTool: Tool {
  let name = "assign_issue"
  let description = "Hand one issue to an assignee."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first. If no issue is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The new assignee's name — one of the assignees the state lists.") var assignee: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.assign(id: arguments.id, assignee: arguments.assignee)
  }
}

@available(iOS 27.0, *)
struct ChangeIssueStatusTool: Tool {
  let name = "change_issue_status"
  let description = "Move one issue to a different status column."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first. If no issue is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The column the issue moves to.", .anyOf(PmData.statuses)) var status: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.changeStatus(id: arguments.id, status: arguments.status)
  }
}

@available(iOS 27.0, *)
struct ChangeIssuePriorityTool: Tool {
  let name = "change_issue_priority"
  let description = "Change one issue's priority."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first. If no issue is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The new priority.", .anyOf(PmData.priorities)) var priority: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.changePriority(id: arguments.id, priority: arguments.priority)
  }
}

@available(iOS 27.0, *)
struct ChangeDueDateTool: Tool {
  let name = "change_due_date"
  let description = "Change one issue's due date."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first. If no issue is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The new due date, YYYY-MM-DD — count from the today the state names.") var due_date: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.changeDue(id: arguments.id, due: arguments.due_date)
  }
}

@available(iOS 27.0, *)
struct AddCommentTool: Tool {
  let name = "add_comment"
  let description = "Add a comment to one issue."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first.")
    var id: String
    @Guide(description: "The comment.") var text: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.addComment(id: arguments.id, text: arguments.text)
  }
}

@available(iOS 27.0, *)
struct CloseIssueTool: Tool {
  let name = "close_issue"
  let description = "Close one issue — use this for \"close it\"; the issue leaves the open board."
  @Generable struct Arguments {
    @Guide(description: "The issue's id, e.g. APP-3, from the request or the state — no need to search first. If no issue is identified anywhere, call ask_user instead.")
    var id: String
  }
  func call(arguments: Arguments) async throws -> String {
    IssueBox.shared.close(id: arguments.id)
  }
}

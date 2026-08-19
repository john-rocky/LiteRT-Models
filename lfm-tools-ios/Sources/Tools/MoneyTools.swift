// The money pack: a household-budget app (the Money Forward type), said out loud.
//
// Records again, but the records are the user's spending: canned transactions
// over the last month, categories, budgets. The finders select (a period, a
// category, a payee, the uncategorized), the actions act on the selection
// (categorize, flag), and the reports fold the rows into an answer with the
// arithmetic done by the app. "They're all groceries" is the anaphora beat:
// the rows it means are the ones the last finder found.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class MoneyBox: @unchecked Sendable {
  static let shared = MoneyBox()

  struct Transaction: Sendable {
    let id: Int
    let payee: String
    let amount: Int  // yen, positive = spent
    var category: String?  // nil = uncategorized
    let daysAgo: Int
    var flagged = false
  }

  static let categories = [
    "groceries", "eating_out", "transport", "utilities", "rent", "subscriptions",
    "shopping", "health", "leisure",
  ]

  private let lock = NSLock()
  private var transactions: [Transaction] = MoneyData.transactions
  private var budgets: [String: Int] = [
    "groceries": 40000, "eating_out": 20000, "transport": 10000, "subscriptions": 5000,
  ]
  private var selection: [Int] = []
  private var selectionHow = ""
  private let history = UndoStack<([Transaction], [String: Int], [Int], String)>()

  private func pushHistory(_ what: String) {
    history.push((transactions, budgets, selection, selectionHow), what)
  }

  func undoLast() -> String {
    guard let (snap, what) = history.pop() else { return "nothing to undo" }
    sync { (transactions, budgets, selection, selectionHow) = snap }
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
    let (rows, selection, how, budgets) = sync { (transactions, selection, selectionHow, budgets) }
    let month = rows.filter { $0.daysAgo < 30 }
    let spent = month.reduce(0) { $0 + $1.amount }
    let uncategorized = rows.filter { $0.category == nil }.count
    var line = "Money: \(rows.count) transactions in the last month, \(StoreBox.yen(spent)) spent, \(uncategorized) uncategorized."
    line += " Budgets: " + budgets.keys.sorted().map { "\($0) \(StoreBox.yen(budgets[$0]!))" }.joined(separator: "; ") + "."
    if selection.isEmpty {
      line += " Selection: none — list, search or filter first, then act."
    } else {
      let selected = rows.filter { selection.contains($0.id) }
      let names = selected.prefix(5).map { "\($0.payee) \(StoreBox.yen($0.amount)) (\(Self.when($0.daysAgo))\($0.category.map { ", \($0)" } ?? ", uncategorized"))" }
      let more = selected.count > 5 ? ", and \(selected.count - 5) more" : ""
      line += " Selection: \(selected.count) from \(how): " + names.joined(separator: "; ") + more + "."
    }
    return line
  }

  static func when(_ daysAgo: Int) -> String {
    daysAgo == 0 ? "today" : (daysAgo == 1 ? "yesterday" : "\(daysAgo) d ago")
  }

  // MARK: The panel

  func snapshot() -> TablePanel {
    let (rows, selection, how) = sync { (transactions, selection, selectionHow) }
    let shown = selection.isEmpty ? rows.sorted { $0.daysAgo < $1.daysAgo } : rows.filter { selection.contains($0.id) }
    let spent = rows.filter { $0.daysAgo < 30 }.reduce(0) { $0 + $1.amount }
    return TablePanel(
      title: selection.isEmpty ? "This month" : "\(shown.count) selected — \(how)",
      columns: ["When", "Payee", "Amount", "Category"],
      rows: shown.map { [Self.when($0.daysAgo), $0.payee, StoreBox.yen($0.amount), ($0.category ?? "—") + ($0.flagged ? " ⚑" : "")] },
      totalRows: shown.count,
      overview: "\(StoreBox.yen(spent)) this month")
  }

  private func post() {
    let panel = snapshot()
    ArtifactBox.shared.post(.table(
      title: panel.title, columns: panel.columns, rows: Array(panel.rows.prefix(8))))
  }

  // MARK: Finding (each replaces the selection)

  private func select(_ rows: [Transaction], how: String) -> String {
    sync {
      selection = rows.map(\.id)
      selectionHow = how
    }
    post()
    guard !rows.isEmpty else { return "no transactions match \(how)" }
    let listed = rows.prefix(8).map { "\(Self.when($0.daysAgo)) \($0.payee) — \(StoreBox.yen($0.amount))\($0.category.map { ", \($0)" } ?? ", uncategorized")" }
    let total = rows.reduce(0) { $0 + $1.amount }
    return "\(rows.count) transaction\(rows.count == 1 ? "" : "s") (\(how)), \(StoreBox.yen(total)) in all:\n"
      + listed.joined(separator: "\n") + (rows.count > 8 ? "\n… and \(rows.count - 8) more" : "")
  }

  func list(days: Int) -> String {
    let span = max(1, days)
    let rows = sync { transactions }.filter { $0.daysAgo < span }.sorted { $0.daysAgo < $1.daysAgo }
    return select(rows, how: "last \(span) day\(span == 1 ? "" : "s")")
  }

  func filter(category: String) -> String {
    let want = category.lowercased()
    if want == "uncategorized" || want == "none" {
      let rows = sync { transactions }.filter { $0.category == nil }.sorted { $0.daysAgo < $1.daysAgo }
      return select(rows, how: "uncategorized")
    }
    let rows = sync { transactions }.filter { $0.category == want }.sorted { $0.daysAgo < $1.daysAgo }
    return select(rows, how: "category \(want)")
  }

  func search(payee: String) -> String {
    let needle = payee.lowercased()
    let rows = sync { transactions }.filter { $0.payee.lowercased().contains(needle) }
      .sorted { $0.daysAgo < $1.daysAgo }
    return select(rows, how: "payee \"\(payee)\"")
  }

  // MARK: Acting on the selection

  func categorize(as category: String) -> String {
    let want = category.lowercased()
    guard Self.categories.contains(want) else {
      return "unknown category; try \(Self.categories.joined(separator: ", "))"
    }
    let ids = sync { selection }
    guard !ids.isEmpty else { return "nothing is selected — list, search or filter first" }
    sync {
      pushHistory("categorizing")
      for index in transactions.indices where ids.contains(transactions[index].id) {
        transactions[index].category = want
      }
    }
    post()
    return "categorized \(ids.count) transaction\(ids.count == 1 ? "" : "s") as \(want)"
  }

  func flag() -> String {
    let ids = sync { selection }
    guard !ids.isEmpty else { return "nothing is selected — list, search or filter first" }
    sync {
      pushHistory("flags")
      for index in transactions.indices where ids.contains(transactions[index].id) {
        transactions[index].flagged = true
      }
    }
    post()
    return "flagged \(ids.count) transaction\(ids.count == 1 ? "" : "s") to check later"
  }

  // MARK: Budgets and reports

  func setBudget(category: String, amount: Int) -> String {
    let want = category.lowercased()
    guard Self.categories.contains(want) else {
      return "unknown category; try \(Self.categories.joined(separator: ", "))"
    }
    sync {
      pushHistory("budget change")
      budgets[want] = max(0, amount)
    }
    return "budget for \(want) set to \(StoreBox.yen(max(0, amount))) a month"
  }

  func report(days: Int) -> String {
    let span = max(1, days)
    let rows = sync { transactions }.filter { $0.daysAgo < span }
    let total = rows.reduce(0) { $0 + $1.amount }
    var byCategory: [String: Int] = [:]
    for row in rows { byCategory[row.category ?? "uncategorized", default: 0] += row.amount }
    let top = byCategory.sorted { $0.value > $1.value }
    ArtifactBox.shared.post(.table(
      title: "last \(span) days — \(StoreBox.yen(total))",
      columns: ["Category", "Spent"],
      rows: top.map { [$0.key, StoreBox.yen($0.value)] }))
    return "last \(span) day\(span == 1 ? "" : "s"): \(StoreBox.yen(total)) across \(rows.count) transactions — "
      + top.prefix(4).map { "\($0.key) \(StoreBox.yen($0.value))" }.joined(separator: ", ")
  }

  func budgetReport() -> String {
    let (rows, budgets) = sync { (transactions, budgets) }
    let month = rows.filter { $0.daysAgo < 30 }
    var lines: [String] = []
    var table: [[String]] = []
    for (category, budget) in budgets.sorted(by: { $0.key < $1.key }) {
      let spent = month.filter { $0.category == category }.reduce(0) { $0 + $1.amount }
      let state = spent > budget ? "over by \(StoreBox.yen(spent - budget))" : "\(StoreBox.yen(budget - spent)) left"
      lines.append("\(category): \(StoreBox.yen(spent)) of \(StoreBox.yen(budget)) — \(state)")
      table.append([category, StoreBox.yen(spent), StoreBox.yen(budget), state])
    }
    ArtifactBox.shared.post(.table(
      title: "budgets, this month", columns: ["Category", "Spent", "Budget", ""], rows: table))
    return "this month against budgets: " + lines.joined(separator: "; ")
  }

  func findSubscriptions() -> String {
    // A subscription is a payee that shows up on a near-identical amount
    // more than once in the window — the app's heuristic, not the model's.
    let rows = sync { transactions }
    var byPayee: [String: [Transaction]] = [:]
    for row in rows { byPayee[row.payee, default: []].append(row) }
    let subs = byPayee.filter { _, rows in
      rows.count >= 2 && Set(rows.map { $0.amount / 100 }).count == 1
    }.map { payee, rows in (payee, rows[0].amount, rows.count) }
      .sorted { $0.1 > $1.1 }
    guard !subs.isEmpty else { return "no recurring payments found" }
    let ids = rows.filter { row in subs.contains { $0.0 == row.payee } }.map(\.id)
    sync {
      selection = ids
      selectionHow = "subscriptions"
    }
    post()
    let monthly = subs.reduce(0) { $0 + $1.1 }
    return "\(subs.count) recurring payment\(subs.count == 1 ? "" : "s"), about \(StoreBox.yen(monthly)) a month: "
      + subs.map { "\($0.0) \(StoreBox.yen($0.1))" }.joined(separator: "; ")
  }
}

/// A month of one person's spending, frozen. Two payees recur on the same
/// amount (the subscriptions), five rows are uncategorized (the triage), and
/// eating out is over budget (the report has something to say).
enum MoneyData {
  static let transactions: [MoneyBox.Transaction] = [
    .init(id: 1, payee: "Maruetsu", amount: 4820, category: "groceries", daysAgo: 1),
    .init(id: 2, payee: "Seven-Eleven", amount: 680, category: nil, daysAgo: 0),
    .init(id: 3, payee: "JR East", amount: 1340, category: "transport", daysAgo: 1),
    .init(id: 4, payee: "Netflix", amount: 1490, category: "subscriptions", daysAgo: 3),
    .init(id: 5, payee: "Ichiran Ramen", amount: 1180, category: "eating_out", daysAgo: 2),
    .init(id: 6, payee: "Maruetsu", amount: 6210, category: "groceries", daysAgo: 4),
    .init(id: 7, payee: "Starbucks", amount: 720, category: nil, daysAgo: 3),
    .init(id: 8, payee: "Tokyo Gas", amount: 5830, category: "utilities", daysAgo: 5),
    .init(id: 9, payee: "Spotify", amount: 980, category: "subscriptions", daysAgo: 6),
    .init(id: 10, payee: "Sushiro", amount: 3460, category: "eating_out", daysAgo: 6),
    .init(id: 11, payee: "Don Quijote", amount: 2890, category: nil, daysAgo: 7),
    .init(id: 12, payee: "JR East", amount: 890, category: "transport", daysAgo: 8),
    .init(id: 13, payee: "Maruetsu", amount: 5130, category: "groceries", daysAgo: 9),
    .init(id: 14, payee: "Saizeriya", amount: 2140, category: "eating_out", daysAgo: 10),
    .init(id: 15, payee: "Matsumoto Kiyoshi", amount: 1670, category: "health", daysAgo: 11),
    .init(id: 16, payee: "Uniqlo", amount: 5980, category: "shopping", daysAgo: 12),
    .init(id: 17, payee: "Seven-Eleven", amount: 540, category: nil, daysAgo: 12),
    .init(id: 18, payee: "TOHO Cinemas", amount: 1900, category: "leisure", daysAgo: 13),
    .init(id: 19, payee: "Ootoya", amount: 1580, category: "eating_out", daysAgo: 14),
    .init(id: 20, payee: "Tokyo Water", amount: 3120, category: "utilities", daysAgo: 15),
    .init(id: 21, payee: "Maruetsu", amount: 4470, category: "groceries", daysAgo: 16),
    .init(id: 22, payee: "Kura Sushi", amount: 2980, category: "eating_out", daysAgo: 17),
    .init(id: 23, payee: "Netflix", amount: 1490, category: "subscriptions", daysAgo: 33),
    .init(id: 24, payee: "Spotify", amount: 980, category: "subscriptions", daysAgo: 36),
    .init(id: 25, payee: "JR East", amount: 1120, category: "transport", daysAgo: 18),
    .init(id: 26, payee: "Book Off", amount: 880, category: nil, daysAgo: 19),
    .init(id: 27, payee: "Torikizoku", amount: 3890, category: "eating_out", daysAgo: 20),
    .init(id: 28, payee: "Maruetsu", amount: 5560, category: "groceries", daysAgo: 22),
    .init(id: 29, payee: "Gong Cha", amount: 640, category: "eating_out", daysAgo: 24),
    .init(id: 30, payee: "Yoshinoya", amount: 690, category: "eating_out", daysAgo: 26),
  ]
}

// MARK: - Tools (the app's screens, in its words)

@available(iOS 27.0, *)
struct ListTransactionsTool: Tool {
  let name = "list_transactions"
  let description = "The transactions of the last N days; they become the selection."
  @Generable struct Arguments {
    @Guide(description: "How many days back. 7 is this week, 30 the month.") var days: Int
  }
  func call(arguments: Arguments) async throws -> String {
    MoneyBox.shared.list(days: arguments.days)
  }
}

@available(iOS 27.0, *)
struct FilterTransactionsTool: Tool {
  let name = "filter_by_category"
  let description = "Transactions in one category — or the uncategorized ones; they become the selection."
  @Generable struct Arguments {
    @Guide(
      description: "The category, or uncategorized.",
      .anyOf(MoneyBox.categories + ["uncategorized"]))
    var category: String
  }
  func call(arguments: Arguments) async throws -> String {
    MoneyBox.shared.filter(category: arguments.category)
  }
}

@available(iOS 27.0, *)
struct SearchPayeeTool: Tool {
  let name = "search_payee"
  let description = "Transactions at one shop or service; they become the selection."
  @Generable struct Arguments {
    @Guide(description: "Part of the payee's name, e.g. Maruetsu.") var payee: String
  }
  func call(arguments: Arguments) async throws -> String {
    MoneyBox.shared.search(payee: arguments.payee)
  }
}

@available(iOS 27.0, *)
struct CategorizeTool: Tool {
  let name = "categorize_selection"
  let description = "Put the selected transactions in a category."
  @Generable struct Arguments {
    @Guide(description: "The category.", .anyOf(MoneyBox.categories)) var category: String
  }
  func call(arguments: Arguments) async throws -> String {
    MoneyBox.shared.categorize(as: arguments.category)
  }
}

@available(iOS 27.0, *)
struct FlagTransactionsTool: Tool {
  let name = "flag_selection"
  let description = "Flag the selected transactions to check later."
  func call(arguments: NoArguments) async throws -> String {
    MoneyBox.shared.flag()
  }
}

@available(iOS 27.0, *)
struct SetBudgetTool: Tool {
  let name = "set_budget"
  let description = "Set a category's monthly budget."
  @Generable struct Arguments {
    @Guide(description: "The category.", .anyOf(MoneyBox.categories)) var category: String
    @Guide(description: "The monthly budget in yen.") var amount: Int
  }
  func call(arguments: Arguments) async throws -> String {
    MoneyBox.shared.setBudget(category: arguments.category, amount: arguments.amount)
  }
}

@available(iOS 27.0, *)
struct SpendingReportTool: Tool {
  let name = "spending_report"
  let description = "What was spent in the last N days, by category."
  @Generable struct Arguments {
    @Guide(description: "How many days back. 7 is this week, 30 the month.") var days: Int
  }
  func call(arguments: Arguments) async throws -> String {
    MoneyBox.shared.report(days: arguments.days)
  }
}

@available(iOS 27.0, *)
struct BudgetReportTool: Tool {
  let name = "budget_report"
  let description = "Compare this month's spending with each budget — over or under, by how much."
  func call(arguments: NoArguments) async throws -> String {
    MoneyBox.shared.budgetReport()
  }
}

@available(iOS 27.0, *)
struct FindSubscriptionsTool: Tool {
  let name = "find_subscriptions"
  let description = "Find recurring payments (subscriptions); they become the selection."
  func call(arguments: NoArguments) async throws -> String {
    MoneyBox.shared.findSubscriptions()
  }
}

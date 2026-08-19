// The CRM pack: a Salesforce's pipeline, said out loud, over canned data.
//
// The first business-wing pack (docs/business-packs.md): contacts, companies,
// opportunities and tasks as frozen fixtures, and the tools a sales rep
// actually says — search by owner / stage / amount / close date, move a deal
// through the pipeline, hand it to a colleague, put a follow-up on it.
//
// Two shapes deliberately coexist. The finders make a *selection* (the rows
// on screen), which is what "this one" resolves through; the actions take
// their record as an **id argument** — the video pack measured that chains
// which exist only to set an implicit target don't survive, so O3 in the
// request or the selection line is the contract, never a search-first habit.
// The world is frozen, including the calendar: the state pins today's date,
// so "tomorrow" and "this month" resolve to absolute dates a case can score.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class CrmBox: @unchecked Sendable {
  static let shared = CrmBox()

  struct Opportunity: Sendable {
    let id: String  // O1…: the handle the user points with, shown in every row
    let company: String
    var amount: Int  // yen
    var stage: String
    var owner: String
    var closeDate: String  // YYYY-MM-DD, absolute — the world's calendar is frozen
  }

  struct Contact: Sendable {
    let id: String  // C1…
    let name: String
    let company: String
    let role: String
    let email: String
  }

  struct Company: Sendable {
    let name: String
    let industry: String
    let location: String
  }

  struct FollowUpTask: Sendable {
    let id: String  // T1…
    var title: String
    var due: String  // YYYY-MM-DD
    var about: String?  // an O*/C* id
  }

  static let stages = ["prospecting", "qualification", "proposal", "negotiation", "won", "lost"]

  enum Selection: Sendable {
    case none
    case opportunities([String], how: String)
    case contacts([String], how: String)
    case companies([String], how: String)
  }

  private let lock = NSLock()
  private var opportunities: [Opportunity] = CrmData.opportunities
  private var contacts: [Contact] = CrmData.contacts
  private var tasks: [FollowUpTask] = CrmData.tasks
  private var notes: [String: [String]] = [:]
  private var selection: Selection = .none
  private let history = UndoStack<([Opportunity], [FollowUpTask], [String: [String]], Selection)>()

  private func pushHistory(_ what: String) {
    history.push((opportunities, tasks, notes, selection), what)
  }

  func undoLast() -> String {
    guard let (snap, what) = history.pop() else { return "nothing to undo" }
    sync { (opportunities, tasks, notes, selection) = snap }
    post()
    return "undid the last change (\(what))"
  }

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  // MARK: The state the model reads

  /// Counts and handles, not rows: listing every deal here would let search
  /// questions be answered from the state (the docs pack's title trap), and
  /// the finders are what the pack measures. What the state must carry is
  /// every word a person points with — the owners, the companies, today's
  /// date — and, when something is selected, the full handles of those rows.
  func describe() -> String {
    let (opps, contacts, tasks, selection) = sync { (opportunities, self.contacts, self.tasks, self.selection) }
    let open = opps.filter { $0.stage != "won" && $0.stage != "lost" }
    let pipeline = open.reduce(0) { $0 + $1.amount }
    let byStage = Self.stages.compactMap { stage -> String? in
      let count = opps.filter { $0.stage == stage }.count
      return count > 0 ? "\(count) \(stage)" : nil
    }
    var line = "Today: \(CrmData.today). CRM: \(opps.count) opportunities (\(byStage.joined(separator: ", "))) — open pipeline \(StoreBox.yen(pipeline))."
    line += " Owners: \(CrmData.owners.joined(separator: ", "))."
    line += " Companies: " + CrmData.companies.map(\.name).sorted().joined(separator: ", ") + "."
    line += " \(contacts.count) contacts, \(tasks.count) open tasks."
    if let last = history.peekWhat() { line += " Last change: \(last)." }
    switch selection {
    case .none:
      line += " Selection: none."
    case .opportunities(let ids, let how):
      let rows = ids.compactMap { id in opps.first { $0.id == id } }
      let names = rows.prefix(6).map(Self.oneLine).joined(separator: "; ")
      let more = rows.count > 6 ? ", and \(rows.count - 6) more" : ""
      let word = rows.count == 1 ? "opportunity" : "opportunities"
      line += " Selection: \(rows.count) \(word) from \(how): \(names)\(more)."
    case .contacts(let ids, let how):
      let rows = ids.compactMap { id in contacts.first { $0.id == id } }
      let names = rows.prefix(6)
        .map { "\($0.id) \($0.name) — \($0.company), \($0.role), \($0.email)" }
        .joined(separator: "; ")
      let more = rows.count > 6 ? ", and \(rows.count - 6) more" : ""
      let word = rows.count == 1 ? "contact" : "contacts"
      line += " Selection: \(rows.count) \(word) from \(how): \(names)\(more)."
    case .companies(let names, let how):
      let rows = names.compactMap { name in CrmData.companies.first { $0.name == name } }
      let listed = rows.map { "\($0.name) — \($0.industry), \($0.location)" }.joined(separator: "; ")
      let word = rows.count == 1 ? "company" : "companies"
      line += " Selection: \(rows.count) \(word) from \(how): \(listed)."
    }
    return line
  }

  /// Every handle a person grabs a deal by, in one selection row. Internal,
  /// not private: the bench's CrmEcho renders the same rows over the same
  /// frozen data, and a second copy of this format would drift.
  static func oneLine(_ o: Opportunity) -> String {
    "\(o.id) \(o.company) — \(StoreBox.yen(o.amount)), \(o.stage), owner \(o.owner), closes \(o.closeDate)"
  }

  // MARK: The panel

  func snapshot() -> TablePanel {
    let (opps, contacts, selection) = sync { (opportunities, self.contacts, self.selection) }
    let open = opps.filter { $0.stage != "won" && $0.stage != "lost" }
    let overview = "\(opps.count) deals · \(StoreBox.yen(open.reduce(0) { $0 + $1.amount })) open pipeline"
    switch selection {
    case .none:
      return TablePanel(
        title: "Pipeline",
        columns: ["ID", "Company", "Amount", "Stage", "Owner", "Closes"],
        rows: opps.map { [$0.id, $0.company, StoreBox.yen($0.amount), $0.stage, $0.owner, $0.closeDate] },
        totalRows: opps.count, overview: overview)
    case .opportunities(let ids, let how):
      let rows = ids.compactMap { id in opps.first { $0.id == id } }
      return TablePanel(
        title: "\(rows.count) selected — \(how)",
        columns: ["ID", "Company", "Amount", "Stage", "Owner", "Closes"],
        rows: rows.map { [$0.id, $0.company, StoreBox.yen($0.amount), $0.stage, $0.owner, $0.closeDate] },
        totalRows: rows.count, overview: overview)
    case .contacts(let ids, let how):
      let rows = ids.compactMap { id in contacts.first { $0.id == id } }
      return TablePanel(
        title: "\(rows.count) contacts — \(how)",
        columns: ["ID", "Name", "Company", "Role", "Email"],
        rows: rows.map { [$0.id, $0.name, $0.company, $0.role, $0.email] },
        totalRows: rows.count, overview: overview)
    case .companies(let names, let how):
      let rows = names.compactMap { name in CrmData.companies.first { $0.name == name } }
      return TablePanel(
        title: "\(rows.count) companies — \(how)",
        columns: ["Company", "Industry", "Location"],
        rows: rows.map { [$0.name, $0.industry, $0.location] },
        totalRows: rows.count, overview: overview)
    }
  }

  private func post() {
    let panel = snapshot()
    ArtifactBox.shared.post(.table(
      title: panel.title, columns: panel.columns, rows: Array(panel.rows.prefix(8))))
  }

  // MARK: Finding (each replaces the selection)

  func searchOpportunities(
    company: String?, owner: String?, stage: String?,
    minAmount: Int?, maxAmount: Int?, closeFrom: String?, closeTo: String?
  ) -> String {
    var how: [String] = []
    var rows = sync { opportunities }
    if let company, !company.isEmpty {
      let needle = CrmData.romaji(company)
      rows = rows.filter { $0.company.lowercased().contains(needle) }
      how.append("company \"\(company)\"")
    }
    if let owner, !owner.isEmpty {
      guard let canonical = CrmData.owner(owner) else {
        return "no owner called \(owner) — the owners are \(CrmData.owners.joined(separator: ", "))"
      }
      rows = rows.filter { $0.owner == canonical }
      how.append("owner \(canonical)")
    }
    if let stage, !stage.isEmpty {
      let want = stage.lowercased()
      guard Self.stages.contains(want) else {
        return "unknown stage; the stages are \(Self.stages.joined(separator: ", "))"
      }
      rows = rows.filter { $0.stage == want }
      how.append("stage \(want)")
    }
    if let minAmount {
      rows = rows.filter { $0.amount >= minAmount }
      how.append("\(StoreBox.yen(minAmount))+")
    }
    if let maxAmount {
      rows = rows.filter { $0.amount <= maxAmount }
      how.append("up to \(StoreBox.yen(maxAmount))")
    }
    // ISO dates compare correctly as strings; the whole calendar is frozen.
    if let closeFrom, !closeFrom.isEmpty {
      rows = rows.filter { $0.closeDate >= closeFrom }
      how.append("closing from \(closeFrom)")
    }
    if let closeTo, !closeTo.isEmpty {
      rows = rows.filter { $0.closeDate <= closeTo }
      how.append("closing by \(closeTo)")
    }
    let sorted = rows.sorted { $0.closeDate < $1.closeDate }
    let caption = how.isEmpty ? "all opportunities" : how.joined(separator: ", ")
    sync { selection = sorted.isEmpty ? .none : .opportunities(sorted.map(\.id), how: caption) }
    post()
    guard !sorted.isEmpty else { return "no opportunities match \(caption)" }
    let total = sorted.reduce(0) { $0 + $1.amount }
    return "\(sorted.count) opportunit\(sorted.count == 1 ? "y" : "ies") (\(caption)), \(StoreBox.yen(total)) in all:\n"
      + sorted.prefix(8).map(Self.oneLine).joined(separator: "\n")
      + (sorted.count > 8 ? "\n… and \(sorted.count - 8) more" : "")
  }

  func searchContacts(name: String?, company: String?, email: String?) -> String {
    var how: [String] = []
    var rows = sync { contacts }
    if let name, !name.isEmpty {
      let needle = CrmData.romaji(name)
      rows = rows.filter { $0.name.lowercased().contains(needle) }
      how.append("name \"\(name)\"")
    }
    if let company, !company.isEmpty {
      let needle = CrmData.romaji(company)
      rows = rows.filter { $0.company.lowercased().contains(needle) }
      how.append("company \"\(company)\"")
    }
    if let email, !email.isEmpty {
      let needle = email.lowercased()
      rows = rows.filter { $0.email.lowercased().contains(needle) }
      how.append("email \"\(email)\"")
    }
    let caption = how.isEmpty ? "all contacts" : how.joined(separator: ", ")
    sync { selection = rows.isEmpty ? .none : .contacts(rows.map(\.id), how: caption) }
    post()
    guard !rows.isEmpty else { return "no contacts match \(caption)" }
    return "\(rows.count) contact\(rows.count == 1 ? "" : "s") (\(caption)):\n"
      + rows.prefix(8).map { "\($0.id) \($0.name) — \($0.company), \($0.role), \($0.email)" }.joined(separator: "\n")
  }

  func searchCompanies(name: String?, industry: String?, location: String?) -> String {
    var how: [String] = []
    var rows = CrmData.companies
    if let name, !name.isEmpty {
      let needle = CrmData.romaji(name)
      rows = rows.filter { $0.name.lowercased().contains(needle) }
      how.append("name \"\(name)\"")
    }
    if let industry, !industry.isEmpty {
      let needle = CrmData.romaji(industry)
      rows = rows.filter { $0.industry.lowercased().contains(needle) }
      how.append("industry \"\(industry)\"")
    }
    if let location, !location.isEmpty {
      let needle = CrmData.romaji(location)
      rows = rows.filter { $0.location.lowercased().contains(needle) }
      how.append("location \"\(location)\"")
    }
    let caption = how.isEmpty ? "all companies" : how.joined(separator: ", ")
    sync { selection = rows.isEmpty ? .none : .companies(rows.map(\.name), how: caption) }
    post()
    guard !rows.isEmpty else { return "no companies match \(caption)" }
    return "\(rows.count) compan\(rows.count == 1 ? "y" : "ies") (\(caption)):\n"
      + rows.map { "\($0.name) — \($0.industry), \($0.location)" }.joined(separator: "\n")
  }

  // MARK: Reading one record

  func getOpportunity(_ id: String) -> String {
    let want = id.uppercased()
    guard let o = sync({ opportunities }).first(where: { $0.id == want }) else {
      return "there is no opportunity \(id) — ids look like O3"
    }
    let attached = sync { (tasks.filter { $0.about == want }, notes[want] ?? []) }
    var text = "\(Self.oneLine(o))"
    let contact = sync { contacts }.first { $0.company == o.company }
    if let contact { text += "\ncontact: \(contact.name) (\(contact.role), \(contact.email))" }
    if !attached.0.isEmpty {
      text += "\ntasks: " + attached.0.map { "\($0.id) \"\($0.title)\" due \($0.due)" }.joined(separator: "; ")
    }
    if !attached.1.isEmpty { text += "\nnotes: " + attached.1.joined(separator: "; ") }
    return text
  }

  func getContact(_ id: String) -> String {
    let want = id.uppercased()
    guard let c = sync({ contacts }).first(where: { $0.id == want }) else {
      return "there is no contact \(id) — ids look like C2"
    }
    var text = "\(c.id) \(c.name) — \(c.company), \(c.role), \(c.email)"
    let deals = sync { opportunities }.filter { $0.company == c.company && $0.stage != "won" && $0.stage != "lost" }
    if !deals.isEmpty { text += "\nopen deals: " + deals.map(Self.oneLine).joined(separator: "; ") }
    if let kept = sync({ notes[want] }), !kept.isEmpty { text += "\nnotes: " + kept.joined(separator: "; ") }
    return text
  }

  // MARK: Acting on one record, by id

  private func changeOpportunity(
    _ id: String, what: String, _ change: (inout Opportunity) -> Void
  ) -> String? {
    let want = id.uppercased()
    return sync {
      guard let index = opportunities.firstIndex(where: { $0.id == want }) else { return nil }
      pushHistory(what)
      change(&opportunities[index])
      return Self.oneLine(opportunities[index])
    }
  }

  func updateStage(id: String, stage: String) -> String {
    let want = stage.lowercased()
    guard Self.stages.contains(want) else {
      return "unknown stage; the stages are \(Self.stages.joined(separator: ", "))"
    }
    guard let row = changeOpportunity(id, what: "stage → \(want) on \(id.uppercased())", { $0.stage = want }) else {
      return "there is no opportunity \(id) — ids look like O3"
    }
    post()
    return "moved to \(want): \(row)"
  }

  func updateAmount(id: String, amount: Int) -> String {
    guard let row = changeOpportunity(id, what: "amount → \(StoreBox.yen(max(0, amount))) on \(id.uppercased())", { $0.amount = max(0, amount) }) else {
      return "there is no opportunity \(id) — ids look like O3"
    }
    post()
    return "amount changed: \(row)"
  }

  func assign(id: String, owner: String) -> String {
    guard let canonical = CrmData.owner(owner) else {
      return "no owner called \(owner) — the owners are \(CrmData.owners.joined(separator: ", "))"
    }
    guard let row = changeOpportunity(id, what: "owner → \(canonical) on \(id.uppercased())", { $0.owner = canonical }) else {
      return "there is no opportunity \(id) — ids look like O3"
    }
    post()
    return "reassigned: \(row)"
  }

  func createTask(contact: String?, opportunity: String?, title: String?, due: String) -> String {
    let about = (opportunity ?? contact)?.uppercased()
    if let about {
      let exists = sync {
        opportunities.contains { $0.id == about } || contacts.contains { $0.id == about }
      }
      guard exists else { return "there is no record \(about) — ids look like O3 or C2" }
    }
    let words = (title?.isEmpty == false ? title! : "Follow up")
    let task: FollowUpTask = sync {
      pushHistory("task added")
      let next = FollowUpTask(id: "T\(tasks.count + 1)", title: words, due: due, about: about)
      tasks.append(next)
      return next
    }
    post()
    let tied = about.flatMap { id in
      sync { opportunities.first { $0.id == id }.map { "\($0.id) \($0.company)" } }
    }
    return "added task \(task.id) \"\(task.title)\" due \(task.due)" + (tied.map { " (\($0))" } ?? "")
  }

  func addNote(entityId: String, text: String) -> String {
    let want = entityId.uppercased()
    let named: String? = sync {
      if let o = opportunities.first(where: { $0.id == want }) { return "\(o.id) \(o.company)" }
      if let c = contacts.first(where: { $0.id == want }) { return "\(c.id) \(c.name)" }
      return nil
    }
    guard let named else { return "there is no record \(entityId) — ids look like O3 or C2" }
    sync {
      pushHistory("note on \(want)")
      notes[want, default: []].append(text)
    }
    return "note added to \(named): \"\(text)\""
  }
}

/// One sales team's quarter, frozen — calendar included: every date is
/// absolute and `today` never moves, so "closing this month" means the same
/// rows on any day the bench runs. Two deals clear ¥1M and close this month
/// (the search beat has an answer), Tanaka owns three (the owner beat has a
/// list), one deal is already won (the stage counts have shape).
@available(iOS 27.0, *)
enum CrmData {
  /// The frozen "now". The state line opens with it; relative dates in
  /// requests resolve against it, and expected arguments stay absolute.
  static let today = "2026-08-20"

  static let owners = ["Tanaka", "Sato", "Suzuki"]

  /// The names the Japanese beats say, mapped to how the canned rows spell
  /// them — the money pack's kana lesson: a search that routes perfectly
  /// into an empty answer is the failure a routing bench cannot see.
  static let kana: [String: String] = [
    "田中": "tanaka", "タナカ": "tanaka", "佐藤": "sato", "サトウ": "sato",
    "鈴木": "suzuki", "スズキ": "suzuki",
    "アオゾラ": "aozora", "あおぞら": "aozora", "ホシノ": "hoshino", "星野": "hoshino",
    "サクラ": "sakura", "さくら": "sakura", "カンダ": "kanda", "神田": "kanda",
    "ミドリ": "midori", "みどり": "midori", "ヤマト": "yamato", "大和": "yamato",
    "アオキ": "aoki", "青木": "aoki", "ウエダ": "ueda", "上田": "ueda",
    "フジイ": "fujii", "藤井": "fujii", "オカダ": "okada", "岡田": "okada",
    "モリ": "mori", "森": "mori", "オノ": "ono", "小野": "ono",
    "食品": "food", "電機": "electronics", "物流": "logistics", "商社": "trading",
    "銀行": "finance", "金融": "finance", "印刷": "printing",
    "東京": "tokyo", "大阪": "osaka", "名古屋": "nagoya", "福岡": "fukuoka",
  ]

  static func romaji(_ text: String) -> String {
    let needle = text.lowercased().trimmingCharacters(in: .whitespaces)
    for (kana, latin) in kana where needle.contains(kana.lowercased()) { return latin }
    return needle
  }

  /// An owner however the user said it — Suzuki, suzuki, 鈴木 — canonical,
  /// or nil for a name that is nobody here.
  static func owner(_ said: String) -> String? {
    let needle = romaji(said)
    return owners.first { needle.contains($0.lowercased()) || $0.lowercased().contains(needle) }
  }

  static let opportunities: [CrmBox.Opportunity] = [
    .init(id: "O1", company: "Aozora Foods", amount: 1_200_000, stage: "proposal", owner: "Tanaka", closeDate: "2026-08-28"),
    .init(id: "O2", company: "Hoshino Denki", amount: 3_500_000, stage: "negotiation", owner: "Sato", closeDate: "2026-09-04"),
    .init(id: "O3", company: "Sakura Logistics", amount: 800_000, stage: "proposal", owner: "Tanaka", closeDate: "2026-08-25"),
    .init(id: "O4", company: "Kanda Trading", amount: 5_000_000, stage: "qualification", owner: "Suzuki", closeDate: "2026-10-15"),
    .init(id: "O5", company: "Midori Bank", amount: 2_400_000, stage: "negotiation", owner: "Sato", closeDate: "2026-08-31"),
    .init(id: "O6", company: "Aozora Foods", amount: 600_000, stage: "prospecting", owner: "Suzuki", closeDate: "2026-09-18"),
    .init(id: "O7", company: "Yamato Print", amount: 1_800_000, stage: "prospecting", owner: "Tanaka", closeDate: "2026-09-30"),
    .init(id: "O8", company: "Hoshino Denki", amount: 950_000, stage: "won", owner: "Sato", closeDate: "2026-08-12"),
  ]

  static let contacts: [CrmBox.Contact] = [
    .init(id: "C1", name: "Yui Aoki", company: "Aozora Foods", role: "purchasing manager", email: "aoki@aozora-foods.example"),
    .init(id: "C2", name: "Kenji Ueda", company: "Hoshino Denki", role: "engineering director", email: "ueda@hoshino-denki.example"),
    .init(id: "C3", name: "Mari Fujii", company: "Sakura Logistics", role: "operations lead", email: "fujii@sakura-logi.example"),
    .init(id: "C4", name: "Sho Okada", company: "Kanda Trading", role: "buyer", email: "okada@kanda-trading.example"),
    .init(id: "C5", name: "Rina Mori", company: "Midori Bank", role: "IT planning", email: "mori@midori-bank.example"),
    .init(id: "C6", name: "Daiki Ono", company: "Yamato Print", role: "plant manager", email: "ono@yamato-print.example"),
  ]

  static let companies: [CrmBox.Company] = [
    .init(name: "Aozora Foods", industry: "food", location: "Tokyo"),
    .init(name: "Hoshino Denki", industry: "electronics", location: "Osaka"),
    .init(name: "Sakura Logistics", industry: "logistics", location: "Nagoya"),
    .init(name: "Kanda Trading", industry: "trading", location: "Tokyo"),
    .init(name: "Midori Bank", industry: "finance", location: "Fukuoka"),
    .init(name: "Yamato Print", industry: "printing", location: "Osaka"),
  ]

  static let tasks: [CrmBox.FollowUpTask] = [
    .init(id: "T1", title: "Send revised quote", due: "2026-08-22", about: "O1"),
    .init(id: "T2", title: "Schedule demo", due: "2026-08-26", about: "O2"),
  ]
}

// MARK: - Tools (the rep's menu, in its words)

@available(iOS 27.0, *)
struct SearchOpportunitiesTool: Tool {
  let name = "search_opportunities"
  let description =
    "Find opportunities (deals) by any mix of company, owner, stage, amount range and close-date range; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Part of the company's name.") var company: String?
    @Guide(description: "The owner's name — one of the owners the state lists.") var owner: String?
    @Guide(description: "One stage: prospecting, qualification, proposal, negotiation, won or lost.")
    var stage: String?
    @Guide(description: "Only deals at or above this amount, in yen.") var min_amount: Int?
    @Guide(description: "Only deals at or below this amount, in yen.") var max_amount: Int?
    @Guide(description: "Only deals closing on or after this date, YYYY-MM-DD.") var close_date_from: String?
    @Guide(description: "Only deals closing on or before this date, YYYY-MM-DD.") var close_date_to: String?
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.searchOpportunities(
      company: arguments.company, owner: arguments.owner, stage: arguments.stage,
      minAmount: arguments.min_amount, maxAmount: arguments.max_amount,
      closeFrom: arguments.close_date_from, closeTo: arguments.close_date_to)
  }
}

@available(iOS 27.0, *)
struct GetOpportunityTool: Tool {
  let name = "get_opportunity"
  let description = "One opportunity's full details — contact, tasks and notes included."
  @Generable struct Arguments {
    @Guide(description: "The opportunity's id, e.g. O3, from the request or the state — no need to search first.")
    var id: String
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.getOpportunity(arguments.id)
  }
}

@available(iOS 27.0, *)
struct UpdateOpportunityStageTool: Tool {
  let name = "update_opportunity_stage"
  let description = "Move one opportunity to a different pipeline stage (won or lost closes it)."
  @Generable struct Arguments {
    @Guide(description: "The opportunity's id, e.g. O3, from the request or the state — no need to search first. If no opportunity is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The stage the deal moves to.", .anyOf(CrmBox.stages)) var stage: String
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.updateStage(id: arguments.id, stage: arguments.stage)
  }
}

@available(iOS 27.0, *)
struct UpdateOpportunityAmountTool: Tool {
  let name = "update_opportunity_amount"
  let description = "Change one opportunity's deal amount."
  @Generable struct Arguments {
    @Guide(description: "The opportunity's id, e.g. O3, from the request or the state — no need to search first. If no opportunity is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The new deal amount in yen.") var amount: Int
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.updateAmount(id: arguments.id, amount: arguments.amount)
  }
}

@available(iOS 27.0, *)
struct AssignOpportunityTool: Tool {
  let name = "assign_opportunity"
  let description = "Hand one opportunity to a different owner (sales rep)."
  @Generable struct Arguments {
    @Guide(description: "The opportunity's id, e.g. O3, from the request or the state — no need to search first. If no opportunity is identified anywhere, call ask_user instead.")
    var id: String
    @Guide(description: "The new owner's name — one of the owners the state lists.") var owner: String
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.assign(id: arguments.id, owner: arguments.owner)
  }
}

@available(iOS 27.0, *)
struct CrmSearchContactsTool: Tool {
  let name = "search_contacts"
  let description = "Find people by name, company or email; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Part of the person's name.") var name: String?
    @Guide(description: "Part of their company's name.") var company: String?
    @Guide(description: "Part of their email address.") var email: String?
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.searchContacts(name: arguments.name, company: arguments.company, email: arguments.email)
  }
}

@available(iOS 27.0, *)
struct CrmGetContactTool: Tool {
  let name = "get_contact"
  let description = "One contact's full details — their open deals and notes included."
  @Generable struct Arguments {
    @Guide(description: "The contact's id, e.g. C2, from the request or the state — no need to search first.")
    var id: String
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.getContact(arguments.id)
  }
}

@available(iOS 27.0, *)
struct CrmSearchCompaniesTool: Tool {
  let name = "search_companies"
  let description = "Find companies by name, industry or location; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Part of the company's name.") var name: String?
    @Guide(description: "The industry, e.g. food, electronics, logistics, finance.") var industry: String?
    @Guide(description: "The city.") var location: String?
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.searchCompanies(name: arguments.name, industry: arguments.industry, location: arguments.location)
  }
}

@available(iOS 27.0, *)
struct CreateFollowUpTaskTool: Tool {
  let name = "create_follow_up_task"
  let description = "Add a follow-up task, tied to an opportunity or a contact."
  @Generable struct Arguments {
    @Guide(description: "The contact's id, e.g. C2, when the task is about a person.") var contact: String?
    @Guide(description: "The opportunity's id, e.g. O3, from the request or the state — when the task is about a deal.")
    var opportunity: String?
    @Guide(description: "What to do; leave it out for a plain follow-up.") var title: String?
    @Guide(description: "When it is due, YYYY-MM-DD — count from the today the state names.") var due_date: String
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.createTask(
      contact: arguments.contact, opportunity: arguments.opportunity,
      title: arguments.title, due: arguments.due_date)
  }
}

@available(iOS 27.0, *)
struct CrmAddNoteTool: Tool {
  let name = "add_note"
  let description = "Attach a note to an opportunity or a contact."
  @Generable struct Arguments {
    @Guide(description: "The record's id from the request or the state, e.g. O3 or C2.") var entity_id: String
    @Guide(description: "The note.") var text: String
  }
  func call(arguments: Arguments) async throws -> String {
    CrmBox.shared.addNote(entityId: arguments.entity_id, text: arguments.text)
  }
}

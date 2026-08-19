// The store pack: a Shopify admin's menu, said out loud, over canned data.
//
// A different kind of agent from every pack before it. The photo and video
// packs shortcut an editor's UI; this one operates records: natural language
// → search / filter → update the rows found → a business step (fulfil,
// invoice, refund, report). The inputs are tiny — a few dozen products and
// orders — so a local model is in reach, and the bench gets a second class
// of case: was the query right, not was the edit right.
//
// The shape mirrors the admin's own: a filter or search makes a *selection*
// (the admin's checked rows), and the bulk actions — reprice, tag, change
// status, restock, fulfil, send invoices — act on that selection. "Cut their
// prices 10 %" needs no list of ids in the argument: "their" is resolved by
// the app, the way "the photo" is. The state line at the top of every
// message says what is selected and how it was found.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class StoreBox: @unchecked Sendable {
  static let shared = StoreBox()

  struct Product: Sendable {
    let id: Int
    var title: String
    var vendor: String
    var type: String
    var tags: [String]
    var price: Int  // yen
    var stock: Int
    var status: String  // active | draft | archived
  }

  struct Order: Sendable {
    let number: Int
    var customer: String
    var total: Int  // yen
    var payment: String  // paid | pending | refunded
    var fulfillment: String  // unfulfilled | fulfilled | partial
    var daysAgo: Int
    var items: String
    var note: String?
  }

  enum Selection: Sendable {
    case none
    case products([Int], how: String)
    case orders([Int], how: String)
  }

  private let lock = NSLock()
  private var products: [Product] = StoreData.products
  private var orders: [Order] = StoreData.orders
  private var selection: Selection = .none
  private var invoicesSent = 0
  /// A refund waiting on the user's yes (the confirm pattern): what the
  /// state names until refund_order is called again with confirm true.
  private var pendingConfirmation: String?
  private let history = UndoStack<([Product], [Order], Selection, String)>()

  private func pushHistory(_ what: String) {
    history.push((products, orders, selection, selectionHow), what)
  }
  private var selectionHow = ""

  func undoLast() -> String {
    guard let (snap, what) = history.pop() else { return "nothing to undo" }
    sync { (products, orders, selection, selectionHow) = snap }
    post()
    return "undid the last change (\(what))"
  }

  private func post() {
    let panel = snapshot()
    ArtifactBox.shared.post(.table(
      title: panel.title, columns: panel.columns, rows: Array(panel.rows.prefix(8))))
  }

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  static func yen(_ amount: Int) -> String {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    return "¥" + (formatter.string(from: NSNumber(value: amount)) ?? String(amount))
  }

  // MARK: The state the model reads

  func describe() -> String {
    let (products, orders, selection) = sync { (products, orders, selection) }
    let active = products.filter { $0.status == "active" }.count
    let draft = products.filter { $0.status == "draft" }.count
    let archived = products.filter { $0.status == "archived" }.count
    let unfulfilled = orders.filter { $0.fulfillment == "unfulfilled" && $0.payment != "refunded" }.count
    let pending = orders.filter { $0.payment == "pending" }.count
    var line =
      "Store: \(products.count) products (\(active) active, \(draft) draft, \(archived) archived), "
      + "\(orders.count) orders (\(unfulfilled) unfulfilled, \(pending) awaiting payment)."
    if let pending = sync({ pendingConfirmation }) { line += " Awaiting confirmation: \(pending)." }
    switch selection {
    case .none:
      line += " Selection: none — search or filter first, then act."
    case .products(let ids, let how):
      let rows = ids.compactMap { id in products.first { $0.id == id } }
      let names = rows.prefix(6).map { "\($0.title) (\(Self.yen($0.price)), stock \($0.stock))" }
      let more = rows.count > 6 ? ", and \(rows.count - 6) more" : ""
      line += " Selection: \(rows.count) product\(rows.count == 1 ? "" : "s") from \(how): "
        + names.joined(separator: "; ") + more + "."
    case .orders(let numbers, let how):
      let rows = numbers.compactMap { n in orders.first { $0.number == n } }
      let names = rows.prefix(6).map { "#\($0.number) \($0.customer) \(Self.yen($0.total)) \($0.payment)/\($0.fulfillment)" }
      let more = rows.count > 6 ? ", and \(rows.count - 6) more" : ""
      line += " Selection: \(rows.count) order\(rows.count == 1 ? "" : "s") from \(how): "
        + names.joined(separator: "; ") + more + "."
    }
    return line
  }

  /// The admin's list for the stage — the selection when there is one, the
  /// products overview otherwise.
  func snapshot() -> TablePanel {
    let (products, orders, selection) = sync { (products, orders, selection) }
    let unfulfilled = orders.filter { $0.fulfillment == "unfulfilled" && $0.payment != "refunded" }.count
    let pending = orders.filter { $0.payment == "pending" }.count
    let overview = "\(products.count) products · \(orders.count) orders · \(unfulfilled) to ship · \(pending) unpaid"
    switch selection {
    case .none:
      return TablePanel(
        title: "Products",
        columns: ["Product", "Price", "Stock", "Status"],
        rows: products.map { [$0.title, Self.yen($0.price), String($0.stock), $0.status] },
        totalRows: products.count, overview: overview)
    case .products(let ids, let how):
      let rows = ids.compactMap { id in products.first { $0.id == id } }
      return TablePanel(
        title: "\(rows.count) selected — \(how)",
        columns: ["Product", "Price", "Stock", "Status"],
        rows: rows.map { [$0.title, Self.yen($0.price), String($0.stock), $0.status] },
        totalRows: rows.count, overview: overview)
    case .orders(let numbers, let how):
      let rows = numbers.compactMap { n in orders.first { $0.number == n } }
      return TablePanel(
        title: "\(rows.count) orders — \(how)",
        columns: ["Order", "Customer", "Total", "Payment", "Fulfilment"],
        rows: rows.map { ["#\($0.number)", $0.customer, Self.yen($0.total), $0.payment, $0.fulfillment] },
        totalRows: rows.count, overview: overview)
    }
  }

  // MARK: Finding (each replaces the selection)

  private func selectProducts(_ rows: [Product], how: String) -> String {
    sync { selection = rows.isEmpty ? .none : .products(rows.map(\.id), how: how) }
    guard !rows.isEmpty else { return "no products match \(how)" }
    ArtifactBox.shared.post(.table(
      title: "\(rows.count) product\(rows.count == 1 ? "" : "s") — \(how)",
      columns: ["Product", "Price", "Stock", "Status"],
      rows: rows.map { [$0.title, Self.yen($0.price), String($0.stock), $0.status] }))
    let listed = rows.prefix(8).map { "\($0.title) — \(Self.yen($0.price)), stock \($0.stock), \($0.status)" }
    return "\(rows.count) product\(rows.count == 1 ? "" : "s") selected (\(how)):\n"
      + listed.joined(separator: "\n") + (rows.count > 8 ? "\n… and \(rows.count - 8) more" : "")
  }

  private func selectOrders(_ rows: [Order], how: String) -> String {
    sync { selection = rows.isEmpty ? .none : .orders(rows.map(\.number), how: how) }
    guard !rows.isEmpty else { return "no orders match \(how)" }
    ArtifactBox.shared.post(.table(
      title: "\(rows.count) order\(rows.count == 1 ? "" : "s") — \(how)",
      columns: ["Order", "Customer", "Total", "Payment", "Fulfilment"],
      rows: rows.map { ["#\($0.number)", $0.customer, Self.yen($0.total), $0.payment, $0.fulfillment] }))
    let listed = rows.prefix(8).map {
      "#\($0.number) \($0.customer) — \(Self.yen($0.total)), \($0.payment), \($0.fulfillment), \($0.daysAgo == 0 ? "today" : "\($0.daysAgo) d ago")"
    }
    return "\(rows.count) order\(rows.count == 1 ? "" : "s") selected (\(how)):\n"
      + listed.joined(separator: "\n") + (rows.count > 8 ? "\n… and \(rows.count - 8) more" : "")
  }

  /// By name only. Vendor, tag, type and status are `filter_products`'s —
  /// a search that also matched vendors made "everything from Hokkaido
  /// Wool" a coin toss between two tools, and a bench cannot score a coin
  /// toss.
  func searchProducts(_ query: String) -> String {
    let needle = query.lowercased().trimmingCharacters(in: .whitespaces)
    let words = needle.split(separator: " ").map(String.init)
    let rows = sync { products }.filter { p in
      needle.isEmpty || words.allSatisfy { p.title.lowercased().contains($0) }
    }
    return selectProducts(rows, how: needle.isEmpty ? "all products" : "search \"\(query)\"")
  }

  func filterProducts(by field: String, value: String) -> String {
    let want = value.lowercased().trimmingCharacters(in: .whitespaces)
    let rows = sync { products }.filter { p in
      switch field.lowercased() {
      case "status": return p.status == want
      case "vendor": return p.vendor.lowercased() == want || p.vendor.lowercased().contains(want)
      case "tag": return p.tags.contains { $0.lowercased() == want }
      case "product_type", "type": return p.type.lowercased() == want || p.type.lowercased().contains(want)
      default: return false
      }
    }
    return selectProducts(rows, how: "\(field.lowercased()) = \(value)")
  }

  func lowStock(below: Int) -> String {
    let rows = sync { products }.filter { $0.stock < below && $0.status != "archived" }
      .sorted { $0.stock < $1.stock }
    return selectProducts(rows, how: "stock below \(below)")
  }

  /// A second finder on orders — by who placed them. search vs filter is a
  /// discrimination axis on the order side too.
  func searchOrders(customer: String) -> String {
    let needle = customer.lowercased().trimmingCharacters(in: .whitespaces)
    let rows = sync { orders }.filter { $0.customer.lowercased().contains(needle) }
      .sorted { $0.daysAgo < $1.daysAgo }
    return selectOrders(rows, how: "customer \"\(customer)\"")
  }

  func filterOrders(payment: String, fulfillment: String) -> String {
    let p = payment.lowercased()
    let f = fulfillment.lowercased()
    let rows = sync { orders }.filter { o in
      (p == "any" || o.payment == p) && (f == "any" || o.fulfillment == f)
    }.sorted { $0.daysAgo < $1.daysAgo }
    var parts: [String] = []
    if p != "any" { parts.append("payment \(p)") }
    if f != "any" { parts.append("\(f)") }
    return selectOrders(rows, how: parts.isEmpty ? "all orders" : parts.joined(separator: ", "))
  }

  // MARK: Acting on the selection

  private func selectedProducts() -> [Int]? {
    if case .products(let ids, _) = sync({ selection }) { return ids }
    return nil
  }

  private func selectedOrders() -> [Int]? {
    if case .orders(let numbers, _) = sync({ selection }) { return numbers }
    return nil
  }

  private static let needProducts = "nothing is selected — search or filter products first, then act on them"
  private static let needOrders = "no orders are selected — filter orders first, then act on them"

  private func updateSelectedProducts(_ caption: String, _ change: (inout Product) -> Void) -> String {
    guard let ids = selectedProducts() else { return Self.needProducts }
    let rows: [Product] = sync {
      pushHistory(caption)
      for index in products.indices where ids.contains(products[index].id) { change(&products[index]) }
      return products.filter { ids.contains($0.id) }
    }
    ArtifactBox.shared.post(.table(
      title: "\(rows.count) product\(rows.count == 1 ? "" : "s") — \(caption)",
      columns: ["Product", "Price", "Stock", "Status"],
      rows: rows.map { [$0.title, Self.yen($0.price), String($0.stock), $0.status] }))
    let listed = rows.prefix(8).map { "\($0.title) — \(Self.yen($0.price)), stock \($0.stock), \($0.status)\($0.tags.isEmpty ? "" : ", tags: \($0.tags.joined(separator: ", "))")" }
    return "\(caption) on \(rows.count) product\(rows.count == 1 ? "" : "s"):\n" + listed.joined(separator: "\n")
  }

  func updatePrice(percent: Double) -> String {
    let factor = 1 + percent / 100
    let word = percent < 0 ? "cut" : "raised"
    return updateSelectedProducts("prices \(word) \(Self.pct(abs(percent)))%") { p in
      // Round to a price a shop would print: nearest ¥10.
      p.price = max(0, Int((Double(p.price) * factor / 10).rounded()) * 10)
    }
  }

  func setPrice(_ amount: Int) -> String {
    updateSelectedProducts("price set to \(Self.yen(amount))") { $0.price = max(0, amount) }
  }

  func addTag(_ tag: String) -> String {
    let clean = tag.trimmingCharacters(in: .whitespaces).lowercased()
    return updateSelectedProducts("tagged \"\(clean)\"") { p in
      if !p.tags.contains(clean) { p.tags.append(clean) }
    }
  }

  func setStatus(_ status: String) -> String {
    let want = status.lowercased()
    guard ["active", "draft", "archived"].contains(want) else { return "status must be active, draft or archived" }
    return updateSelectedProducts("status → \(want)") { $0.status = want }
  }

  func adjustInventory(_ change: Int) -> String {
    updateSelectedProducts("stock \(change >= 0 ? "+" : "")\(change)") { $0.stock = max(0, $0.stock + change) }
  }

  func fulfillOrders() -> String {
    guard let numbers = selectedOrders() else { return Self.needOrders }
    let (done, skipped): ([Order], [Order]) = sync {
      pushHistory("fulfilment")
      var done: [Order] = []
      var skipped: [Order] = []
      for index in orders.indices where numbers.contains(orders[index].number) {
        if orders[index].payment == "paid", orders[index].fulfillment != "fulfilled" {
          orders[index].fulfillment = "fulfilled"
          done.append(orders[index])
        } else {
          skipped.append(orders[index])
        }
      }
      return (done, skipped)
    }
    ArtifactBox.shared.post(.table(
      title: "\(done.count) order\(done.count == 1 ? "" : "s") fulfilled",
      columns: ["Order", "Customer", "Total", "Payment", "Fulfilment"],
      rows: done.map { ["#\($0.number)", $0.customer, Self.yen($0.total), $0.payment, $0.fulfillment] }))
    var text = "fulfilled \(done.count) order\(done.count == 1 ? "" : "s"): " + done.map { "#\($0.number)" }.joined(separator: ", ")
    if !skipped.isEmpty {
      text += "; skipped \(skipped.count) (not paid or already fulfilled): " + skipped.map { "#\($0.number)" }.joined(separator: ", ")
    }
    return text
  }

  func sendInvoices() -> String {
    guard let numbers = selectedOrders() else { return Self.needOrders }
    let rows = sync { orders.filter { numbers.contains($0.number) && $0.payment == "pending" } }
    guard !rows.isEmpty else { return "none of the selected orders is awaiting payment" }
    sync { invoicesSent += rows.count }
    ArtifactBox.shared.post(.table(
      title: "payment reminder sent to \(rows.count)",
      columns: ["Order", "Customer", "Total", "Days"],
      rows: rows.map { ["#\($0.number)", $0.customer, Self.yen($0.total), String($0.daysAgo)] }))
    return "sent a payment reminder for \(rows.count) order\(rows.count == 1 ? "" : "s"): "
      + rows.map { "#\($0.number) \($0.customer)" }.joined(separator: ", ")
  }

  func refund(order number: Int, confirmed: Bool) -> String {
    let result: String = sync {
      guard let index = orders.firstIndex(where: { $0.number == number }) else {
        return "there is no order #\(number)"
      }
      guard orders[index].payment == "paid" else {
        return "order #\(number) is \(orders[index].payment) — nothing to refund"
      }
      guard confirmed else {
        let o = orders[index]
        pendingConfirmation = "refund order #\(o.number) (\(o.customer), \(Self.yen(o.total)))"
        return "refunding order #\(o.number) (\(o.customer), \(Self.yen(o.total))) is permanent — ask the user to confirm, then call refund_order again with confirm true"
      }
      pendingConfirmation = nil
      pushHistory("refund")
      orders[index].payment = "refunded"
      let o = orders[index]
      ArtifactBox.shared.post(.table(
        title: "refunded #\(o.number)",
        columns: ["Order", "Customer", "Total", "Payment"],
        rows: [["#\(o.number)", o.customer, Self.yen(o.total), o.payment]]))
      return "refunded order #\(o.number) (\(o.customer), \(Self.yen(o.total)))"
    }
    return result
  }

  func addOrderNote(_ note: String) -> String {
    guard let numbers = selectedOrders() else { return Self.needOrders }
    sync {
      pushHistory("note")
      for index in orders.indices where numbers.contains(orders[index].number) {
        orders[index].note = note
      }
    }
    return "note added to \(numbers.count) order\(numbers.count == 1 ? "" : "s"): \"\(note)\""
  }

  /// The selected products (or all, when nothing is selected) as a CSV in
  /// the app's Documents — the admin's "export" button.
  func exportCSV() -> String {
    let (rows, selected): ([Product], Bool) = sync {
      if case .products(let ids, _) = selection {
        return (products.filter { ids.contains($0.id) }, true)
      }
      return (products, false)
    }
    var csv = "id,title,vendor,type,tags,price,stock,status\n"
    for p in rows {
      let title = p.title.contains(",") ? "\"\(p.title)\"" : p.title
      csv += "\(p.id),\(title),\(p.vendor),\(p.type),\(p.tags.joined(separator: "|")),\(p.price),\(p.stock),\(p.status)\n"
    }
    let url = AppFiles.documents
      .appendingPathComponent("products.csv")
    do {
      try Data(csv.utf8).write(to: url)
      return "exported \(rows.count) \(selected ? "selected " : "")product\(rows.count == 1 ? "" : "s") to products.csv in the app's Documents"
    } catch {
      return "could not write products.csv: \(error.localizedDescription)"
    }
  }

  func salesSummary(days: Int) -> String {
    let span = max(1, days)
    let rows = sync { orders }.filter { $0.daysAgo < span && $0.payment != "refunded" }
    let total = rows.reduce(0) { $0 + $1.total }
    let previous = sync { orders }.filter { $0.daysAgo >= span && $0.daysAgo < span * 2 && $0.payment != "refunded" }
    let previousTotal = previous.reduce(0) { $0 + $1.total }
    var best: [String: Int] = [:]
    for order in rows {
      for item in order.items.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) }) {
        best[item, default: 0] += 1
      }
    }
    let top = best.sorted { $0.value > $1.value }.prefix(3).map { "\($0.key) ×\($0.value)" }
    let change = previousTotal > 0 ? Int((Double(total - previousTotal) / Double(previousTotal) * 100).rounded()) : nil
    ArtifactBox.shared.post(.table(
      title: "last \(span) day\(span == 1 ? "" : "s")",
      columns: ["Orders", "Sales", "vs previous", "Top items"],
      rows: [[String(rows.count), Self.yen(total), change.map { "\($0 >= 0 ? "+" : "")\($0)%" } ?? "—", top.joined(separator: ", ")]]))
    return "last \(span) day\(span == 1 ? "" : "s"): \(rows.count) orders, \(Self.yen(total)) in sales"
      + (change.map { ", \($0 >= 0 ? "up" : "down") \(abs($0))% on the \(span) days before" } ?? "")
      + (top.isEmpty ? "" : "; top items: \(top.joined(separator: ", "))")
  }

  private static func pct(_ value: Double) -> String {
    value == value.rounded() ? String(Int(value)) : String(format: "%.1f", value)
  }
}

// MARK: - The canned store

/// A small goods shop's admin, frozen. Enough rows that filters have
/// something to find and bulk actions something to change; few enough that
/// the state line can name what is selected. Prices in yen; days count back
/// from "today", so `sales_summary(7)` means the same thing on any date.
@available(iOS 27.0, *)
enum StoreData {
  static let products: [StoreBox.Product] = [
    .init(id: 1, title: "Linen Shirt", vendor: "Kanda Goods", type: "Shirts", tags: ["summer", "new"], price: 6800, stock: 3, status: "active"),
    .init(id: 2, title: "Oxford Shirt", vendor: "Kanda Goods", type: "Shirts", tags: ["classic"], price: 7200, stock: 14, status: "active"),
    .init(id: 3, title: "Denim Jacket", vendor: "Setouchi Denim", type: "Outerwear", tags: ["classic"], price: 18000, stock: 6, status: "active"),
    .init(id: 4, title: "Selvedge Jeans", vendor: "Setouchi Denim", type: "Pants", tags: ["classic"], price: 15800, stock: 9, status: "active"),
    .init(id: 5, title: "Wide Chinos", vendor: "Kanda Goods", type: "Pants", tags: ["summer"], price: 8900, stock: 2, status: "active"),
    .init(id: 6, title: "Wool Beanie", vendor: "Hokkaido Wool", type: "Accessories", tags: ["winter"], price: 3200, stock: 0, status: "active"),
    .init(id: 7, title: "Wool Scarf", vendor: "Hokkaido Wool", type: "Accessories", tags: ["winter"], price: 5400, stock: 4, status: "active"),
    .init(id: 8, title: "Canvas Tote", vendor: "Kanda Goods", type: "Bags", tags: ["summer", "new"], price: 4200, stock: 22, status: "active"),
    .init(id: 9, title: "Leather Belt", vendor: "Kanda Goods", type: "Accessories", tags: ["classic"], price: 6500, stock: 11, status: "active"),
    .init(id: 10, title: "Indigo Bandana", vendor: "Setouchi Denim", type: "Accessories", tags: ["summer"], price: 1800, stock: 30, status: "active"),
    .init(id: 11, title: "Enamel Mug", vendor: "Kanda Goods", type: "Home", tags: ["gift"], price: 2400, stock: 18, status: "active"),
    .init(id: 12, title: "Wool Blanket", vendor: "Hokkaido Wool", type: "Home", tags: ["winter", "gift"], price: 12800, stock: 5, status: "active"),
    .init(id: 13, title: "Rain Poncho", vendor: "Kanda Goods", type: "Outerwear", tags: ["summer"], price: 5900, stock: 8, status: "active"),
    .init(id: 14, title: "Straw Hat", vendor: "Kanda Goods", type: "Accessories", tags: ["summer"], price: 4600, stock: 1, status: "active"),
    .init(id: 15, title: "Sashiko Coasters", vendor: "Setouchi Denim", type: "Home", tags: ["gift"], price: 1600, stock: 40, status: "active"),
    .init(id: 16, title: "Merino Socks", vendor: "Hokkaido Wool", type: "Accessories", tags: ["winter"], price: 1900, stock: 26, status: "active"),
    .init(id: 17, title: "Camp Shirt", vendor: "Kanda Goods", type: "Shirts", tags: ["summer"], price: 6200, stock: 7, status: "active"),
    .init(id: 18, title: "Field Jacket", vendor: "Kanda Goods", type: "Outerwear", tags: ["autumn", "new"], price: 16500, stock: 4, status: "draft"),
    .init(id: 19, title: "Corduroy Pants", vendor: "Kanda Goods", type: "Pants", tags: ["autumn", "new"], price: 9800, stock: 12, status: "draft"),
    .init(id: 20, title: "Wool Cardigan", vendor: "Hokkaido Wool", type: "Knitwear", tags: ["autumn", "new"], price: 13800, stock: 9, status: "draft"),
    .init(id: 21, title: "Flannel Shirt", vendor: "Kanda Goods", type: "Shirts", tags: ["autumn"], price: 7800, stock: 15, status: "draft"),
    .init(id: 22, title: "Linen Shorts", vendor: "Kanda Goods", type: "Pants", tags: ["summer"], price: 5600, stock: 0, status: "archived"),
    .init(id: 23, title: "Bucket Hat", vendor: "Kanda Goods", type: "Accessories", tags: ["summer"], price: 3900, stock: 0, status: "archived"),
    .init(id: 24, title: "Cotton Throw", vendor: "Kanda Goods", type: "Home", tags: ["gift"], price: 7400, stock: 6, status: "active"),
  ]

  static let orders: [StoreBox.Order] = [
    .init(number: 1001, customer: "Aoi Tanaka", total: 6800, payment: "paid", fulfillment: "fulfilled", daysAgo: 13, items: "Linen Shirt"),
    .init(number: 1002, customer: "Ben Carter", total: 22200, payment: "paid", fulfillment: "fulfilled", daysAgo: 12, items: "Denim Jacket, Canvas Tote"),
    .init(number: 1003, customer: "Chiaki Mori", total: 4200, payment: "paid", fulfillment: "fulfilled", daysAgo: 11, items: "Canvas Tote"),
    .init(number: 1004, customer: "Daniel Ross", total: 15800, payment: "paid", fulfillment: "fulfilled", daysAgo: 10, items: "Selvedge Jeans"),
    .init(number: 1005, customer: "Emi Sato", total: 3600, payment: "paid", fulfillment: "fulfilled", daysAgo: 9, items: "Indigo Bandana, Indigo Bandana"),
    .init(number: 1006, customer: "Farah Khan", total: 12800, payment: "paid", fulfillment: "fulfilled", daysAgo: 8, items: "Wool Blanket"),
    .init(number: 1007, customer: "Goro Ito", total: 7200, payment: "paid", fulfillment: "fulfilled", daysAgo: 7, items: "Oxford Shirt"),
    .init(number: 1008, customer: "Hana Kim", total: 8900, payment: "pending", fulfillment: "unfulfilled", daysAgo: 6, items: "Wide Chinos"),
    .init(number: 1009, customer: "Ivan Petrov", total: 6500, payment: "paid", fulfillment: "fulfilled", daysAgo: 6, items: "Leather Belt"),
    .init(number: 1010, customer: "Jun Watanabe", total: 24800, payment: "paid", fulfillment: "unfulfilled", daysAgo: 5, items: "Denim Jacket, Camp Shirt"),
    .init(number: 1011, customer: "Kate Lin", total: 2400, payment: "paid", fulfillment: "fulfilled", daysAgo: 4, items: "Enamel Mug"),
    .init(number: 1012, customer: "Leo Brandt", total: 5400, payment: "pending", fulfillment: "unfulfilled", daysAgo: 4, items: "Wool Scarf"),
    .init(number: 1013, customer: "Mio Hayashi", total: 11000, payment: "paid", fulfillment: "unfulfilled", daysAgo: 3, items: "Linen Shirt, Canvas Tote"),
    .init(number: 1014, customer: "Noah Silva", total: 4600, payment: "paid", fulfillment: "partial", daysAgo: 3, items: "Straw Hat, Sashiko Coasters"),
    .init(number: 1015, customer: "Olivia Park", total: 15800, payment: "pending", fulfillment: "unfulfilled", daysAgo: 2, items: "Selvedge Jeans"),
    .init(number: 1016, customer: "Pat Murphy", total: 6200, payment: "paid", fulfillment: "unfulfilled", daysAgo: 2, items: "Camp Shirt"),
    .init(number: 1017, customer: "Rin Kato", total: 3800, payment: "paid", fulfillment: "unfulfilled", daysAgo: 1, items: "Merino Socks, Merino Socks"),
    .init(number: 1018, customer: "Sam Doyle", total: 5900, payment: "pending", fulfillment: "unfulfilled", daysAgo: 1, items: "Rain Poncho"),
    .init(number: 1019, customer: "Tomo Nakamura", total: 7400, payment: "paid", fulfillment: "unfulfilled", daysAgo: 0, items: "Cotton Throw"),
    .init(number: 1020, customer: "Uma Reddy", total: 1800, payment: "pending", fulfillment: "unfulfilled", daysAgo: 0, items: "Indigo Bandana"),
  ]
}

// MARK: - Tools (the admin's menu, in its words)

@available(iOS 27.0, *)
struct SearchProductsTool: Tool {
  let name = "search_products"
  let description =
    "Find products by words in the product name; the matches become the selection. Vendors, tags, types and statuses are filter_products' job."
  @Generable struct Arguments {
    @Guide(description: "Words from the product name, e.g. linen shirt.") var query: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.searchProducts(arguments.query)
  }
}

@available(iOS 27.0, *)
struct FilterProductsTool: Tool {
  let name = "filter_products"
  let description = "Filter products by one field; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Which field.", .anyOf(["status", "vendor", "tag", "product_type"])) var by: String
    @Guide(description: "The value to match, e.g. draft, Kanda Goods, summer, Shirts.") var value: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.filterProducts(by: arguments.by, value: arguments.value)
  }
}

@available(iOS 27.0, *)
struct LowStockTool: Tool {
  let name = "find_low_stock"
  let description = "Products with inventory below a number; they become the selection."
  @Generable struct Arguments {
    @Guide(description: "Stock threshold: products with fewer units than this.") var below: Int
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.lowStock(below: arguments.below)
  }
}

@available(iOS 27.0, *)
struct UpdatePriceTool: Tool {
  let name = "update_price"
  let description = "Change the price of the selected products by a percentage."
  @Generable struct Arguments {
    @Guide(description: "Percent change: -10 lowers prices by 10%, 15 raises them by 15%.")
    var percent_change: Double
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.updatePrice(percent: arguments.percent_change)
  }
}

@available(iOS 27.0, *)
struct SetPriceTool: Tool {
  let name = "set_price"
  let description = "Set the selected products to an exact price."
  @Generable struct Arguments {
    @Guide(description: "New price in yen.") var price: Int
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.setPrice(arguments.price)
  }
}

@available(iOS 27.0, *)
struct AddTagTool: Tool {
  let name = "add_tag"
  let description = "Add a tag to the selected products."
  @Generable struct Arguments {
    @Guide(description: "The tag, e.g. clearance.") var tag: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.addTag(arguments.tag)
  }
}

@available(iOS 27.0, *)
struct SetProductStatusTool: Tool {
  let name = "set_product_status"
  let description = "Publish, unpublish or archive the selected products."
  @Generable struct Arguments {
    @Guide(description: "New status.", .anyOf(["active", "draft", "archived"])) var status: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.setStatus(arguments.status)
  }
}

@available(iOS 27.0, *)
struct AdjustInventoryTool: Tool {
  let name = "adjust_inventory"
  let description = "Add to or remove from the stock of the selected products."
  @Generable struct Arguments {
    @Guide(description: "Units to add; negative removes. 20 adds twenty units.") var change: Int
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.adjustInventory(arguments.change)
  }
}

@available(iOS 27.0, *)
struct FilterOrdersTool: Tool {
  let name = "filter_orders"
  let description = "Filter orders by payment and fulfilment status; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Payment status, or any.", .anyOf(["any", "paid", "pending", "refunded"]))
    var payment_status: String
    @Guide(description: "Fulfilment status, or any.", .anyOf(["any", "unfulfilled", "fulfilled", "partial"]))
    var fulfillment_status: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.filterOrders(payment: arguments.payment_status, fulfillment: arguments.fulfillment_status)
  }
}

@available(iOS 27.0, *)
struct SearchOrdersTool: Tool {
  let name = "search_orders"
  let description = "Find orders by customer name; the matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "Part of the customer's name, e.g. Tanaka.") var customer: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.searchOrders(customer: arguments.customer)
  }
}

@available(iOS 27.0, *)
struct ExportProductsTool: Tool {
  let name = "export_products_csv"
  let description = "Export the selected products (or all products) to a CSV file."
  func call(arguments: NoArguments) async throws -> String {
    StoreBox.shared.exportCSV()
  }
}

@available(iOS 27.0, *)
struct FulfillOrdersTool: Tool {
  let name = "fulfill_orders"
  let description = "Mark the selected orders as fulfilled (shipped)."
  func call(arguments: NoArguments) async throws -> String {
    StoreBox.shared.fulfillOrders()
  }
}

@available(iOS 27.0, *)
struct SendInvoiceTool: Tool {
  let name = "send_payment_reminder"
  let description = "Email a payment reminder (invoice) for the selected orders that are still unpaid."
  func call(arguments: NoArguments) async throws -> String {
    StoreBox.shared.sendInvoices()
  }
}

@available(iOS 27.0, *)
struct RefundOrderTool: Tool {
  let name = "refund_order"
  let description = "Refund one order in full."
  @Generable struct Arguments {
    @Guide(description: "The order number, e.g. 1007, from the request or the state. If no order is identified anywhere, call ask_user instead.")
    var order_number: Int
    @Guide(description: "Pass false unless the user has already said yes to this exact refund; false shows them what will happen so they can decide.")
    var confirm: Bool
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.refund(order: arguments.order_number, confirmed: arguments.confirm)
  }
}

@available(iOS 27.0, *)
struct OrderNoteTool: Tool {
  let name = "add_order_note"
  let description = "Attach a note to the selected orders."
  @Generable struct Arguments {
    @Guide(description: "The note.") var note: String
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.addOrderNote(arguments.note)
  }
}

@available(iOS 27.0, *)
struct SalesSummaryTool: Tool {
  let name = "sales_summary"
  let description = "Orders and sales for the last N days, compared with the N days before."
  @Generable struct Arguments {
    @Guide(description: "How many days back. 7 is this week, 30 this month.") var days: Int
  }
  func call(arguments: Arguments) async throws -> String {
    StoreBox.shared.salesSummary(days: arguments.days)
  }
}

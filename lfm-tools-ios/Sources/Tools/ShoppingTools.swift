// The shopping pack: a buyer's store (the Amazon type), said out loud.
//
// The other side of the counter from the store pack: not an admin operating
// records, a customer finding and buying. Search → sort → "the second one" →
// cart → coupon → checkout. The state names the numbered results and the
// cart with its total, so "add the second one" is a number the model reads,
// and the cart maths is the app's, never the model's.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class ShopBox: @unchecked Sendable {
  static let shared = ShopBox()

  struct Item: Sendable {
    let id: Int
    let name: String
    let brand: String
    let category: String
    let price: Int  // yen
    let rating: Double
    let reviews: Int
    let shipsToday: Bool
  }

  struct CartLine: Sendable {
    let itemID: Int
    var quantity: Int
  }

  private let lock = NSLock()
  /// The last search, in the order the sort left it. Numbers in the state
  /// are 1-based positions in this list.
  private var results: [Item] = []
  private var resultsHow = ""
  private var cart: [CartLine] = []
  private var savedForLater: [Int] = []
  private var coupon: (code: String, percent: Int)?
  private var ordersPlaced = 0

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  // MARK: The state the model reads

  func describe() -> String {
    let (results, how, cart, coupon) = sync { (results, resultsHow, cart, coupon) }
    var line = ""
    if results.isEmpty {
      line = "Results: none yet — search the catalog first."
    } else {
      let listed = results.prefix(6).enumerated().map { index, item in
        "\(index + 1) \(item.name) \(StoreBox.yen(item.price)) ★\(String(format: "%.1f", item.rating))"
      }
      let more = results.count > 6 ? ", and \(results.count - 6) more" : ""
      line = "Results (\(how)): " + listed.joined(separator: "; ") + more + "."
    }
    if cart.isEmpty {
      line += " Cart: empty."
    } else {
      let names = cart.compactMap { line -> String? in
        guard let item = Catalog.item(line.itemID) else { return nil }
        return "\(item.name) ×\(line.quantity)"
      }
      line += " Cart: \(names.joined(separator: "; ")) — subtotal \(StoreBox.yen(subtotal()))"
      if let coupon { line += ", coupon \(coupon.code) −\(coupon.percent)%" }
      line += ", total \(StoreBox.yen(total()))."
    }
    return line
  }

  private func subtotal() -> Int {
    cart.reduce(0) { sum, line in
      sum + (Catalog.item(line.itemID)?.price ?? 0) * line.quantity
    }
  }

  private func total() -> Int {
    let sub = subtotal()
    guard let coupon else { return sub }
    return Int((Double(sub) * (1 - Double(coupon.percent) / 100) / 10).rounded()) * 10
  }

  // MARK: The panel

  func snapshot() -> TablePanel {
    let (results, how, cart) = sync { (results, resultsHow, cart) }
    let overview = cart.isEmpty
      ? "cart empty"
      : "cart \(cart.reduce(0) { $0 + $1.quantity }) item\(cart.count == 1 ? "" : "s") · \(StoreBox.yen(sync { total() }))"
    if !cart.isEmpty && results.isEmpty {
      return TablePanel(
        title: "Cart",
        columns: ["Item", "Qty", "Price"],
        rows: cart.compactMap { line in
          guard let item = Catalog.item(line.itemID) else { return nil }
          return [item.name, "×\(line.quantity)", StoreBox.yen(item.price * line.quantity)]
        },
        totalRows: cart.count, overview: overview)
    }
    return TablePanel(
      title: results.isEmpty ? "Catalog" : "Results — \(how)",
      columns: ["#", "Item", "Price", "Rating", "Ships"],
      rows: (results.isEmpty ? Catalog.items : results).enumerated().map { index, item in
        ["\(index + 1)", item.name, StoreBox.yen(item.price),
         "★\(String(format: "%.1f", item.rating)) (\(item.reviews))",
         item.shipsToday ? "today" : "2–3 d"]
      },
      totalRows: results.isEmpty ? Catalog.items.count : results.count, overview: overview)
  }

  private func post() {
    ArtifactBox.shared.post(.table(
      title: snapshot().title, columns: snapshot().columns,
      rows: Array(snapshot().rows.prefix(8))))
  }

  // MARK: Finding

  func search(_ query: String) -> String {
    let words = query.lowercased().split(separator: " ").map(String.init)
    let rows = Catalog.items.filter { item in
      words.allSatisfy {
        item.name.lowercased().contains($0) || item.brand.lowercased().contains($0)
          || item.category.lowercased().contains($0)
      }
    }
    sync {
      results = rows
      resultsHow = "search \"\(query)\""
    }
    post()
    guard !rows.isEmpty else { return "nothing in the catalog matches \"\(query)\"" }
    let listed = rows.prefix(6).enumerated().map { index, item in
      "\(index + 1). \(item.name) (\(item.brand)) — \(StoreBox.yen(item.price)), ★\(String(format: "%.1f", item.rating)) from \(item.reviews) reviews\(item.shipsToday ? ", ships today" : "")"
    }
    return "\(rows.count) result\(rows.count == 1 ? "" : "s") for \"\(query)\":\n" + listed.joined(separator: "\n")
  }

  func sort(by field: String) -> String {
    let rows = sync { results }
    guard !rows.isEmpty else { return "search first, then sort the results" }
    let sorted: [Item]
    let how: String
    switch field.lowercased() {
    case "price_low": sorted = rows.sorted { $0.price < $1.price }; how = "cheapest first"
    case "price_high": sorted = rows.sorted { $0.price > $1.price }; how = "most expensive first"
    default: sorted = rows.sorted { $0.rating > $1.rating }; how = "best rated first"
    }
    sync {
      results = sorted
      resultsHow += ", \(how)"
    }
    post()
    let listed = sorted.prefix(6).enumerated().map { index, item in
      "\(index + 1). \(item.name) — \(StoreBox.yen(item.price)), ★\(String(format: "%.1f", item.rating))"
    }
    return "sorted \(how):\n" + listed.joined(separator: "\n")
  }

  private func item(atResult number: Int) -> Item? {
    let rows = sync { results }
    guard rows.indices.contains(number - 1) else { return nil }
    return rows[number - 1]
  }

  func show(_ number: Int) -> String {
    guard let item = item(atResult: number) else {
      return "there is no result \(number); the list has \(sync { results }.count)"
    }
    return "\(item.name) by \(item.brand) — \(StoreBox.yen(item.price)), ★\(String(format: "%.1f", item.rating)) from \(item.reviews) reviews, category \(item.category), \(item.shipsToday ? "ships today" : "ships in 2–3 days")"
  }

  // MARK: The cart

  func addToCart(result number: Int, quantity: Int) -> String {
    guard let item = item(atResult: number) else {
      return "there is no result \(number); search first, then add by its number"
    }
    let count = max(1, quantity)
    sync {
      if let index = cart.firstIndex(where: { $0.itemID == item.id }) {
        cart[index].quantity += count
      } else {
        cart.append(CartLine(itemID: item.id, quantity: count))
      }
    }
    post()
    return "added \(item.name) ×\(count) — cart total \(StoreBox.yen(sync { total() }))"
  }

  private func cartLine(named name: String) -> Int? {
    let needle = name.lowercased()
    return sync {
      cart.firstIndex { line in
        guard let item = Catalog.item(line.itemID) else { return false }
        return item.name.lowercased().contains(needle) || needle.contains(item.name.lowercased())
      }
    }
  }

  func changeQuantity(itemNamed name: String, to quantity: Int) -> String {
    guard let index = cartLine(named: name) else {
      return "nothing in the cart matches \"\(name)\""
    }
    let itemName = Catalog.item(sync { cart[index].itemID })?.name ?? name
    if quantity <= 0 {
      sync { cart.remove(at: index) }
      post()
      return "removed \(itemName) from the cart — total \(StoreBox.yen(sync { total() }))"
    }
    sync { cart[index].quantity = quantity }
    post()
    return "\(itemName) is now ×\(quantity) — cart total \(StoreBox.yen(sync { total() }))"
  }

  func removeFromCart(itemNamed name: String) -> String {
    changeQuantity(itemNamed: name, to: 0)
  }

  func applyCoupon(_ code: String) -> String {
    let want = code.uppercased().trimmingCharacters(in: .whitespaces)
    guard let percent = Catalog.coupons[want] else {
      return "coupon \(want) is not valid; nothing applied"
    }
    sync { coupon = (want, percent) }
    post()
    return "coupon \(want) applied: −\(percent)% — total \(StoreBox.yen(sync { total() }))"
  }

  func checkout() -> String {
    let lines = sync { cart }
    guard !lines.isEmpty else { return "the cart is empty — nothing to order" }
    let paid = sync { total() }
    let count = lines.reduce(0) { $0 + $1.quantity }
    let orderNumber: Int = sync {
      ordersPlaced += 1
      cart = []
      coupon = nil
      return 5230 + ordersPlaced
    }
    post()
    return "order #\(orderNumber) placed: \(count) item\(count == 1 ? "" : "s"), \(StoreBox.yen(paid)) — arriving in 2 days"
  }

  func trackOrder(_ number: Int) -> String {
    // The canned world has one order already on its way.
    if number == 5230 {
      return "order #5230 (USB-C cable ×2) is out for delivery — arriving today by 21:00"
    }
    if number > 5230 {
      return "order #\(number) is being packed — arriving in 2 days"
    }
    return "there is no order #\(number)"
  }
}

/// A small catalog, frozen. Enough that a search finds several things and a
/// sort visibly reorders them; prices in yen.
enum Catalog {
  static let items: [ShopBox.Item] = [
    .init(id: 1, name: "Wireless Earbuds Pro", brand: "Soundcore", category: "Audio", price: 12800, rating: 4.5, reviews: 8231, shipsToday: true),
    .init(id: 2, name: "Wireless Earbuds Lite", brand: "Soundcore", category: "Audio", price: 4990, rating: 4.2, reviews: 15402, shipsToday: true),
    .init(id: 3, name: "Noise Cancelling Earbuds", brand: "Sony", category: "Audio", price: 24800, rating: 4.7, reviews: 5120, shipsToday: false),
    .init(id: 4, name: "Budget Earbuds", brand: "JVC", category: "Audio", price: 2480, rating: 3.9, reviews: 20933, shipsToday: true),
    .init(id: 5, name: "Over-Ear Headphones", brand: "Audio-Technica", category: "Audio", price: 9800, rating: 4.4, reviews: 3311, shipsToday: true),
    .init(id: 6, name: "USB-C Cable 2 m", brand: "Anker", category: "Accessories", price: 1290, rating: 4.6, reviews: 44210, shipsToday: true),
    .init(id: 7, name: "USB-C Charger 65 W", brand: "Anker", category: "Accessories", price: 4490, rating: 4.6, reviews: 18777, shipsToday: true),
    .init(id: 8, name: "Power Bank 10000", brand: "Anker", category: "Accessories", price: 3990, rating: 4.5, reviews: 30125, shipsToday: true),
    .init(id: 9, name: "Phone Stand", brand: "Lamicall", category: "Accessories", price: 1590, rating: 4.4, reviews: 9214, shipsToday: true),
    .init(id: 10, name: "Mechanical Keyboard", brand: "Keychron", category: "Computers", price: 13800, rating: 4.5, reviews: 2811, shipsToday: false),
    .init(id: 11, name: "Wireless Mouse", brand: "Logicool", category: "Computers", price: 5980, rating: 4.6, reviews: 12083, shipsToday: true),
    .init(id: 12, name: "4K Webcam", brand: "Logicool", category: "Computers", price: 16800, rating: 4.3, reviews: 1904, shipsToday: false),
    .init(id: 13, name: "Laptop Sleeve 14\"", brand: "tomtoc", category: "Computers", price: 2990, rating: 4.7, reviews: 6633, shipsToday: true),
    .init(id: 14, name: "Drip Coffee Maker", brand: "HARIO", category: "Kitchen", price: 3480, rating: 4.5, reviews: 7521, shipsToday: true),
    .init(id: 15, name: "Coffee Grinder", brand: "HARIO", category: "Kitchen", price: 4950, rating: 4.4, reviews: 5218, shipsToday: true),
    .init(id: 16, name: "Electric Kettle 0.8 L", brand: "BALMUDA", category: "Kitchen", price: 13200, rating: 4.6, reviews: 3120, shipsToday: false),
    .init(id: 17, name: "Insulated Bottle 500 ml", brand: "Zojirushi", category: "Kitchen", price: 3280, rating: 4.6, reviews: 11255, shipsToday: true),
    .init(id: 18, name: "Running Shoes", brand: "ASICS", category: "Sports", price: 8990, rating: 4.4, reviews: 4187, shipsToday: true),
    .init(id: 19, name: "Yoga Mat 10 mm", brand: "adidas", category: "Sports", price: 3590, rating: 4.3, reviews: 8022, shipsToday: true),
    .init(id: 20, name: "Resistance Bands Set", brand: "TheFitLife", category: "Sports", price: 2180, rating: 4.2, reviews: 6120, shipsToday: true),
  ]

  static let coupons = ["SAVE10": 10, "HELLO5": 5]

  static func item(_ id: Int) -> ShopBox.Item? { items.first { $0.id == id } }
}

// MARK: - Tools (the shop's screens, in its words)

@available(iOS 27.0, *)
struct SearchCatalogTool: Tool {
  let name = "search_catalog"
  let description = "Search the shop for products; the results are numbered."
  @Generable struct Arguments {
    @Guide(description: "What to look for, e.g. wireless earbuds.") var query: String
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.search(arguments.query)
  }
}

@available(iOS 27.0, *)
struct SortResultsTool: Tool {
  let name = "sort_results"
  let description = "Reorder the current search results."
  @Generable struct Arguments {
    @Guide(description: "Sort order.", .anyOf(["price_low", "price_high", "rating"])) var by: String
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.sort(by: arguments.by)
  }
}

@available(iOS 27.0, *)
struct ShowProductTool: Tool {
  let name = "show_product"
  let description = "Open one result to see its details."
  @Generable struct Arguments {
    @Guide(description: "The result's number in the list, from 1.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.show(arguments.number)
  }
}

@available(iOS 27.0, *)
struct AddToCartTool: Tool {
  let name = "add_to_cart"
  let description = "Put a result in the cart, by its number in the list."
  @Generable struct Arguments {
    @Guide(description: "The result's number in the list, from 1.") var number: Int
    @Guide(description: "How many. 1 unless the user says otherwise.") var quantity: Int
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.addToCart(result: arguments.number, quantity: arguments.quantity)
  }
}

@available(iOS 27.0, *)
struct ChangeQuantityTool: Tool {
  let name = "change_quantity"
  let description = "Change how many of a cart item to buy. 0 removes it."
  @Generable struct Arguments {
    @Guide(description: "The item, by (part of) its name as the cart shows it.") var item: String
    @Guide(description: "The new quantity; 0 removes it.") var quantity: Int
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.changeQuantity(itemNamed: arguments.item, to: arguments.quantity)
  }
}

@available(iOS 27.0, *)
struct RemoveFromCartTool: Tool {
  let name = "remove_from_cart"
  let description = "Take an item out of the cart."
  @Generable struct Arguments {
    @Guide(description: "The item, by (part of) its name as the cart shows it.") var item: String
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.removeFromCart(itemNamed: arguments.item)
  }
}

@available(iOS 27.0, *)
struct ApplyCouponTool: Tool {
  let name = "apply_coupon"
  let description = "Apply a discount code to the cart."
  @Generable struct Arguments {
    @Guide(description: "The code, exactly as given, e.g. SAVE10.") var code: String
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.applyCoupon(arguments.code)
  }
}

@available(iOS 27.0, *)
struct CheckoutTool: Tool {
  let name = "checkout"
  let description = "Place the order for everything in the cart."
  func call(arguments: NoArguments) async throws -> String {
    ShopBox.shared.checkout()
  }
}

@available(iOS 27.0, *)
struct TrackOrderTool: Tool {
  let name = "track_order"
  let description = "Where an order is right now."
  @Generable struct Arguments {
    @Guide(description: "The order number, e.g. 5230.") var order_number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    ShopBox.shared.trackOrder(arguments.order_number)
  }
}

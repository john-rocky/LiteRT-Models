// One benchmark case: an input, and the tool calls a correct run makes.
//
// The file format is JSON, one array of cases, pushed to the app's Documents
// as `toolbench-cases.json`. `expected: []` is a real case, not a gap — a
// model that calls a tool when none applies fails it.
import Foundation

struct BenchCase: Decodable {
  var id: String
  var input: String
  var lang: String
  /// Reserved for the VLM stage: a fixture image name.
  var image: String?
  var expected: [ExpectedCall]
}

struct ExpectedCall: Decodable {
  var tool: String
  /// Argument name → matcher. Absent means the call's arguments are not
  /// scored — deliberate for cases where two phrasings are both right
  /// (a JP request may legitimately produce a JP or an EN query string).
  var args: [String: Matcher]?
}

/// The four matchers the case format allows. Anything richer belongs in a
/// purpose-written check, not in JSON.
enum Matcher: Decodable {
  case equals(String)
  case contains(String)
  case number(Double, tol: Double)
  case dateResolvesTo(String)

  private enum Keys: String, CodingKey {
    case equals, contains, number, tol, dateResolvesTo
  }

  init(from decoder: Decoder) throws {
    let box = try decoder.container(keyedBy: Keys.self)
    if let value = try box.decodeIfPresent(String.self, forKey: .equals) {
      self = .equals(value)
    } else if let value = try box.decodeIfPresent(String.self, forKey: .contains) {
      self = .contains(value)
    } else if let value = try box.decodeIfPresent(Double.self, forKey: .number) {
      self = .number(value, tol: try box.decodeIfPresent(Double.self, forKey: .tol) ?? 0)
    } else if let value = try box.decodeIfPresent(String.self, forKey: .dateResolvesTo) {
      self = .dateResolvesTo(value)
    } else {
      throw DecodingError.dataCorrupted(
        .init(
          codingPath: decoder.codingPath,
          debugDescription: "matcher needs equals / contains / number / dateResolvesTo"))
    }
  }

  /// Matching is case-insensitive throughout: argument casing is the model's
  /// prose style, not a correctness signal.
  func matches(_ actual: Any?) -> Bool {
    guard let actual else { return false }
    let text = Self.text(of: actual)
    switch self {
    case .equals(let want):
      return text.compare(want, options: .caseInsensitive) == .orderedSame
    case .contains(let want):
      return text.range(of: want, options: .caseInsensitive) != nil
    case .number(let want, let tol):
      guard let value = actual as? Double ?? Double(text) else { return false }
      return abs(value - want) <= tol
    case .dateResolvesTo(let relative):
      guard let target = Self.resolve(relative) else { return false }
      // The actual value only has to name that day: "2026-08-19",
      // "2026-08-19T09:00" and "August 19" all resolve to tomorrow.
      let formatter = DateFormatter()
      formatter.dateFormat = "yyyy-MM-dd"
      if text.hasPrefix(formatter.string(from: target)) { return true }
      formatter.dateFormat = "MMMM d"
      return text.range(of: formatter.string(from: target), options: .caseInsensitive) != nil
    }
  }

  private static func text(of value: Any) -> String {
    if let string = value as? String { return string }
    return String(describing: value)
  }

  private static func resolve(_ relative: String) -> Date? {
    let calendar = Calendar.current
    switch relative.lowercased() {
    case "today": return Date()
    case "tomorrow": return calendar.date(byAdding: .day, value: 1, to: Date())
    case "yesterday": return calendar.date(byAdding: .day, value: -1, to: Date())
    default: return nil
    }
  }
}

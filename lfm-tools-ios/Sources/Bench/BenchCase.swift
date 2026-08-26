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
  /// For the state packs (video): the app state the message opens with,
  /// verbatim — the same block the stage sends ahead of the words. Cases
  /// that name a playhead or a clip edge are only scorable against it.
  var state: String?
  /// The ask-back axis: true means the input deliberately omits a required
  /// argument and a correct run calls nothing and asks a question back.
  /// `expected` stays [] on these cases; the runner additionally requires a
  /// question in the answer.
  var expectAsk: Bool?
  /// The goal-driven loop (polish): the runner re-attaches the edited photo
  /// after every round and asks the model to judge again, until a round
  /// makes no tool call or `maxRounds` (default 4) cuts it off. `expected`
  /// stays [] — a loop case is scored on `needs`/`avoid`, the fixture's
  /// ground truth: ops of which a correct run makes at least one, and ops a
  /// correct run never makes. Empty `needs` means the photo needs nothing —
  /// a correct run stops in round one without a call.
  var loop: Bool?
  var maxRounds: Int?
  var needs: [OpAxis]?
  var avoid: [OpAxis]?
  /// The perception control: the pixels are the question and the answer is
  /// prose. Any listed keyword in the answer passes (case-insensitive);
  /// `expected` still gates the calls — usually [], because an edit is not
  /// an answer. Separates "cannot see the defect" from "sees it but the
  /// judgment does not steer the op".
  var answerContains: [String]?
  var expected: [ExpectedCall]
}

/// One op family on the loop's ground-truth axes: a tool, optionally pinned
/// to a direction ("brighter", "up", "warmer"…). Strength is never scored —
/// gentleness is the pack's character; the fixture's defect names only the
/// direction that fixes it.
struct OpAxis: Decodable {
  var tool: String
  var direction: String?
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
    // JSONSerialization hands booleans back as NSNumber; described, that is
    // "1", and a case's {"equals": "true"} would never match.
    if let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() {
      return number.boolValue ? "true" : "false"
    }
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

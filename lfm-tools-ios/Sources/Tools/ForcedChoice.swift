// The forced-choice verdict matcher, in one place.
//
// A tool that looks at one item and answers a question about it with one of
// the options it was given is now the third instance of the same job
// (check_moment's real index, check_moment's canned echo, check_photo). The
// semantics are the moment-seek lane's, bought at a high price and written up
// in docs/demo-playbook.md specs A and D2:
//
//   1. Partition the options by negation markers — EN and JA both, because a
//      JA pair reads all-positive without them and the model then reports the
//      thing absent (r38, m-ja-check-2).
//   2. Direct match on the positive options first: option text against the
//      truths, containment either way.
//   3. Otherwise presence is decided by the content words of the question and
//      the positive options — with a floor that keeps a short token carrying a
//      digit or written as an acronym, and a boundary that splits where a
//      search tokenizer splits so "1-0" reaches the truths whole.
//   4. A question this code cannot evaluate must not vote no. No content word
//      outside the wrapper nouns, or none that could appear in the truths at
//      all, means the check says it cannot tell and lists what it saw. A
//      confident no from a tool that never read the question is the failure
//      this lane keeps rediscovering (verification vetoes retrieval).
//
// The verdict word always leads and the evidence tail follows: answers follow
// the verdict word, not the evidence list.
//
// The two moment-seek copies stay where they are on purpose — they are the
// control r38–r45 measured, and a shared helper that changed one character of
// their behavior would silently re-base nine rounds. New packs use this one.
import Foundation

@available(iOS 27.0, *)
enum ForcedChoice {
  /// Words that build a question's frame without naming what it asks about.
  /// A question left holding only these is being tested for a word it never
  /// asked about: the absence is real and the verdict is not. Their own list
  /// rather than more stopwords — a stopword is dropped and the rest of the
  /// question still carries it.
  static let wrappers: Set<String> = [
    "scene", "moment", "part", "section", "place", "spot", "thing", "area",
    "photo", "picture", "image", "shot",
    "場面", "瞬間", "部分", "箇所", "ところ", "写真", "画像",
  ]

  static let stopWords: Set<String> = [
    "the", "there", "this", "that", "does", "did", "is", "are", "was", "were", "and",
    "not", "frame", "video", "clip", "show", "shows", "shown", "appear",
    "appears", "visible", "have", "has", "any", "still", "you", "can", "see", "around",
    "second", "seconds", "present", "yes", "true", "what", "which", "contain", "contains",
    "with", "who", "where", "when", "taken", "number",
  ]

  static func negated(_ option: String) -> Bool {
    // JA carries the negation inside the word and writes no spaces, so these
    // match as bare substrings. 「ません」 is the polite negation ありません/
    // いません are two instances of; 無い/無し are the kanji spellings.
    for marker in [
      "なし", "ない", "いない", "ありません", "いません",
      "いいえ", "ません", "無い", "無し",
    ] where option.contains(marker) { return true }
    let o = " " + option.lowercased() + " "
    return o.contains(" no ") || o.contains(" not ") || o.contains("n't ")
      || o.contains(" none ") || o.contains(" without ") || o.contains(" nothing ")
  }

  /// Same separators the search tokenizers use, so a question naming "1-0"
  /// meets a truth's own "1-0" instead of arriving as "1" and "0".
  static func contentWords(_ text: String) -> [String] {
    text.split(whereSeparator: { " ,.!?'\"「」『』、。".contains($0) })
      .map { (typed: String($0), word: String($0).lowercased()) }
      .filter { token in
        guard !stopWords.contains(token.word) else { return false }
        return token.word.count >= 3 || token.word.contains(where: \.isNumber)
          || (token.typed.count >= 2 && token.typed.allSatisfy { $0.isUppercase })
      }
      .map(\.word)
  }

  /// - Parameters:
  ///   - truths: what the item actually holds, lowercased by the caller.
  ///   - shows: the evidence tail's opening — " — photo 4 shows: " and the
  ///     like. The caller owns the wording; this owns the verdict.
  ///   - aliases: the pack's findability table, applied to the question and
  ///     the positive options so a JA noun can reach an English truth. JA
  ///     writes no spaces, so without it the presence test is blind and the
  ///     cannot-tell branch is the only honest answer left.
  static func answer(
    question: String, options: [String], truths: [String], shows: String,
    aliases: [(String, String)] = []
  ) -> String {
    let evidence = truths.prefix(8).joined(separator: ", ")
    guard !truths.isEmpty else { return "cannot tell" + shows + "nothing" }
    let positives = options.filter { !negated($0) }
    let negatives = options.filter { negated($0) }
    if let hit = positives.first(where: { option in
      let o = option.lowercased()
      return truths.contains { $0.contains(o) || o.contains($0) }
    }) {
      return hit + shows + truths.prefix(6).joined(separator: ", ")
    }
    var words = contentWords(question) + positives.flatMap(contentWords)
    let asked = ([question] + positives).joined(separator: " ")
    for (from, to) in aliases where asked.contains(from) { words.append(to) }
    let named = words.filter { !wrappers.contains($0) }
    let testable = named.contains { word in
      word.contains { $0.isASCII && ($0.isLetter || $0.isNumber) }
    }
    guard testable else { return "cannot tell" + shows + evidence }
    let present = words.contains { word in truths.contains { $0.contains(word) } }
    if present {
      let verdict = positives.first { !["yes", "true"].contains($0.lowercased()) } ?? "yes"
      return verdict + shows + truths.prefix(6).joined(separator: ", ")
    }
    if let no = negatives.first
      ?? options.first(where: { ["no", "false"].contains($0.lowercased()) })
    {
      return no + shows + truths.prefix(6).joined(separator: ", ")
    }
    return "none of those" + shows + evidence
  }
}

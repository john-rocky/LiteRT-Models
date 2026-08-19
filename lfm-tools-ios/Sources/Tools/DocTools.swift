// The documents pack: an Acrobat / Goodnotes menu, said out loud, on a real PDF.
//
// PDFKit does the work — the document is a PDFDocument, pages are removed,
// moved, rotated and inserted for real, highlights and notes are
// PDFAnnotations, the signature is an ink annotation, save writes a file.
// The tools are the app's own menu items in its own words. State in, tools
// out: every message opens with the document as it is — how many pages,
// which one is open, each page's title, what is annotated — so "delete the
// cover page" is a page number the model reads, and "sign the last page"
// too. The stage shows the open page, and the pages as a strip under it.
//
// The document is the newest PDF in the app's Documents (drop one in with
// Files.app), or, when there is none, a six-page lease agreement the app
// draws itself — text enough for highlights and searches to find.
import Foundation
import FoundationModels
import PDFKit
import UIKit

@available(iOS 27.0, *)
final class DocBox: @unchecked Sendable {
  static let shared = DocBox()

  struct Snapshot: @unchecked Sendable {
    let title: String
    let pageCount: Int
    let current: Int  // 1-based
    let titles: [String]
    let annotated: [Int: String]  // page number → short summary
    let thumbnails: [UIImage]
  }

  enum Failure: LocalizedError {
    case noDocument
    var errorDescription: String? { "no document is open" }
  }

  private let lock = NSLock()
  private var document: PDFDocument?
  private var originalData: Data?
  private var title = ""
  private var current = 0  // 0-based
  private var thumbnails: [UIImage] = []

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  var isLoaded: Bool { sync { document != nil } }

  // MARK: Loading

  func preload() {
    if isLoaded { return }
    let documents = AppFiles.documents
    let pdfs = ((try? FileManager.default.contentsOfDirectory(
      at: documents, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
      .filter { $0.pathExtension.lowercased() == "pdf" && !$0.lastPathComponent.hasPrefix("saved-") }
      .sorted {
        let a = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        let b = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        return a > b
      }
    if let url = pdfs.first, let data = try? Data(contentsOf: url), let doc = PDFDocument(data: data) {
      load(doc, data: data, title: url.deletingPathExtension().lastPathComponent)
    } else {
      let data = SampleLease.pdfData()
      if let doc = PDFDocument(data: data) { load(doc, data: data, title: "Lease Agreement") }
    }
  }

  private func load(_ doc: PDFDocument, data: Data, title: String) {
    sync {
      document = doc
      originalData = data
      self.title = title
      current = 0
      thumbnails = []
    }
    refreshThumbnails()
  }

  // MARK: The state the model reads

  func describe() -> String {
    let (document, title, current) = sync { (self.document, self.title, self.current) }
    guard let doc = document else { return "No document open." }
    let count = doc.pageCount
    var line = "Document: \"\(title)\", \(count) page\(count == 1 ? "" : "s"), open at page \(current + 1)."
    let titles = (0..<count).map { "\($0 + 1) \(Self.pageTitle(doc, $0))" }
    line += " Pages: " + titles.joined(separator: "; ") + "."
    var notes: [String] = []
    for index in 0..<count {
      if let summary = Self.annotationSummary(doc, index) { notes.append("page \(index + 1): \(summary)") }
    }
    line += notes.isEmpty ? " Annotations: none." : " Annotations: " + notes.joined(separator: "; ") + "."
    return line
  }

  /// The first line of text on the page, as the page's name in the state —
  /// what a person would call it ("the cover", "the rules page").
  private static func pageTitle(_ doc: PDFDocument, _ index: Int) -> String {
    guard let page = doc.page(at: index) else { return "?" }
    if let text = page.string {
      for raw in text.split(whereSeparator: \.isNewline) {
        let line = raw.trimmingCharacters(in: .whitespaces)
        if line.count >= 3 { return String(line.prefix(28)) }
      }
    }
    return "(blank)"
  }

  private static func annotationSummary(_ doc: PDFDocument, _ index: Int) -> String? {
    guard let page = doc.page(at: index), !page.annotations.isEmpty else { return nil }
    var parts: [String] = []
    let highlights = page.annotations.filter { $0.type == "Highlight" }.count
    let notes = page.annotations.filter { $0.type == "Text" }.count
    let ink = page.annotations.filter { $0.type == "Ink" }.count
    if highlights > 0 { parts.append("\(highlights) highlight\(highlights == 1 ? "" : "s")") }
    if notes > 0 { parts.append("\(notes) note\(notes == 1 ? "" : "s")") }
    if ink > 0 { parts.append("signed") }
    if page.rotation != 0 { parts.append("rotated \(page.rotation)°") }
    return parts.isEmpty ? nil : parts.joined(separator: ", ")
  }

  // MARK: Edits (each returns what the model is told)

  private func withDocument(_ body: (PDFDocument, inout Int) -> String) -> String {
    let result: String = sync {
      guard let doc = document else { return Failure.noDocument.localizedDescription }
      var page = current
      let out = body(doc, &page)
      current = min(max(0, page), max(0, doc.pageCount - 1))
      return out
    }
    refreshThumbnails()
    return result
  }

  private func page(_ number: Int, of doc: PDFDocument) -> Int? {
    let index = number - 1
    return doc.pageCount > 0 && index >= 0 && index < doc.pageCount ? index : nil
  }

  private func noPage(_ number: Int, _ doc: PDFDocument) -> String {
    "there is no page \(number); the document has \(doc.pageCount)"
  }

  func goTo(page number: Int) -> String {
    withDocument { doc, current in
      guard let index = page(number, of: doc) else { return noPage(number, doc) }
      current = index
      return "on page \(number) — \(Self.pageTitle(doc, index))"
    }
  }

  func delete(page number: Int) -> String {
    withDocument { doc, current in
      guard let index = page(number, of: doc) else { return noPage(number, doc) }
      guard doc.pageCount > 1 else { return "cannot delete the only page" }
      let name = Self.pageTitle(doc, index)
      doc.removePage(at: index)
      if current >= index { current = max(0, current - 1) }
      return "deleted page \(number) (\(name)); \(doc.pageCount) pages left"
    }
  }

  func move(page number: Int, to position: Int) -> String {
    withDocument { doc, current in
      guard let from = page(number, of: doc) else { return noPage(number, doc) }
      guard let to = page(position, of: doc) else { return noPage(position, doc) }
      guard from != to, let moving = doc.page(at: from) else { return "page \(number) is already at position \(position)" }
      doc.removePage(at: from)
      doc.insert(moving, at: to)
      current = to
      return "moved \"\(Self.pageTitle(doc, to))\" to position \(position); pages are now " + (0..<doc.pageCount).map { "\($0 + 1) \(Self.pageTitle(doc, $0))" }.joined(separator: "; ")
    }
  }

  func rotate(page number: Int, degrees: Int) -> String {
    withDocument { doc, current in
      guard let index = page(number, of: doc), let p = doc.page(at: index) else { return noPage(number, doc) }
      p.rotation = (p.rotation + degrees) % 360
      current = index
      return "rotated page \(number) by \(degrees)° (now \(p.rotation)°)"
    }
  }

  func insertBlank(after number: Int) -> String {
    withDocument { doc, current in
      let at = min(max(0, number), doc.pageCount)
      let blank = PDFPage()
      if let reference = doc.page(at: max(0, min(number - 1, doc.pageCount - 1))) {
        blank.setBounds(reference.bounds(for: .mediaBox), for: .mediaBox)
      }
      doc.insert(blank, at: at)
      current = at
      return "inserted a blank page as page \(at + 1); \(doc.pageCount) pages now"
    }
  }

  func highlight(_ text: String, color: String) -> String {
    withDocument { doc, current in
      let selections = doc.findString(text, withOptions: [.caseInsensitive])
      guard !selections.isEmpty else { return "\"\(text)\" is not in the document" }
      let tint: UIColor = {
        switch color.lowercased() {
        case "green": return UIColor(red: 0.55, green: 0.95, blue: 0.45, alpha: 1)
        case "pink": return UIColor(red: 1.0, green: 0.6, blue: 0.8, alpha: 1)
        case "blue": return UIColor(red: 0.55, green: 0.8, blue: 1.0, alpha: 1)
        default: return UIColor(red: 1.0, green: 0.92, blue: 0.3, alpha: 1)
        }
      }()
      var pages = Set<Int>()
      var count = 0
      for selection in selections {
        for line in selection.selectionsByLine() {
          for p in line.pages {
            let annotation = PDFAnnotation(bounds: line.bounds(for: p), forType: .highlight, withProperties: nil)
            annotation.color = tint
            p.addAnnotation(annotation)
            pages.insert(doc.index(for: p))
            count += 1
          }
        }
      }
      if let first = pages.min() { current = first }
      let list = pages.sorted().map { String($0 + 1) }.joined(separator: ", ")
      return "highlighted \(count) occurrence\(count == 1 ? "" : "s") of \"\(text)\" in \(color.lowercased()) on page\(pages.count == 1 ? "" : "s") \(list)"
    }
  }

  func removeHighlights(scope: String) -> String {
    withDocument { doc, current in
      let indices = scope.lowercased() == "this_page" ? [current] : Array(0..<doc.pageCount)
      var removed = 0
      for index in indices {
        guard let p = doc.page(at: index) else { continue }
        for annotation in p.annotations where annotation.type == "Highlight" {
          p.removeAnnotation(annotation)
          removed += 1
        }
      }
      return removed == 0
        ? "no highlights to remove \(scope.lowercased() == "this_page" ? "on this page" : "in the document")"
        : "removed \(removed) highlight\(removed == 1 ? "" : "s") \(scope.lowercased() == "this_page" ? "from page \(current + 1)" : "from the whole document")"
    }
  }

  func addNote(_ text: String) -> String {
    withDocument { doc, current in
      guard let p = doc.page(at: current) else { return Failure.noDocument.localizedDescription }
      let box = p.bounds(for: .mediaBox)
      // Top-right corner, stacked down for each note already there.
      let existing = p.annotations.filter { $0.type == "Text" }.count
      let bounds = CGRect(x: box.maxX - 48, y: box.maxY - 48 - CGFloat(existing) * 34, width: 28, height: 28)
      let note = PDFAnnotation(bounds: bounds, forType: .text, withProperties: nil)
      note.contents = text
      note.iconType = .note
      note.color = UIColor(red: 1.0, green: 0.85, blue: 0.2, alpha: 1)
      p.addAnnotation(note)
      return "note added to page \(current + 1): \"\(text)\""
    }
  }

  func sign(page number: Int) -> String {
    withDocument { doc, current in
      guard let index = page(number, of: doc), let p = doc.page(at: index) else { return noPage(number, doc) }
      let box = p.bounds(for: .mediaBox)
      // Bottom-right, above the margin: a hand-drawn-looking signature.
      let origin = CGPoint(x: box.maxX - 220, y: box.minY + 70)
      let path = UIBezierPath()
      path.move(to: CGPoint(x: origin.x, y: origin.y + 10))
      path.addCurve(
        to: CGPoint(x: origin.x + 40, y: origin.y + 30),
        controlPoint1: CGPoint(x: origin.x + 5, y: origin.y + 45),
        controlPoint2: CGPoint(x: origin.x + 30, y: origin.y - 15))
      path.addCurve(
        to: CGPoint(x: origin.x + 95, y: origin.y + 12),
        controlPoint1: CGPoint(x: origin.x + 50, y: origin.y + 60),
        controlPoint2: CGPoint(x: origin.x + 70, y: origin.y - 20))
      path.addCurve(
        to: CGPoint(x: origin.x + 160, y: origin.y + 22),
        controlPoint1: CGPoint(x: origin.x + 115, y: origin.y + 40),
        controlPoint2: CGPoint(x: origin.x + 135, y: origin.y - 5))
      path.addLine(to: CGPoint(x: origin.x + 175, y: origin.y + 8))
      let ink = PDFAnnotation(
        bounds: CGRect(x: origin.x - 10, y: origin.y - 25, width: 200, height: 90), forType: .ink,
        withProperties: nil)
      ink.add(path)
      ink.color = UIColor(red: 0.1, green: 0.2, blue: 0.7, alpha: 1)
      let border = PDFBorder()
      border.lineWidth = 2.2
      ink.border = border
      p.addAnnotation(ink)
      current = index
      return "signed page \(number)"
    }
  }

  /// A big translucent word across every page — the review copy's stamp.
  func watermark(_ text: String) -> String {
    withDocument { doc, current in
      let word = text.trimmingCharacters(in: .whitespaces)
      guard !word.isEmpty else { return "the watermark needs a word" }
      for index in 0..<doc.pageCount {
        guard let p = doc.page(at: index) else { continue }
        let box = p.bounds(for: .mediaBox)
        let height = box.width * 0.16
        let bounds = CGRect(
          x: box.midX - box.width * 0.45, y: box.midY - height / 2,
          width: box.width * 0.9, height: height)
        let mark = PDFAnnotation(bounds: bounds, forType: .freeText, withProperties: nil)
        mark.contents = word
        mark.font = UIFont.boldSystemFont(ofSize: height * 0.6)
        mark.fontColor = UIColor.red.withAlphaComponent(0.18)
        mark.color = .clear
        mark.alignment = .center
        p.addAnnotation(mark)
      }
      return "\"\(word)\" watermarked across all \(doc.pageCount) pages"
    }
  }

  /// A span of pages as a new PDF in Documents; the open document is untouched.
  func extract(from: Int, to: Int) -> String {
    let doc = sync { document }
    guard let doc else { return Failure.noDocument.localizedDescription }
    let low = min(from, to), high = max(from, to)
    guard low >= 1, high <= doc.pageCount else {
      return "pages must be within 1–\(doc.pageCount)"
    }
    let out = PDFDocument()
    for (offset, index) in ((low - 1)..<high).enumerated() {
      if let page = doc.page(at: index), let copy = page.copy() as? PDFPage {
        out.insert(copy, at: offset)
      }
    }
    let url = AppFiles.documents
      .appendingPathComponent("saved-pages-\(low)-\(high).pdf")
    guard out.write(to: url) else { return "could not write \(url.lastPathComponent)" }
    return "extracted pages \(low)–\(high) to \(url.lastPathComponent) in the app's Documents (\(out.pageCount) pages)"
  }

  func search(_ text: String) -> String {
    let doc = sync { document }
    guard let doc else { return Failure.noDocument.localizedDescription }
    let selections = doc.findString(text, withOptions: [.caseInsensitive])
    guard !selections.isEmpty else { return "\"\(text)\" is not in the document" }
    var perPage: [Int: Int] = [:]
    for selection in selections {
      for p in selection.pages { perPage[doc.index(for: p) + 1, default: 0] += 1 }
    }
    let list = perPage.keys.sorted().map { "page \($0) (\(perPage[$0]!)×, \(Self.pageTitle(doc, $0 - 1)))" }
    ArtifactBox.shared.post(.table(
      title: "\"\(text)\" — \(selections.count) match\(selections.count == 1 ? "" : "es")",
      columns: ["Page", "Title", "Matches"],
      rows: perPage.keys.sorted().map { [String($0), Self.pageTitle(doc, $0 - 1), String(perPage[$0]!)] }))
    return "\"\(text)\" appears \(selections.count) time\(selections.count == 1 ? "" : "s"): " + list.joined(separator: "; ")
  }

  func save(as name: String) -> String {
    let (doc, fallback) = sync { (document, title) }
    guard let doc else { return Failure.noDocument.localizedDescription }
    var stem = name.trimmingCharacters(in: .whitespaces)
    if stem.lowercased().hasSuffix(".pdf") { stem = String(stem.dropLast(4)) }
    if stem.isEmpty { stem = fallback }
    let url = AppFiles.documents
      .appendingPathComponent("saved-\(stem).pdf")
    guard doc.write(to: url) else { return "could not write \(url.lastPathComponent)" }
    return "saved as \(url.lastPathComponent) in the app's Documents (\(doc.pageCount) pages)"
  }

  func revert() -> String {
    let (data, title) = sync { (originalData, self.title) }
    guard let data, let doc = PDFDocument(data: data) else { return Failure.noDocument.localizedDescription }
    load(doc, data: data, title: title)
    return "back to the original document — \(doc.pageCount) pages, no annotations"
  }

  // MARK: The stage

  private func refreshThumbnails() {
    let doc = sync { document }
    guard let doc else { return }
    var images: [UIImage] = []
    for index in 0..<doc.pageCount {
      if let page = doc.page(at: index) {
        images.append(page.thumbnail(of: CGSize(width: 90, height: 120), for: .mediaBox))
      }
    }
    sync { thumbnails = images }
  }

  /// The open page, big, with its annotations.
  func currentPageImage() -> UIImage? {
    let (doc, current) = sync { (document, self.current) }
    guard let doc, let page = doc.page(at: current) else { return nil }
    return page.thumbnail(of: CGSize(width: 900, height: 1200), for: .mediaBox)
  }

  func snapshot() -> Snapshot? {
    let (doc, title, current, thumbs) = sync { (document, self.title, self.current, thumbnails) }
    guard let doc else { return nil }
    var annotated: [Int: String] = [:]
    for index in 0..<doc.pageCount {
      if let summary = Self.annotationSummary(doc, index) { annotated[index + 1] = summary }
    }
    return Snapshot(
      title: title, pageCount: doc.pageCount, current: current + 1,
      titles: (0..<doc.pageCount).map { Self.pageTitle(doc, $0) },
      annotated: annotated, thumbnails: thumbs)
  }
}

/// A six-page lease, drawn on the phone: enough headings and paragraphs
/// that "the cover", "the rules page", "every 'deposit'" all mean
/// something. Nothing in it is a real agreement.
enum SampleLease {
  static let pages: [(title: String, body: [String])] = [
    ("Lease Agreement", [
      "Residential Tenancy — Apartment 4B, 12 Sakura Street",
      "Prepared for review. This document has six pages: the parties, rent and deposit, the term, the house rules, and the signature page.",
      "Please read every page before signing.",
    ]),
    ("Parties", [
      "The Landlord: Sakura Property Management, 3-1 Kanda, Tokyo.",
      "The Tenant: the person named on the signature page, together with any occupants listed there.",
      "Both parties agree to the terms set out in the following pages. Notices to either party are to be sent to the addresses above, in writing.",
    ]),
    ("Rent and Deposit", [
      "Monthly rent is ¥128,000, due on the first day of each month by bank transfer.",
      "A security deposit of ¥256,000 (two months' rent) is payable before the keys are handed over. The deposit is held for the duration of the tenancy.",
      "The deposit, less any deductions for damage beyond normal wear, is returned within 30 days of the tenant moving out. Late rent incurs a fee of ¥3,000 per week.",
    ]),
    ("Term", [
      "The tenancy begins on 1 October 2026 and runs for two years, ending on 30 September 2028.",
      "Either party may end the tenancy early with two months' written notice. If the tenant leaves early, the deposit is still returned under the terms on the previous page.",
      "The lease may be renewed for a further term by agreement, at a rent to be reviewed at that time.",
    ]),
    ("House Rules", [
      "Quiet hours are from 22:00 to 07:00. No smoking anywhere in the building. Pets are allowed only with written permission from the landlord.",
      "The tenant keeps the apartment clean and reports any damage or fault promptly. Rubbish is sorted and put out on the days posted in the lobby.",
      "Bicycles are kept in the rack by the entrance, not in the hallway. Guests staying longer than 14 nights must be notified to the landlord.",
    ]),
    ("Signatures", [
      "Signed by the tenant and the landlord on the dates written beside each signature.",
      "Tenant: ______________________________    Date: ______________",
      "Landlord: ____________________________    Date: ______________",
    ]),
  ]

  static func pdfData() -> Data {
    let bounds = CGRect(x: 0, y: 0, width: 595, height: 842)  // A4 in points
    let renderer = UIGraphicsPDFRenderer(bounds: bounds)
    return renderer.pdfData { context in
      let title: [NSAttributedString.Key: Any] = [.font: UIFont.boldSystemFont(ofSize: 26)]
      let body: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 13)]
      let footer: [NSAttributedString.Key: Any] = [.font: UIFont.systemFont(ofSize: 10), .foregroundColor: UIColor.gray]
      for (index, page) in pages.enumerated() {
        context.beginPage()
        page.title.draw(in: CGRect(x: 60, y: 70, width: 475, height: 40), withAttributes: title)
        var y: CGFloat = 130
        for paragraph in page.body {
          let box = CGRect(x: 60, y: y, width: 475, height: 200)
          let height = (paragraph as NSString).boundingRect(
            with: CGSize(width: 475, height: 400), options: [.usesLineFragmentOrigin], attributes: body, context: nil
          ).height
          paragraph.draw(in: box, withAttributes: body)
          y += height + 18
        }
        "Page \(index + 1) of \(pages.count)".draw(at: CGPoint(x: 60, y: 790), withAttributes: footer)
      }
    }
  }
}

// MARK: - Tools (the menu, in its words)

@available(iOS 27.0, *)
struct GoToPageTool: Tool {
  let name = "go_to_page"
  let description = "Open a page of the document."
  @Generable struct Arguments {
    @Guide(description: "Page number, from 1. The page list is in the document state.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.goTo(page: arguments.number)
  }
}

@available(iOS 27.0, *)
struct DeletePageTool: Tool {
  let name = "delete_page"
  let description = "Remove a page from the document."
  @Generable struct Arguments {
    @Guide(description: "Page number, from 1.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.delete(page: arguments.number)
  }
}

@available(iOS 27.0, *)
struct MovePageTool: Tool {
  let name = "move_page"
  let description = "Move a page to another position in the document."
  @Generable struct Arguments {
    @Guide(description: "The page to move, from 1.") var number: Int
    @Guide(description: "Its new position, from 1; the page count is in the state (the last position is the count).") var to: Int
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.move(page: arguments.number, to: arguments.to)
  }
}

@available(iOS 27.0, *)
struct RotatePageTool: Tool {
  let name = "rotate_page"
  let description = "Rotate a page."
  @Generable struct Arguments {
    @Guide(description: "Page number, from 1.") var number: Int
    @Guide(description: "Clockwise degrees.", .anyOf(["90", "180", "270"])) var degrees: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.rotate(page: arguments.number, degrees: Int(arguments.degrees) ?? 90)
  }
}

@available(iOS 27.0, *)
struct InsertBlankPageTool: Tool {
  let name = "insert_blank_page"
  let description = "Insert an empty page after a given page."
  @Generable struct Arguments {
    @Guide(description: "Insert after this page number; 0 puts it first.") var after: Int
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.insertBlank(after: arguments.after)
  }
}

@available(iOS 27.0, *)
struct HighlightTextTool: Tool {
  let name = "highlight_text"
  let description = "Highlight every occurrence of some words in the document."
  @Generable struct Arguments {
    @Guide(description: "The words to highlight, exactly as they appear.") var text: String
    @Guide(description: "Highlighter colour.", .anyOf(["yellow", "green", "pink", "blue"])) var color: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.highlight(arguments.text, color: arguments.color)
  }
}

@available(iOS 27.0, *)
struct RemoveHighlightsTool: Tool {
  let name = "remove_highlights"
  let description = "Remove highlights from the open page or from the whole document."
  @Generable struct Arguments {
    @Guide(description: "How far to reach.", .anyOf(["this_page", "all_pages"])) var scope: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.removeHighlights(scope: arguments.scope)
  }
}

@available(iOS 27.0, *)
struct AddNoteTool: Tool {
  let name = "add_note"
  let description = "Stick a note (a comment) on the open page."
  @Generable struct Arguments {
    @Guide(description: "What the note says, exactly as the user gave it. If the user did not say what the note should say, do not call this — ask them.")
    var text: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.addNote(arguments.text)
  }
}

@available(iOS 27.0, *)
struct SignPageTool: Tool {
  let name = "sign_page"
  let description = "Put the user's signature on a page."
  @Generable struct Arguments {
    @Guide(description: "Page number, from 1. The last page's number is the page count in the state.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.sign(page: arguments.number)
  }
}

@available(iOS 27.0, *)
struct WatermarkTool: Tool {
  let name = "add_watermark"
  let description = "Stamp a big translucent word across every page."
  @Generable struct Arguments {
    @Guide(description: "The word, e.g. DRAFT or CONFIDENTIAL.") var text: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.watermark(arguments.text)
  }
}

@available(iOS 27.0, *)
struct ExtractPagesTool: Tool {
  let name = "extract_pages"
  let description = "Copy a span of pages into a new PDF file. The open document is unchanged."
  @Generable struct Arguments {
    @Guide(description: "First page of the span, from 1.") var from: Int
    @Guide(description: "Last page of the span.") var to: Int
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.extract(from: arguments.from, to: arguments.to)
  }
}

@available(iOS 27.0, *)
struct SearchDocumentTool: Tool {
  let name = "search_document"
  let description = "Find which pages mention some words. Changes nothing."
  @Generable struct Arguments {
    @Guide(description: "The words to look for.") var text: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.search(arguments.text)
  }
}

@available(iOS 27.0, *)
struct SavePDFTool: Tool {
  let name = "save_as"
  let description = "Save the document under a new name."
  @Generable struct Arguments {
    @Guide(description: "The file name, without .pdf.") var name: String
  }
  func call(arguments: Arguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.save(as: arguments.name)
  }
}

@available(iOS 27.0, *)
struct RevertDocumentTool: Tool {
  let name = "revert_to_original"
  let description = "Throw away all changes and go back to the document as it was opened."
  func call(arguments: NoArguments) async throws -> String {
    DocBox.shared.preload()
    return DocBox.shared.revert()
  }
}

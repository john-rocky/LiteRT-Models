// The photo-library pack: a camera roll, searched and operated in plain words.
//
// The orchestrator thesis's first pack (ROADMAP): the OS already ships the
// judges — metadata, an embedding index, the classical detectors, and a
// per-item look — and the model's job is the podium, routing each clause to
// the cheapest layer that can answer it. So the layers are the tool names,
// and which one the model reaches for first is the thing this pack measures:
//
//   find_photos            when / where / which album      — metadata, free
//   search_photos          what is in the picture          — the index
//   find_photos_of_person  who is in it                    — the face layer
//   find_photos_with_text  words written in the picture     — OCR
//   find_blurry_photos     sharpness                        — a detector
//   find_duplicates        near-identical pairs             — a detector
//   check_photo            one photo, one forced choice     — the long tail
//
// Distinct from the photo / vision / polish packs, which edit one photo. This
// one never touches pixels: it finds photos across a library and operates on
// what it found (albums, favourites, deletion) — the retrieval archetype with
// the cost gradient made explicit.
//
// The library is canned (LibraryData below), the same way the store's
// products and the CRM's pipeline are: one frozen world both the stage and
// the bench render, so "it went to the metadata layer" is a checkable claim.
// What is *not* here yet is the perception rung — VNClassify / VNDetectFace /
// VNRecognizeText / a CoreImage sharpness meter over real pixels, which is
// what makes the ROADMAP's "real Vision calls" true. That rung replaces
// `looks` / `people` / `text` / `sharp` with what the OS says about fixture
// images; the tool boundary and every case above it stay exactly as they are.
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class PhotoLibraryBox: @unchecked Sendable {
  static let shared = PhotoLibraryBox()

  struct Photo: Sendable {
    let id: Int
    /// Frozen dates, YYYY-MM-DD: string order is date order, and a frozen
    /// today in the state makes "last summer" a scorable window (the CRM
    /// pack's rule).
    let date: String
    let place: String
    var album: String?
    var favorite: Bool = false
    /// What the picture shows — where VNClassify's scene nouns and the
    /// specialized detectors' objects will land when the rung goes in.
    let looks: [String]
    /// Who is in it. Detection is the OS's job; the *names* are the library's
    /// own data — Vision finds a face, it never knows it is Mei.
    var people: [String] = []
    /// Words written in the picture (OCR's half).
    var text: String? = nil
    var sharp: Bool = true
    /// The other half of a near-identical pair.
    var twin: Int? = nil
    var deleted: Bool = false
  }

  private let lock = NSLock()
  private var photos: [Photo] = LibraryData.photos
  private var selection: [Int] = []
  private var selectionHow = ""
  /// The confirm gate's other half: a false delete leaves this behind so the
  /// state can say what is waiting, and the yes-turn has something to mean.
  private var pendingDelete: [Int] = []
  private let history = UndoStack<([Photo], [Int], String, [Int])>()

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  private func pushHistory(_ what: String) {
    history.push((photos, selection, selectionHow, pendingDelete), what)
  }

  func undoLast() -> String {
    guard let (snap, what) = history.pop() else { return "nothing to undo" }
    sync { (photos, selection, selectionHow, pendingDelete) = snap }
    post()
    return "undid the last change (\(what))"
  }

  // MARK: The state the model reads

  func describe() -> String {
    let (rows, selected, how, pending) = sync {
      (photos.filter { !$0.deleted }, selection, selectionHow, pendingDelete)
    }
    let dates = rows.map(\.date).sorted()
    var albums: [String: Int] = [:]
    for row in rows { if let album = row.album { albums[album, default: 0] += 1 } }
    var line = "Library: \(rows.count) photos"
    if let first = dates.first, let last = dates.last { line += ", \(first) to \(last)" }
    line += "."
    if !albums.isEmpty {
      line += " Albums: "
        + albums.sorted { $0.key < $1.key }.map { "\($0.key) (\($0.value))" }
          .joined(separator: ", ") + "."
    }
    // The state must carry the words a person points with, and a filter's
    // vocabulary is one of them (r1: with the albums named and the places
    // not, "in Kyoto" became album "Kyoto trip" — the state's own vocabulary
    // is what the model fills a slot from — and "the beach photos" became
    // place "beach", then "sea", "coast", "shore", five calls into a false
    // "you have no beach photos from last summer").
    let places = Set(rows.map(\.place)).sorted()
    if !places.isEmpty { line += " Places: " + places.joined(separator: ", ") + "." }
    line += " Today: \(LibraryData.today) (\(LibraryData.todayWeekday))."
    let live = selected.filter { id in rows.contains { $0.id == id } }
    if live.isEmpty {
      line += " Selection: none."
    } else {
      let shown = rows.filter { live.contains($0.id) }
      let names = shown.prefix(5).map { LibraryData.line($0) }
      let more = shown.count > 5 ? "; and \(shown.count - 5) more" : ""
      line += " Selection: \(shown.count) from \(how): " + names.joined(separator: "; ") + more + "."
    }
    if !pending.isEmpty {
      line += " Awaiting confirmation: delete \(pending.count) photo\(pending.count == 1 ? "" : "s")."
    }
    if let what = history.peekWhat() { line += " Last change: \(what)." }
    return line
  }

  // MARK: The panel

  func snapshot() -> TablePanel {
    let (rows, selected, how) = sync {
      (photos.filter { !$0.deleted }, selection, selectionHow)
    }
    let shown = selected.isEmpty ? rows : rows.filter { selected.contains($0.id) }
    return TablePanel(
      title: selected.isEmpty ? "All photos" : "\(shown.count) selected — \(how)",
      columns: ["#", "When", "Where", "What", "Album"],
      rows: shown.map {
        [
          "\($0.id)", $0.date, $0.place,
          $0.looks.prefix(3).joined(separator: ", ") + ($0.sharp ? "" : " (blurry)"),
          $0.album ?? "—",
        ]
      },
      totalRows: shown.count,
      overview: "\(rows.count) photos")
  }

  private func post() {
    let panel = snapshot()
    ArtifactBox.shared.post(
      .table(
        title: panel.title, columns: panel.columns, rows: Array(panel.rows.prefix(8))))
  }

  // MARK: Finding (each replaces the selection, or narrows it)

  private func pool(_ refine: Bool?) -> [Photo] {
    let (rows, selected) = sync { (photos.filter { !$0.deleted }, selection) }
    guard refine == true, !selected.isEmpty else { return rows }
    return rows.filter { selected.contains($0.id) }
  }

  /// The matching itself lives in LibraryData, as pure functions over a pool
  /// of rows: the bench's echo renders the same world and must not be able to
  /// drift from the app (the moment-seek pack keeps two parallel matchers on
  /// purpose — they are its measured control — but nothing here is measured
  /// yet, so one implementation is the cheaper honesty).
  private func answer(_ result: LibraryData.Answer) -> String {
    switch result {
    // A refusal changes nothing: the selection the user had is still theirs.
    case .refusal(let text): return text
    case .rows(let matched, let how): return select(matched, how: how)
    }
  }

  private func select(_ matched: [Photo], how: String) -> String {
    sync {
      selection = matched.map(\.id)
      selectionHow = how
      pendingDelete = []
    }
    post()
    return LibraryData.found(matched, how: how)
  }

  func findPhotos(when: String?, place: String?, album: String?, favorites: Bool?, refine: Bool?)
    -> String
  {
    answer(
      LibraryData.findPhotos(
        pool(refine), when: when, place: place, album: album, favorites: favorites))
  }

  func searchPhotos(query: String, refine: Bool?) -> String {
    answer(LibraryData.search(pool(refine), query: query))
  }

  func findPeople(name: String, refine: Bool?) -> String {
    answer(LibraryData.people(pool(refine), name: name))
  }

  func findText(text: String, refine: Bool?) -> String {
    answer(LibraryData.withText(pool(refine), text: text))
  }

  func findBlurry(refine: Bool?) -> String {
    answer(LibraryData.blurry(pool(refine)))
  }

  func findDuplicates() -> String {
    answer(LibraryData.duplicates(pool(false)))
  }

  // MARK: One photo

  func open(id: Int) -> String {
    guard let photo = sync({ photos.first { $0.id == id && !$0.deleted } }) else {
      return "there is no photo #\(id)"
    }
    ArtifactBox.shared.post(.note(text: LibraryData.line(photo)))
    return "photo #\(id) is on screen — " + LibraryData.line(photo)
  }

  func check(id: Int, question: String, options: [String]) -> String {
    guard let photo = sync({ photos.first { $0.id == id && !$0.deleted } }) else {
      return "there is no photo #\(id)"
    }
    return ForcedChoice.answer(
      question: question, options: options, truths: LibraryData.truths(photo),
      shows: " — photo #\(id) shows: ", aliases: LibraryData.aliases)
  }

  // MARK: Acting on the selection

  private func selected() -> [Photo] {
    let (rows, ids) = sync { (photos, selection) }
    return rows.filter { ids.contains($0.id) && !$0.deleted }
  }

  func addToAlbum(_ album: String) -> String {
    let rows = selected()
    guard !rows.isEmpty else { return "no photos are selected — find some first" }
    let existing = Set(sync { photos }.compactMap(\.album))
    let name =
      existing.first { $0.lowercased() == album.lowercased() }
      ?? existing.first { $0.lowercased().contains(album.lowercased()) } ?? album
    let isNew = !existing.contains(name)
    sync {
      pushHistory("album \(name)")
      for index in photos.indices where rows.contains(where: { $0.id == photos[index].id }) {
        photos[index].album = name
      }
    }
    post()
    return "put \(rows.count) photo\(rows.count == 1 ? "" : "s") in \(name)\(isNew ? " (a new album)" : "")"
  }

  func favorite() -> String {
    let rows = selected()
    guard !rows.isEmpty else { return "no photos are selected — find some first" }
    sync {
      pushHistory("favourites")
      for index in photos.indices where rows.contains(where: { $0.id == photos[index].id }) {
        photos[index].favorite = true
      }
    }
    post()
    return "marked \(rows.count) photo\(rows.count == 1 ? "" : "s") as favourite"
  }

  func delete(confirmed: Bool) -> String {
    let rows = selected()
    guard !rows.isEmpty else { return "no photos are selected — find some first" }
    // Defence in depth behind the finder's own refusal (see LibraryData.
    // findPhotos): whatever put every photo in the selection, deleting all of
    // them is not a request this tool can have understood correctly.
    let live = sync { photos.filter { !$0.deleted }.count }
    if rows.count == live, live > 1 {
      return
        "that is every photo in the library (\(live)) — this tool does not empty a library;"
        + " narrow the selection to the photos that should go"
    }
    let numbers = rows.map { "#\($0.id)" }.joined(separator: ", ")
    // The gate is the app's, not the argument's: measured across two packs,
    // a confirm argument collapses the moment the user's own words are the
    // tool's verb (recipes: a confirm argument holds until…). What holds is
    // this branch — the app answers a false call with what would happen and
    // changes nothing.
    guard confirmed else {
      sync { pendingDelete = rows.map(\.id) }
      return "not deleted yet — this would delete \(rows.count) photo\(rows.count == 1 ? "" : "s") (\(numbers)); say yes to confirm"
    }
    sync {
      pushHistory("deleting \(rows.count) photos")
      for index in photos.indices where rows.contains(where: { $0.id == photos[index].id }) {
        photos[index].deleted = true
      }
      selection = []
      selectionHow = ""
      pendingDelete = []
    }
    post()
    return "deleted \(rows.count) photo\(rows.count == 1 ? "" : "s") (\(numbers))"
  }
}

/// One person's camera roll, frozen: fourteen months, five places, three
/// albums, three people the library knows by name. Built so every layer has
/// something only it can answer — a receipt whose *picture* is just paper on a
/// table (only OCR finds it), a puppy on a beach the index finds and no
/// metadata can, three out-of-focus frames, two near-identical pairs — and so
/// the composition case is real: five of the seven beach photos are from last
/// summer, so "the beach photos from last summer" is not either clause alone.
///
/// No cat anywhere, on purpose: the retrieval archetype's core case is the
/// honest empty answer, and it needs a query the world cannot satisfy.
@available(iOS 27.0, *)
enum LibraryData {
  static let today = "2026-08-21"
  static let todayWeekday = "Friday"

  static let photos: [PhotoLibraryBox.Photo] = [
    .init(id: 1, date: "2025-06-14", place: "Kamakura", looks: ["beach", "sea", "sand", "sunset"]),
    .init(
      id: 2, date: "2025-07-05", place: "Kamakura", looks: ["beach", "sea", "surfboard"],
      people: ["Ken"]),
    .init(
      id: 3, date: "2025-07-05", place: "Kamakura", looks: ["beach", "sea", "surfboard"],
      people: ["Ken"], twin: 2),
    .init(
      id: 4, date: "2025-08-03", place: "Kamakura", favorite: true,
      looks: ["beach", "sand", "dog"]),
    .init(
      id: 5, date: "2025-08-03", place: "Kamakura", looks: ["beach", "sand", "dog"], sharp: false),
    .init(id: 6, date: "2025-08-17", place: "Sapporo", looks: ["mountain", "forest", "path"]),
    .init(
      id: 7, date: "2025-10-12", place: "Kyoto", album: "Kyoto trip", favorite: true,
      looks: ["temple", "autumn leaves", "maple"]),
    .init(
      id: 8, date: "2025-10-12", place: "Kyoto", album: "Kyoto trip",
      looks: ["temple", "garden"], people: ["Mei"]),
    .init(
      id: 9, date: "2025-10-13", place: "Kyoto", album: "Kyoto trip",
      looks: ["street", "lantern", "night"], sharp: false),
    .init(
      id: 10, date: "2025-10-13", place: "Kyoto", album: "Kyoto trip",
      looks: ["food", "ramen", "noodles"], text: "RAMEN 950"),
    .init(
      id: 11, date: "2025-10-13", place: "Kyoto", album: "Kyoto trip",
      looks: ["paper", "table", "indoor"], text: "RECEIPT TOTAL 3,480"),
    .init(
      id: 12, date: "2025-10-14", place: "Kyoto", album: "Kyoto trip",
      looks: ["station", "train"]),
    .init(
      id: 13, date: "2025-12-24", place: "Tokyo", album: "Family", favorite: true,
      looks: ["cake", "table", "indoor"], people: ["Mei", "Aoi"]),
    .init(
      id: 14, date: "2025-12-25", place: "Tokyo", album: "Family",
      looks: ["indoor", "tree", "lights"], people: ["Aoi"]),
    .init(
      id: 15, date: "2026-01-01", place: "Tokyo", album: "Family",
      looks: ["shrine", "crowd"], people: ["Mei", "Ken", "Aoi"]),
    .init(id: 16, date: "2026-01-14", place: "Sapporo", looks: ["snow", "street"]),
    .init(
      id: 17, date: "2026-02-08", place: "Tokyo", album: "Work",
      looks: ["whiteboard", "indoor"], text: "Q1 ROADMAP — ship by Mar 31"),
    .init(
      id: 18, date: "2026-02-08", place: "Tokyo", album: "Work",
      looks: ["whiteboard", "indoor"], text: "Q1 ROADMAP — ship by Mar 31", twin: 17),
    .init(
      id: 19, date: "2026-03-02", place: "Tokyo", looks: ["screen", "phone"],
      text: "BOARDING PASS NRT 09:40"),
    .init(
      id: 20, date: "2026-04-05", place: "Osaka", favorite: true,
      looks: ["cherry blossom", "river", "park"], people: ["Mei"]),
    .init(
      id: 21, date: "2026-04-05", place: "Osaka", looks: ["cherry blossom", "park"], sharp: false),
    .init(
      id: 22, date: "2026-05-03", place: "Tokyo", album: "Work",
      looks: ["whiteboard", "sticky notes"], text: "SPRINT 12 backlog"),
    .init(id: 23, date: "2026-06-20", place: "Kamakura", looks: ["beach", "sea", "sunset"]),
    .init(
      id: 24, date: "2026-07-11", place: "Tokyo", looks: ["food", "sushi", "plate"],
      text: "OMAKASE 4,200"),
    .init(
      id: 25, date: "2026-07-28", place: "Tokyo", album: "Family",
      looks: ["dog", "park", "grass"], people: ["Aoi"]),
    .init(
      id: 26, date: "2026-08-09", place: "Osaka", looks: ["street", "sign", "night"],
      text: "道頓堀"),
    .init(
      id: 27, date: "2026-08-15", place: "Kamakura", favorite: true,
      looks: ["beach", "sea", "fireworks", "night"]),
    .init(
      id: 28, date: "2026-08-18", place: "Tokyo", looks: ["paper", "table", "indoor"],
      text: "RECEIPT TOTAL 1,120"),
  ]

  /// The findability rule, made structural: the model does not translate its
  /// query (measured twice — 29 of 37 search calls on JA input carried JA
  /// strings), the labels are English, and a real index's labels always will
  /// be. So the alias table is a load-bearing part, not a canned-world
  /// convenience — it is the only reason 「犬」 finds a row that says "dog",
  /// and it is the first thing the Vision rung will inherit unchanged.
  static let aliases: [(String, String)] = [
    ("海", "sea"), ("ビーチ", "beach"), ("砂浜", "beach"), ("波", "sea"),
    ("犬", "dog"), ("いぬ", "dog"), ("猫", "cat"), ("ねこ", "cat"),
    ("花火", "fireworks"), ("桜", "cherry blossom"), ("紅葉", "autumn leaves"),
    ("寺", "temple"), ("神社", "shrine"), ("雪", "snow"), ("山", "mountain"),
    ("料理", "food"), ("食べ物", "food"), ("ごはん", "food"), ("ラーメン", "ramen"),
    ("寿司", "sushi"), ("ケーキ", "cake"), ("夕日", "sunset"), ("夕焼け", "sunset"),
    ("街", "street"), ("看板", "sign"), ("電車", "train"), ("駅", "station"),
    ("公園", "park"), ("芝生", "grass"), ("砂", "sand"), ("森", "forest"),
    ("ホワイトボード", "whiteboard"), ("レシート", "receipt"), ("領収書", "receipt"),
    ("メイ", "mei"), ("ケン", "ken"), ("アオイ", "aoi"), ("葵", "aoi"),
    ("京都", "kyoto"), ("鎌倉", "kamakura"), ("東京", "tokyo"), ("大阪", "osaka"),
    ("札幌", "sapporo"),
  ]

  static func line(_ photo: PhotoLibraryBox.Photo) -> String {
    var line = "#\(photo.id) \(photo.date) \(photo.place)"
    if let album = photo.album { line += ", \(album)" }
    line += " — " + photo.looks.joined(separator: ", ")
    if !photo.people.isEmpty { line += "; with " + photo.people.joined(separator: ", ") }
    if let text = photo.text { line += "; text \"\(text)\"" }
    if !photo.sharp { line += "; out of focus" }
    if photo.favorite { line += "; favourite" }
    return line
  }

  /// Everything one photo could honestly be asked about — the check's world.
  static func truths(_ photo: PhotoLibraryBox.Photo) -> [String] {
    var truths = photo.looks
    truths += photo.people.map { $0.lowercased() }
    truths.append(photo.place.lowercased())
    if let album = photo.album { truths.append(album.lowercased()) }
    truths.append(photo.date)
    if let text = photo.text { truths.append(text.lowercased()) }
    truths.append(photo.sharp ? "sharp" : "out of focus")
    if !photo.sharp { truths.append("blurry") }
    if photo.favorite { truths.append("favourite") }
    return truths
  }

  /// What a finder found, or why it could not look. A refusal is not an empty
  /// result: the difference is the whole D2 ruling, and the model reads the
  /// two as opposite facts about the user's library.
  enum Answer: Sendable {
    case rows([PhotoLibraryBox.Photo], String)
    case refusal(String)
  }

  /// One row shape for every finder, "found" leading. The verdict word is the
  /// first thing the model reads back and the strongest one wins its final
  /// answer, so a hit has to say it found something (moment-seek: a bare count
  /// lost to two later empty sweeps).
  static func found(_ matched: [PhotoLibraryBox.Photo], how: String) -> String {
    guard !matched.isEmpty else { return "no photos match \(how)" }
    let listed = matched.prefix(8).map { line($0) }
    return "found \(matched.count) photo\(matched.count == 1 ? "" : "s") (\(how)):\n"
      + listed.joined(separator: "\n")
      + (matched.count > 8 ? "\n… and \(matched.count - 8) more" : "")
  }

  static func findPhotos(
    _ pool: [PhotoLibraryBox.Photo], when: String?, place: String?, album: String?,
    favorites: Bool?
  ) -> Answer {
    var rows = pool
    var how: [String] = []
    if let when, !when.trimmingCharacters(in: .whitespaces).isEmpty {
      // A finder that cannot read its own argument must not report absence —
      // the check's ruling (playbook D2), which is a rule about verdicts and
      // not about checks. An unparsed phrase says so and names what it can
      // read; an empty answer would be read as "you have no photos then".
      guard let window = self.window(when) else {
        return .refusal(
          "cannot tell which dates \"\(when)\" means — say a day, a month, a season or a year (today is \(today))"
        )
      }
      rows = rows.filter { $0.date >= window.from && $0.date <= window.to }
      how.append(window.label)
    }
    // A filter over a closed vocabulary answers about the vocabulary, never
    // with absence. This is the check's cannot-tell ruling in the finders'
    // clothes, and r1 measured both halves of it in one round: asked for
    // "the beach photos from last summer" the model put "beach" in the place
    // slot, the app answered "no photos match summer 2025, in beach", and the
    // model told the user their library has no such photos — a false absence
    // manufactured by a slot mistake. The one place r1 recovered gracefully
    // was the person tool, whose unknown-name branch names the roster: the
    // model read it, apologized and asked which person was meant. So every
    // vocabulary filter gets that branch.
    if let place, !place.isEmpty {
      let words = tokens(place)
      let roster = Set(photos.map(\.place)).sorted()
      let known = roster.filter { name in words.contains { name.lowercased().contains($0) } }
      guard !known.isEmpty else {
        // Name the recovery, not the vocabulary: l2 measured what listing the
        // legal values costs (below, in the album branch).
        return .refusal(
          "\"\(place)\" is not one of the places in the state — for what a photo shows "
            + "rather than where it was taken, search the picture instead")
      }
      rows = rows.filter { known.contains($0.place) }
      how.append("in \(known.joined(separator: ", "))")
    }
    if let album, !album.isEmpty {
      let words = tokens(album)
      let roster = Set(photos.compactMap(\.album)).sorted()
      let known = roster.filter { name in words.contains { name.lowercased().contains($0) } }
      // A roster in a result is a work list. l2 named the three albums here
      // and one JA chain case walked all three against four phrasings of the
      // same date — ten calls — before filing the photos in the wrong one:
      // the invent-fodder recipe (an example decides *what* gets invented)
      // holds for tool results, not only for argument guides. The state
      // already lists the albums; what a refusal owes the model is the way
      // out, which here is the tool that makes an album rather than the tool
      // that filters by one.
      guard !known.isEmpty else {
        return .refusal(
          "no album is called \"\(album)\" — add_to_album is what puts photos in a new album; "
            + "this argument only filters by an album that already exists")
      }
      rows = rows.filter { row in row.album.map { known.contains($0) } ?? false }
      how.append("album \"\(known.joined(separator: ", "))\"")
    }
    if favorites == true {
      rows = rows.filter(\.favorite)
      how.append("favourites")
    }
    // **An argument-less finder is not a selection.** This branch used to
    // answer "the whole library" and select all 28 photos, and the stage
    // measured what that arms: told 「Yes, delete them.」 with nothing
    // selected, the model called delete_photos (refused — nothing selected),
    // then `find_photos {}`, then delete_photos again, **and deleted all 27
    // photos**. Every step is locally reasonable; the chain is a wiped
    // library. A destructive tool downstream of a finder means the finder's
    // widest answer is a loaded gun, and "show me everything" is a question
    // about the library, not a selection of it — so it is answered from the
    // shape of the library and the selection is left alone.
    guard !how.isEmpty else {
      return .refusal(
        "\(rows.count) photos in all, \(rows.map(\.date).min() ?? "") to \(rows.map(\.date).max() ?? "")"
          + " — find_photos narrows by when, place, album or favourites; without one of those there"
          + " is nothing to select")
    }
    return .rows(rows, how.joined(separator: ", "))
  }

  static func search(_ pool: [PhotoLibraryBox.Photo], query: String) -> Answer {
    let words = tokens(query)
    guard evaluable(words, against: pool.map { $0.looks.joined(separator: " ") }) else {
      return .refusal(blind(query, "the picture"))
    }
    let matched = pool.filter { row in
      words.contains { word in row.looks.contains { $0.contains(word) } }
    }
    if matched.isEmpty, let elsewhere = otherRung(words, than: .picture) {
      return .refusal(elsewhere)
    }
    return .rows(matched, "\"\(query)\" in the picture")
  }

  /// Which rung a query does not belong on.
  enum Rung { case picture, written }

  /// The cost gradient is a fallback chain, not only a routing choice — and
  /// the app is the half that knows all of it. A layer that finds nothing
  /// while a *different* layer holds the very word asked about is not looking
  /// at an absence; it is looking at a misroute, and answering "no photos"
  /// turns the model's slot mistake into a fact about the user's library.
  /// Measured (l2b): 「メイが写ってる写真を探して。」 went to search_photos,
  /// which answered nothing, and the model reported "the library holds no
  /// photos featuring メイ" about a person in four of them; 「レシート」 does
  /// the same in every round, because in Japanese the object and the word
  /// printed on it are one word and the receipts' *picture* is paper on a
  /// table. So a silent rung names the rung that can answer — one rung, with
  /// the matching value, never a roster (see the album branch above for what
  /// a roster in a result costs).
  static func otherRung(_ words: [String], than rung: Rung) -> String? {
    func hit(_ haystack: [String]) -> String? {
      words.first { word in haystack.contains { $0.contains(word) } }
    }
    if let name = hit(Set(photos.flatMap(\.people)).map { $0.lowercased() }) {
      return
        "nothing in the picture matches \"\(name)\" — but that is one of the people this library "
        + "knows by name, and find_photos_of_person finds the photos they are in"
    }
    if rung != .written, let word = hit(photos.compactMap { $0.text?.lowercased() }) {
      return
        "nothing in the picture matches \"\(word)\" — but those words are written inside some "
        + "photos, and find_photos_with_text finds them"
    }
    if rung != .picture, let word = hit(photos.map { $0.looks.joined(separator: " ") }) {
      return
        "no photo has \"\(word)\" written in it — but photos show it, and search_photos finds "
        + "those"
    }
    if let place = hit(Set(photos.map(\.place)).map { $0.lowercased() }) {
      return
        "nothing in the picture matches \"\(place)\" — but it is one of the places in the state, "
        + "and find_photos takes it as `place`"
    }
    return nil
  }

  static func people(_ pool: [PhotoLibraryBox.Photo], name: String) -> Answer {
    let words = tokens(name)
    let roster = Set(photos.flatMap(\.people)).sorted()
    let known = roster.filter { person in words.contains { person.lowercased().contains($0) } }
    // Nobody of that name is not "no photos": it is a question about the
    // roster, and answering it with an absence sends the model off to tell
    // the user they have no photos of their own daughter.
    guard !known.isEmpty else {
      return .refusal(
        "nobody called \"\(name)\" is named in this library — the people it knows are "
          + roster.joined(separator: ", "))
    }
    let matched = pool.filter { row in row.people.contains { known.contains($0) } }
    return .rows(matched, "with \(known.joined(separator: ", "))")
  }

  static func withText(_ pool: [PhotoLibraryBox.Photo], text: String) -> Answer {
    let words = tokens(text)
    guard evaluable(words, against: pool.compactMap { $0.text?.lowercased() }) else {
      return .refusal(blind(text, "the words written in a picture"))
    }
    let matched = pool.filter { row in
      guard let written = row.text?.lowercased() else { return false }
      return words.contains { written.contains($0) }
    }
    if matched.isEmpty, let elsewhere = otherRung(words, than: .written) {
      return .refusal(elsewhere)
    }
    return .rows(matched, "\"\(text)\" written in the picture")
  }

  static func blurry(_ pool: [PhotoLibraryBox.Photo]) -> Answer {
    .rows(pool.filter { !$0.sharp }, "out of focus")
  }

  static func duplicates(_ pool: [PhotoLibraryBox.Photo]) -> Answer {
    let paired = pool.filter { $0.twin != nil }
    let ids = Set(paired.map(\.id)).union(paired.compactMap(\.twin))
    return .rows(pool.filter { ids.contains($0.id) }, "near-identical pairs")
  }

  /// Query words, the search tokenizer's rule (three letters, or anything
  /// carrying a digit — "to" and "the" match everything, "1-0" must match),
  /// plus whatever the alias table can add.
  static func tokens(_ query: String) -> [String] {
    var tokens = query.lowercased()
      .split(whereSeparator: { " ,.!?'\"「」『』、。".contains($0) })
      .map(String.init)
      .filter { $0.count >= 3 || $0.contains(where: \.isNumber) }
    for (from, to) in aliases where query.contains(from) { tokens.append(to) }
    return tokens
  }

  /// The check's cannot-tell branch, in the finders. A JA clause arrives as
  /// one whole token that no English label can contain, and a finder that
  /// cannot see the word must not answer "no photos" — that reads as absence,
  /// and absence is what the model reports to the user. A query is evaluable
  /// when some token could appear in what is being searched: it carries ASCII,
  /// or it actually occurs in the haystack (which is where a JA sign like
  /// 道頓堀 is answerable and honest).
  static func evaluable(_ tokens: [String], against haystack: [String]) -> Bool {
    tokens.contains { token in
      token.contains { $0.isASCII && ($0.isLetter || $0.isNumber) }
        || haystack.contains { $0.contains(token) }
    }
  }

  static func blind(_ query: String, _ what: String) -> String {
    "cannot tell what \"\(query)\" would look like in \(what) — nothing in the library is described with that word"
  }

  /// The date phrase, in the user's own words, resolved by the app.
  ///
  /// Units the user never says are arithmetic the model must do, and it does
  /// it badly (measured: weekday→date wrong across runs and languages, in both
  /// directions). So the argument takes "last summer" / 「去年の夏」 / "October
  /// 2025" / "2025-10-13" and the calendar happens here, against a frozen
  /// today. Seasons are meteorological, winter runs Dec–Feb into the next
  /// year, and a week is the seven days ending today — say what a window means
  /// rather than guessing what the user meant.
  static func window(_ phrase: String) -> (from: String, to: String, label: String)? {
    let q = phrase.lowercased()
    func has(_ words: [String]) -> Bool { words.contains { q.contains($0) } }
    func day(_ offset: Int) -> String {
      let formatter = DateFormatter()
      formatter.dateFormat = "yyyy-MM-dd"
      formatter.timeZone = TimeZone(identifier: "UTC")
      guard let base = formatter.date(from: today),
        let moved = Calendar(identifier: .gregorian).date(byAdding: .day, value: offset, to: base)
      else { return today }
      return formatter.string(from: moved)
    }
    func month(_ year: Int, _ month: Int) -> (String, String) {
      let lengths = [31, year % 4 == 0 ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
      let index = min(max(month, 1), 12)
      return (
        String(format: "%04d-%02d-01", year, index),
        String(format: "%04d-%02d-%02d", year, index, lengths[index - 1])
      )
    }

    if has(["today", "今日", "きょう"]) { return (today, today, "today") }
    if has(["yesterday", "昨日", "きのう"]) { return (day(-1), day(-1), "yesterday") }
    // Calendar weeks, Monday first — "last week" means the week that ended,
    // not the seven days ending a week ago.
    let sinceMonday: Int = {
      let formatter = DateFormatter()
      formatter.dateFormat = "yyyy-MM-dd"
      formatter.timeZone = TimeZone(identifier: "UTC")
      var calendar = Calendar(identifier: .gregorian)
      calendar.timeZone = TimeZone(identifier: "UTC")!
      guard let base = formatter.date(from: today) else { return 0 }
      return (calendar.component(.weekday, from: base) + 5) % 7
    }()
    if has(["last week", "先週"]) {
      return (day(-sinceMonday - 7), day(-sinceMonday - 1), "last week")
    }
    if has(["this week", "今週"]) { return (day(-sinceMonday), today, "this week") }
    if has(["last month", "先月"]) { return (month(2026, 7).0, month(2026, 7).1, "last month") }
    if has(["this month", "今月"]) { return (month(2026, 8).0, month(2026, 8).1, "this month") }

    // An explicit date or month beats every relative word.
    let digits = q.replacingOccurrences(of: "/", with: "-")
    if let range = digits.range(of: #"20\d\d-\d\d-\d\d"#, options: .regularExpression) {
      let date = String(digits[range])
      return (date, date, date)
    }
    if let range = digits.range(of: #"20\d\d-\d{1,2}"#, options: .regularExpression) {
      let parts = digits[range].split(separator: "-").map { Int($0) ?? 0 }
      let span = month(parts[0], parts[1])
      return (span.0, span.1, String(digits[range]))
    }

    var year = 2026
    var explicit = false
    if let range = q.range(of: #"20\d\d"#, options: .regularExpression) {
      year = Int(q[range]) ?? year
      explicit = true
    } else if has(["last year", "去年", "昨年"]) {
      year = 2025
      explicit = true
    } else if has(["this year", "今年"]) {
      explicit = true
    }
    let backOne = has(["last ", "去年", "昨年", "前の"])

    let months = [
      "january", "february", "march", "april", "may", "june", "july", "august",
      "september", "october", "november", "december",
    ]
    for (index, name) in months.enumerated()
    where q.contains(name) || q.contains(name.prefix(3)) || q.contains("\(index + 1)月") {
      let span = month(year, index + 1)
      return (span.0, span.1, "\(name.capitalized) \(year)")
    }

    // Seasons: named without a year, "last" walks back one, and a season
    // still to come this year means the one just gone.
    func season(_ from: Int, _ to: Int, _ name: String) -> (String, String, String) {
      var start = year
      if !explicit && backOne { start -= 1 }
      let head = month(start, from).0
      let tail = to < from ? month(start + 1, to).1 : month(start, to).1
      return (head, tail, "\(name) \(start)")
    }
    if has(["summer", "夏"]) { return season(6, 8, "summer") }
    if has(["autumn", "fall", "秋"]) { return season(9, 11, "autumn") }
    if has(["winter", "冬"]) { return season(12, 2, "winter") }
    if has(["spring", "春"]) { return season(3, 5, "spring") }

    if explicit { return ("\(year)-01-01", "\(year)-12-31", "\(year)") }
    return nil
  }
}

// MARK: - Tools (the layers, in the words a person says)

@available(iOS 27.0, *)
struct FindPhotosTool: Tool {
  let name = "find_photos"
  let description =
    "Find photos by when they were taken, where, which album they are in, or whether they are favourites. Not for what is in the picture. The matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "When, in the user's own words — \"last summer\", \"yesterday\", \"October 2025\", \"2025-10-13\".")
    var when: String?
    @Guide(description: "Where — a place name from the state or the request.") var place: String?
    @Guide(description: "One album's name.") var album: String?
    @Guide(description: "True for favourites only.") var favorites_only: Bool?
    @Guide(description: "True to narrow the photos already selected instead of searching the whole library.")
    var refine: Bool?
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.findPhotos(
      when: arguments.when, place: arguments.place, album: arguments.album,
      favorites: arguments.favorites_only, refine: arguments.refine)
  }
}

@available(iOS 27.0, *)
struct SearchPhotosTool: Tool {
  let name = "search_photos"
  let description =
    "Find photos by what is in the picture — the subject, the scene, what it looks like. Not for words written in the picture, and not for who is in it. The matches become the selection."
  @Generable struct Arguments {
    @Guide(description: "What to look for, as a short visual phrase.") var query: String
    @Guide(description: "True to narrow the photos already selected instead of searching the whole library.")
    var refine: Bool?
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.searchPhotos(query: arguments.query, refine: arguments.refine)
  }
}

@available(iOS 27.0, *)
struct FindPersonPhotosTool: Tool {
  let name = "find_photos_of_person"
  let description =
    "Find the photos a named person is in. The library knows its people by name — give the name, and call ask_user when the request never says who is meant."
  @Generable struct Arguments {
    @Guide(description: "The person's name.") var name: String
    @Guide(description: "True to narrow the photos already selected instead of searching the whole library.")
    var refine: Bool?
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.findPeople(name: arguments.name, refine: arguments.refine)
  }
}

@available(iOS 27.0, *)
struct FindPhotoTextTool: Tool {
  let name = "find_photos_with_text"
  let description =
    "Find photos by words written in the picture — signs, receipts, whiteboards, screenshots, menus. Not for what the picture shows."
  @Generable struct Arguments {
    @Guide(description: "The written words to look for.") var text: String
    @Guide(description: "True to narrow the photos already selected instead of searching the whole library.")
    var refine: Bool?
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.findText(text: arguments.text, refine: arguments.refine)
  }
}

@available(iOS 27.0, *)
struct FindBlurryPhotosTool: Tool {
  let name = "find_blurry_photos"
  let description = "Find the photos that came out blurry or out of focus."
  @Generable struct Arguments {
    @Guide(description: "True to look only inside the photos already selected.") var refine: Bool?
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.findBlurry(refine: arguments.refine)
  }
}

@available(iOS 27.0, *)
struct FindDuplicatePhotosTool: Tool {
  let name = "find_duplicates"
  let description =
    "Find near-identical photos — the same shot taken twice, where one of each pair is worth deleting."
  func call(arguments: NoArguments) async throws -> String {
    PhotoLibraryBox.shared.findDuplicates()
  }
}

@available(iOS 27.0, *)
struct CheckPhotoTool: Tool {
  let name = "check_photo"
  // The judge-study ruling on the argument: the answers are enumerated and
  // one of them comes back, never an open judgment. The "never to check a
  // search result" clause is the moment-seek lane's, where the same tool's
  // ritual — verifying a search that had already succeeded — cost more
  // rounds than any other single behavior.
  let description =
    "Look at one photo and answer a question the user asked about it. Always give the possible answers — the reply is one of them. Never call this to check a search result you already have."
  @Generable struct Arguments {
    @Guide(description: "The photo's number, from a result or the state.") var id: Int
    @Guide(description: "The question about that photo.") var question: String
    @Guide(description: "The possible answers to choose from.") var options: [String]
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.check(
      id: arguments.id, question: arguments.question, options: arguments.options)
  }
}

@available(iOS 27.0, *)
struct OpenPhotoTool: Tool {
  let name = "open_photo"
  // The seek lesson, transplanted: an opener that reads as a prerequisite
  // grows a ritual in front of every other call.
  let description =
    "Open one photo full screen. Nothing needs opening first — the other tools take their own photo numbers."
  @Generable struct Arguments {
    @Guide(description: "The photo's number, from the state or a result.") var id: Int
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.open(id: arguments.id)
  }
}

@available(iOS 27.0, *)
struct AddToAlbumTool: Tool {
  let name = "add_to_album"
  let description =
    "Put the selected photos into an album, creating the album if it does not exist yet."
  @Generable struct Arguments {
    @Guide(description: "The album's name.") var album: String
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.addToAlbum(arguments.album)
  }
}

@available(iOS 27.0, *)
struct FavoritePhotosTool: Tool {
  let name = "favorite_photos"
  let description = "Mark the selected photos as favourites."
  func call(arguments: NoArguments) async throws -> String {
    PhotoLibraryBox.shared.favorite()
  }
}

@available(iOS 27.0, *)
struct DeletePhotosTool: Tool {
  let name = "delete_photos"
  let description = "Delete the selected photos."
  @Generable struct Arguments {
    @Guide(description: "Pass false unless the user has already said yes to deleting these exact photos; false shows them what would go.")
    var confirm: Bool
  }
  func call(arguments: Arguments) async throws -> String {
    PhotoLibraryBox.shared.delete(confirmed: arguments.confirm)
  }
}

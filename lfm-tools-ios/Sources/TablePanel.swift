// The records packs' stage: one shape for "the app's list, always on screen".
import Foundation

/// What a records pack (store, shopping, money, inbox) shows on stage: a
/// titled table of the rows the last finder selected — or the app's default
/// list — plus a one-line overview. The packs fill it; StageView draws it.
@available(iOS 27.0, *)
struct TablePanel: Sendable {
  let title: String
  let columns: [String]
  let rows: [[String]]
  let totalRows: Int
  let overview: String
}

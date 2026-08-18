// What a tool hands to the *user*, as opposed to the string it hands the model.
//
// A tool returns text because that is what the model can read. But "37.7, 139.0"
// is a map, "142°" is a compass, and a week of step counts is a chart. The tool
// posts the structured version here; the view draws it under the answer.
import CoreLocation
import Foundation
import UIKit

@available(iOS 27.0, *)
enum Artifact: @unchecked Sendable {
  case map(coordinate: CLLocationCoordinate2D, title: String)
  case places([Place])
  case compass(degrees: Double)
  case steps([DayStepCount])
  case photo(UIImage, caption: String)
  case translation(source: String, target: String, language: String)
  case event(title: String, start: Date, minutes: Int)
  case timer(seconds: Int, label: String)
  case tilt(pitch: Double, roll: Double, yaw: Double)
  case gauge(title: String, value: Double, unit: String, caption: String)
  case clock(date: Date, zone: String)
  case clocks(from: String, fromZone: String, to: String, toZone: String)
  case activity(kind: String, confidence: String)
  case equation(expression: String, result: String)
  case speaking(text: String)
  case notice(title: String, body: String, seconds: Int)
  case brightness(percent: Int)
  case note(text: String)
  case area(place: String, accuracy: Int, coordinate: CLLocationCoordinate2D)

  struct Place: Identifiable, @unchecked Sendable {
    let id = UUID()
    let name: String
    let metres: Int
    let category: String?
    let coordinate: CLLocationCoordinate2D?
  }
}

/// One artifact per turn. A tool posts as it returns; the chat picks it up when
/// the turn ends and pins it to the answer, so it appears with the reply rather
/// than before it.
@available(iOS 27.0, *)
final class ArtifactBox: @unchecked Sendable {
  static let shared = ArtifactBox()
  private let lock = NSLock()
  private var pending: Artifact?

  func post(_ artifact: Artifact) {
    lock.lock()
    pending = artifact
    lock.unlock()
  }

  func take() -> Artifact? {
    lock.lock()
    defer { lock.unlock() }
    let value = pending
    pending = nil
    return value
  }
}

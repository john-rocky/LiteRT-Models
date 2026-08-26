// Where the phone is, and what is near it. Every tool here prompts for
// permission the first time; a denied prompt has to read back as a refusal the
// model can relay, not as a crash.
import CoreLocation
import Foundation
import FoundationModels
import MapKit

@available(iOS 27.0, *)
struct LocationTool: Tool {
  let name = "get_location"
  // Tool descriptions and @Guide strings on the demo set are written to be
  // paid for: on the LiteRT path they are prefilled into every turn, and the
  // tool list was 84% of the prefill. Words that do not help the model route
  // or fill arguments are dropped.
  let description = "Where the device is: town or district."

  /// Town-level, on purpose. The precise fix is what the phone has, but a
  /// screen recording of it is somebody's home address — so neither the model
  /// nor the card ever sees the coordinates.
  func call(arguments: NoArguments) async throws -> String {
    try await withDeadline(12, "location") {
      let location = try await LocationBox.shared.current()
      var place = "somewhere with a GPS fix"
      if let request = MKReverseGeocodingRequest(location: location),
        let item = try? await request.mapItems.first
      {
        let address = item.address?.shortAddress ?? item.address?.fullAddress
        place = address.map { Self.coarse($0) } ?? place
      }
      ArtifactBox.shared.post(
        .area(
          place: place, accuracy: Int(location.horizontalAccuracy),
          coordinate: location.coordinate))
      return "\(place) (fix to ±\(Int(location.horizontalAccuracy)) m)"
    }
  }

  /// Keep the last two components of an address — district and city — and drop
  /// the street and the number.
  private static func coarse(_ address: String) -> String {
    let parts = address.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    return parts.suffix(2).joined(separator: ", ")
  }
}

@available(iOS 27.0, *)
struct PlaceNameTool: Tool {
  let name = "describe_location"
  let description = "The current street address."

  func call(arguments: NoArguments) async throws -> String {
    let location = try await LocationBox.shared.current()
    // MKReverseGeocodingRequest, not CLGeocoder: the CoreLocation geocoder is
    // deprecated as of the 26 SDKs and this target is 27.
    guard let request = MKReverseGeocodingRequest(location: location) else {
      return "cannot geocode this position"
    }
    let items = try await request.mapItems
    guard let item = items.first else { return "no address for this position" }
    let address = item.address?.fullAddress ?? item.address?.shortAddress
    return [item.name, address].compactMap { $0 }.joined(separator: " — ")
  }
}

@available(iOS 27.0, *)
struct SearchPlacesTool: Tool {
  let name = "search_places"
  let description = "Find shops, cafes or other places near the user."

  // `query` alone. A `limit` argument was one more thing for a 1.2B to decode
  // and to get wrong, and no request in practice named a count.
  @Generable struct Arguments {
    var query: String
  }

  func call(arguments: Arguments) async throws -> String {
    let location = try await LocationBox.shared.current()
    let request = MKLocalSearch.Request()
    request.naturalLanguageQuery = arguments.query
    request.region = MKCoordinateRegion(
      center: location.coordinate, latitudinalMeters: 3000, longitudinalMeters: 3000)
    let response = try await MKLocalSearch(request: request).start()
    let limit = 3
    let found = response.mapItems.prefix(limit).map { item in
      Artifact.Place(
        name: item.name ?? "unnamed",
        metres: Int(location.distance(from: item.location)),
        category: item.pointOfInterestCategory?.rawValue
          .replacingOccurrences(of: "MKPOICategory", with: ""),
        coordinate: item.location.coordinate)
    }
    if !found.isEmpty { ArtifactBox.shared.post(.places(Array(found))) }
    let items = found.map { "- \($0.name) (\($0.metres) m)" }
    return items.isEmpty ? "nothing found for \(arguments.query)" : items.joined(separator: "\n")
  }
}

/// One `CLLocationManager` for the whole app, wrapping the delegate callbacks as
/// a single `async` call. A fresh manager per tool call would ask for permission
/// again and be deallocated before the fix arrives.
@available(iOS 27.0, *)
final class LocationBox: NSObject, CLLocationManagerDelegate, @unchecked Sendable {
  static let shared = LocationBox()

  enum Failure: LocalizedError {
    case denied
    case unavailable(String)

    var errorDescription: String? {
      switch self {
      case .denied: return "location permission was refused"
      case .unavailable(let why): return "location unavailable: \(why)"
      }
    }
  }

  private let manager = CLLocationManager()
  private var waiting: [CheckedContinuation<CLLocation, Error>] = []
  private let lock = NSLock()

  override private init() {
    super.init()
    manager.delegate = self
    manager.desiredAccuracy = kCLLocationAccuracyHundredMeters
  }

  func current() async throws -> CLLocation {
    if let fix = manager.location, fix.timestamp.timeIntervalSinceNow > -60 { return fix }
    return try await withCheckedThrowingContinuation { continuation in
      lock.lock()
      waiting.append(continuation)
      let count = waiting.count
      lock.unlock()
      guard count == 1 else { return }  // a request is already in flight
      switch manager.authorizationStatus {
      case .notDetermined: manager.requestWhenInUseAuthorization()
      case .denied, .restricted: finish(.failure(Failure.denied))
      default: manager.requestLocation()
      }
    }
  }

  private func finish(_ result: Result<CLLocation, Error>) {
    lock.lock()
    let pending = waiting
    waiting = []
    lock.unlock()
    for continuation in pending { continuation.resume(with: result) }
  }

  func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
    switch manager.authorizationStatus {
    case .notDetermined: return
    case .denied, .restricted: finish(.failure(Failure.denied))
    default: manager.requestLocation()
    }
  }

  func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
    guard let fix = locations.last else {
      return finish(.failure(Failure.unavailable("no fix")))
    }
    finish(.success(fix))
  }

  func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
    finish(.failure(Failure.unavailable(error.localizedDescription)))
  }
}

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
  let description = "The device's current latitude and longitude."

  func call(arguments: NoArguments) async throws -> String {
    let location = try await LocationBox.shared.current()
    return String(
      format: "%.5f, %.5f (±%.0f m)", location.coordinate.latitude,
      location.coordinate.longitude, location.horizontalAccuracy)
  }
}

@available(iOS 27.0, *)
struct PlaceNameTool: Tool {
  let name = "describe_location"
  let description = "The street address of where the device is now."

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
  let description = "Find nearby places by name or category, e.g. coffee, pharmacy."

  @Generable struct Arguments {
    @Guide(description: "What to look for.")
    var query: String
    @Guide(description: "How many results to return, 1 to 5.")
    var limit: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let location = try await LocationBox.shared.current()
    let request = MKLocalSearch.Request()
    request.naturalLanguageQuery = arguments.query
    request.region = MKCoordinateRegion(
      center: location.coordinate, latitudinalMeters: 3000, longitudinalMeters: 3000)
    let response = try await MKLocalSearch(request: request).start()
    let limit = min(5, max(1, arguments.limit))
    let items = response.mapItems.prefix(limit).map { item -> String in
      let metres = Int(location.distance(from: item.location))
      return "- \(item.name ?? "unnamed") (\(metres) m)"
    }
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

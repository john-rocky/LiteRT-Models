// Drawing the artifacts. Deliberately plain: one idea per card, big numbers,
// nothing that needs reading twice on a phone screen in a video.
import Charts
import CoreLocation
import MapKit
import SwiftUI

@available(iOS 27.0, *)
struct ArtifactView: View {
  let artifact: Artifact

  var body: some View {
    card {
      switch artifact {
      case .map(let coordinate, let title):
        MapCard(coordinate: coordinate, title: title)
      case .places(let places):
        PlacesCard(places: places)
      case .compass(let degrees):
        CompassCard(degrees: degrees)
      case .steps(let days):
        StepsCard(days: days)
      case .photo(let image, let caption):
        PhotoCard(image: image, caption: caption)
      case .translation(let source, let target, let language):
        TranslationCard(source: source, target: target, language: language)
      case .event(let title, let start, let minutes):
        EventCard(title: title, start: start, minutes: minutes)
      case .timer(let seconds, let label):
        TimerCard(seconds: seconds, label: label)
      case .tilt(let pitch, let roll, let yaw):
        TiltCard(pitch: pitch, roll: roll, yaw: yaw)
      case .gauge(let title, let value, let unit, let caption):
        GaugeCard(title: title, value: value, unit: unit, caption: caption)
      case .clock(let date, let zone):
        ClockCard(date: date, zone: zone)
      case .clocks(let from, let fromZone, let to, let toZone):
        HStack(spacing: 12) {
          ClockFace(text: from, label: fromZone)
          Image(systemName: "arrow.right").foregroundStyle(.green)
          ClockFace(text: to, label: toZone)
        }
      case .activity(let kind, let confidence):
        ActivityCard(kind: kind, confidence: confidence)
      case .equation(let expression, let result):
        EquationCard(expression: expression, result: result)
      case .speaking(let text):
        SpeakingCard(text: text)
      case .notice(let title, let body, let seconds):
        NoticeCard(title: title, body: body, seconds: seconds)
      case .brightness(let percent):
        BrightnessCard(percent: percent)
      case .note(let text):
        NoteCard(text: text)
      case .area(let place, let accuracy, let coordinate):
        AreaCard(place: place, accuracy: accuracy, coordinate: coordinate)
      }
    }
  }

  private func card<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
    content()
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(12)
      .background(Color.secondary.opacity(0.14), in: .rect(cornerRadius: 14))
  }
}

@available(iOS 27.0, *)
private struct MapCard: View {
  let coordinate: CLLocationCoordinate2D
  let title: String

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      Map(initialPosition: .region(
        MKCoordinateRegion(
          center: coordinate, latitudinalMeters: 600, longitudinalMeters: 600))
      ) {
        Marker(title, coordinate: coordinate)
          .tint(.green)
      }
      .frame(height: 320)
      .clipShape(.rect(cornerRadius: 14))
      .allowsHitTesting(false)
      Text(String(format: "%.5f, %.5f", coordinate.latitude, coordinate.longitude))
        .font(.caption.monospaced())
        .foregroundStyle(.secondary)
    }
  }
}

@available(iOS 27.0, *)
private struct PlacesCard: View {
  let places: [Artifact.Place]

  private var pinned: [Artifact.Place] { places.filter { $0.coordinate != nil } }

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let region = region {
        Map(initialPosition: .region(region)) {
          ForEach(pinned) { place in
            Marker(place.name, systemImage: "cup.and.saucer.fill", coordinate: place.coordinate!)
              .tint(.green)
          }
        }
        .frame(height: 170)
        .clipShape(.rect(cornerRadius: 10))
        .allowsHitTesting(false)
      }
      ForEach(places) { place in
        HStack(alignment: .firstTextBaseline) {
          Image(systemName: "mappin.circle.fill").foregroundStyle(.green)
          VStack(alignment: .leading, spacing: 1) {
            Text(place.name).font(.title3).lineLimit(1)
            if let category = place.category, !category.isEmpty {
              Text(category).font(.caption2).foregroundStyle(.secondary)
            }
          }
          Spacer()
          Text("\(place.metres) m").font(.body).foregroundStyle(.secondary)
        }
      }
    }
  }

  /// A box around the results themselves — centred on the shops, not on the
  /// user, so the card never points at where the phone is.
  private var region: MKCoordinateRegion? {
    let points = pinned.compactMap { $0.coordinate }
    guard let first = points.first else { return nil }
    var minLat = first.latitude, maxLat = first.latitude
    var minLon = first.longitude, maxLon = first.longitude
    for p in points {
      minLat = min(minLat, p.latitude); maxLat = max(maxLat, p.latitude)
      minLon = min(minLon, p.longitude); maxLon = max(maxLon, p.longitude)
    }
    return MKCoordinateRegion(
      center: .init(latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2),
      span: .init(
        latitudeDelta: max(0.01, (maxLat - minLat) * 1.5),
        longitudeDelta: max(0.01, (maxLon - minLon) * 1.5)))
  }
}

@available(iOS 27.0, *)
private struct CompassCard: View {
  let degrees: Double

  var body: some View {
    HStack(spacing: 18) {
      ZStack {
        Circle().stroke(Color.secondary.opacity(0.35), lineWidth: 2)
        ForEach(0..<8) { i in
          Text(["N", "NE", "E", "SE", "S", "SW", "W", "NW"][i])
            .font(.system(size: 9, weight: .semibold))
            .offset(y: -38)
            .rotationEffect(.degrees(Double(i) * 45))
        }
        Image(systemName: "location.north.fill")
          .font(.title2)
          .foregroundStyle(.green)
          .rotationEffect(.degrees(degrees))
      }
      .frame(width: 92, height: 92)
      VStack(alignment: .leading) {
        Text(String(format: "%.0f°", degrees)).font(.system(size: 34, weight: .semibold))
        Text(point).foregroundStyle(.secondary)
      }
    }
  }

  private var point: String {
    ["North", "North-east", "East", "South-east", "South", "South-west", "West", "North-west"][
      Int((degrees / 45).rounded()) % 8]
  }
}

@available(iOS 27.0, *)
private struct StepsCard: View {
  let days: [DayStepCount]

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      Chart(days) { day in
        BarMark(x: .value("day", day.day, unit: .day), y: .value("steps", day.steps))
          .foregroundStyle(Color.green.gradient)
      }
      .chartXAxis {
        AxisMarks(values: .stride(by: .day)) { _ in AxisValueLabel(format: .dateTime.day()) }
      }
      .frame(height: 240)
      Text("\(days.reduce(0) { $0 + $1.steps }) steps over \(days.count) days")
        .font(.caption).foregroundStyle(.secondary)
    }
  }
}

@available(iOS 27.0, *)
private struct PhotoCard: View {
  let image: UIImage
  let caption: String

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      Image(uiImage: image)
        .resizable()
        .scaledToFit()
        .frame(maxHeight: 400)
        .clipShape(.rect(cornerRadius: 10))
      Text(caption).font(.body).foregroundStyle(.secondary)
    }
  }
}

@available(iOS 27.0, *)
private struct TranslationCard: View {
  let source: String
  let target: String
  let language: String

  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text(source).font(.title3).foregroundStyle(.secondary)
      HStack(spacing: 6) {
        Image(systemName: "arrow.down")
        Text(language.uppercased()).font(.caption2.weight(.semibold))
      }
      .foregroundStyle(.green)
      Text(target).font(.system(size: 30, weight: .semibold))
    }
  }
}

@available(iOS 27.0, *)
private struct EventCard: View {
  let title: String
  let start: Date
  let minutes: Int

  var body: some View {
    HStack(spacing: 14) {
      VStack {
        Text(start.formatted(.dateTime.month(.abbreviated))).font(.caption2)
        Text(start.formatted(.dateTime.day())).font(.title.weight(.semibold))
      }
      .frame(width: 54)
      .padding(.vertical, 8)
      .background(Color.green.opacity(0.18), in: .rect(cornerRadius: 10))
      VStack(alignment: .leading, spacing: 3) {
        Text(title).font(.headline)
        Text("\(start.formatted(date: .omitted, time: .shortened)) · \(minutes) min")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}

@available(iOS 27.0, *)
private struct TimerCard: View {
  let seconds: Int
  let label: String
  @State private var spin = false

  var body: some View {
    HStack(spacing: 16) {
      ZStack {
        Circle().stroke(Color.secondary.opacity(0.3), lineWidth: 6)
        Circle()
          .trim(from: 0, to: 0.28)
          .stroke(Color.green, style: .init(lineWidth: 6, lineCap: .round))
          .rotationEffect(.degrees(spin ? 360 : 0))
          .animation(.linear(duration: 2.2).repeatForever(autoreverses: false), value: spin)
      }
      .frame(width: 62, height: 62)
      .onAppear { spin = true }
      VStack(alignment: .leading) {
        Text(seconds >= 60 ? "\(seconds / 60):\(String(format: "%02d", seconds % 60))" : "\(seconds)s")
          .font(.system(size: 30, weight: .semibold, design: .rounded))
        Text(label).foregroundStyle(.secondary)
      }
    }
  }
}

@available(iOS 27.0, *)
private struct TiltCard: View {
  let pitch: Double
  let roll: Double
  let yaw: Double

  var body: some View {
    HStack(spacing: 18) {
      RoundedRectangle(cornerRadius: 6)
        .stroke(Color.green, lineWidth: 3)
        .frame(width: 42, height: 78)
        .rotation3DEffect(.degrees(pitch), axis: (x: 1, y: 0, z: 0))
        .rotation3DEffect(.degrees(roll), axis: (x: 0, y: 1, z: 0))
      VStack(alignment: .leading, spacing: 2) {
        ForEach([("pitch", pitch), ("roll", roll), ("yaw", yaw)], id: \.0) { name, value in
          HStack {
            Text(name).font(.caption).foregroundStyle(.secondary).frame(width: 42, alignment: .leading)
            Text(String(format: "%.0f°", value)).font(.body.monospaced())
          }
        }
      }
    }
  }
}

@available(iOS 27.0, *)
private struct GaugeCard: View {
  let title: String
  let value: Double
  let unit: String
  let caption: String

  var body: some View {
    HStack(spacing: 16) {
      ZStack {
        Circle().stroke(Color.secondary.opacity(0.3), lineWidth: 8)
        Circle()
          .trim(from: 0, to: min(1, max(0, value / 100)))
          .stroke(Color.green, style: .init(lineWidth: 8, lineCap: .round))
          .rotationEffect(.degrees(-90))
        Text("\(Int(value))").font(.title3.weight(.semibold))
      }
      .frame(width: 66, height: 66)
      VStack(alignment: .leading) {
        Text(title).font(.headline)
        Text(caption).font(.body).foregroundStyle(.secondary)
      }
      Spacer()
      Text(unit).font(.caption).foregroundStyle(.secondary)
    }
  }
}


// MARK: - The plain answers, drawn

/// A time is a clock, not a sentence.
@available(iOS 27.0, *)
private struct ClockCard: View {
  let date: Date
  let zone: String

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(date, format: .dateTime.hour().minute().second())
        .font(.system(size: 52, weight: .semibold, design: .monospaced))
        .foregroundStyle(.green)
        .contentTransition(.numericText())
      Text(date, format: .dateTime.weekday(.wide).month().day())
        .foregroundStyle(.secondary)
      Text(zone).font(.caption).foregroundStyle(.tertiary)
    }
  }
}

@available(iOS 27.0, *)
private struct ClockFace: View {
  let text: String
  let label: String

  var body: some View {
    VStack(spacing: 4) {
      Text(text)
        .font(.system(size: 30, weight: .semibold, design: .monospaced))
        .foregroundStyle(.green)
      Text(label.replacingOccurrences(of: "_", with: " "))
        .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
    }
    .frame(maxWidth: .infinity)
  }
}

/// Walking, standing, driving — as a figure that moves.
@available(iOS 27.0, *)
private struct ActivityCard: View {
  let kind: String
  let confidence: String
  @State private var step = false

  private var symbol: String {
    switch true {
    case kind.contains("walking"): return "figure.walk"
    case kind.contains("running"): return "figure.run"
    case kind.contains("cycling"): return "figure.outdoor.cycle"
    case kind.contains("vehicle"): return "car.fill"
    default: return "figure.stand"
    }
  }

  var body: some View {
    HStack(spacing: 18) {
      Image(systemName: symbol)
        .font(.system(size: 46))
        .foregroundStyle(.green)
        .offset(x: step ? 6 : -6)
        .animation(.easeInOut(duration: 0.55).repeatForever(autoreverses: true), value: step)
        .onAppear { step = true }
      VStack(alignment: .leading) {
        Text(kind.capitalized).font(.title3.weight(.semibold))
        Text("\(confidence) confidence").font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}

@available(iOS 27.0, *)
private struct EquationCard: View {
  let expression: String
  let result: String

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(expression).font(.title3.monospaced()).foregroundStyle(.secondary)
      Text("= \(result)")
        .font(.system(size: 40, weight: .semibold, design: .monospaced))
        .foregroundStyle(.green)
    }
  }
}

/// The speaker is doing something audible; the card has to say so on mute.
@available(iOS 27.0, *)
private struct SpeakingCard: View {
  let text: String
  @State private var animating = false

  var body: some View {
    HStack(spacing: 14) {
      Image(systemName: "speaker.wave.3.fill")
        .font(.title)
        .foregroundStyle(.green)
        .symbolEffect(.variableColor.iterative, isActive: animating)
      Text("“\(text)”").font(.title3).italic()
    }
    .onAppear { animating = true }
  }
}

@available(iOS 27.0, *)
private struct NoticeCard: View {
  let title: String
  let body_: String
  let seconds: Int

  init(title: String, body: String, seconds: Int) {
    self.title = title
    self.body_ = body
    self.seconds = seconds
  }

  var body: some View {
    HStack(spacing: 12) {
      RoundedRectangle(cornerRadius: 9)
        .fill(Color.green.opacity(0.25))
        .frame(width: 38, height: 38)
        .overlay(Image(systemName: "bell.fill").foregroundStyle(.green))
      VStack(alignment: .leading, spacing: 2) {
        Text(title).font(.subheadline.weight(.semibold))
        Text(body_).font(.caption).foregroundStyle(.secondary).lineLimit(2)
      }
      Spacer()
      Text("in \(seconds)s").font(.caption2).foregroundStyle(.secondary)
    }
    .padding(8)
    .background(.regularMaterial, in: .rect(cornerRadius: 12))
  }
}

@available(iOS 27.0, *)
private struct BrightnessCard: View {
  let percent: Int

  var body: some View {
    HStack(spacing: 14) {
      Image(systemName: percent > 55 ? "sun.max.fill" : "sun.min.fill")
        .font(.system(size: 34))
        .foregroundStyle(.green)
      GeometryReader { geo in
        ZStack(alignment: .leading) {
          Capsule().fill(Color.secondary.opacity(0.3))
          Capsule().fill(Color.green)
            .frame(width: geo.size.width * CGFloat(percent) / 100)
        }
      }
      .frame(height: 12)
      Text("\(percent)%").font(.headline.monospacedDigit())
    }
  }
}

@available(iOS 27.0, *)
private struct NoteCard: View {
  let text: String

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: "note.text").font(.title2).foregroundStyle(.green)
      Text(text)
    }
  }
}

/// Where you are, at the resolution a demo should show: the town, not the door.
@available(iOS 27.0, *)
private struct AreaCard: View {
  let place: String
  let accuracy: Int
  let coordinate: CLLocationCoordinate2D

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      // Country span, not street span: the map says "Japan", the words say the
      // district. A recording of somebody's home on a 600 m map is not a demo.
      Map(initialPosition: .region(
        MKCoordinateRegion(
          center: coordinate, latitudinalMeters: 2_000_000, longitudinalMeters: 2_000_000))
      ) {
        Marker(place, coordinate: coordinate).tint(.green)
      }
      .frame(height: 300)
      .clipShape(.rect(cornerRadius: 14))
      .allowsHitTesting(false)
      HStack {
        Image(systemName: "location.fill").foregroundStyle(.green)
        Text(place).font(.title2.weight(.semibold))
        Spacer()
        Text("±\(accuracy) m").font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}

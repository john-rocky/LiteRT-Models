// The four that reach furthest out of the app: a real system alarm, Apple's
// offline translator, an edit written back to the photo library, and a chart
// the agent draws.
import AlarmKit
import CoreImage
import Foundation
import FoundationModels
import Photos
import SwiftUI
import Translation
import UIKit

@available(iOS 27.0, *)
struct TimerTool: Tool {
  let name = "set_timer"
  let description = "Set a real iOS timer."

  @Generable struct Arguments {
    @Guide(description: "How long the timer runs, in seconds.")
    var seconds: Int
    @Guide(description: "What the timer is for, shown when it fires.")
    var label: String
  }

  func call(arguments: Arguments) async throws -> String {
    let seconds = max(5, arguments.seconds)
    let manager = AlarmManager.shared
    var state = manager.authorizationState
    if state == .notDetermined {
      do {
        state = try await manager.requestAuthorization()
      } catch {
        // AlarmKit refused before any alarm was described — com.apple.AlarmKit
        // .Alarm error 1 from the authorization request itself (2026-08-19).
        return try await ringAsNotification(seconds: seconds, label: arguments.label, why: "authorization failed")
      }
    }
    guard state == .authorized else {
      return try await ringAsNotification(seconds: seconds, label: arguments.label, why: "permission refused")
    }

    let title = LocalizedStringResource(stringLiteral: arguments.label)
    // A timer is a Live Activity that counts down: alarmd wants the countdown
    // and paused faces as well as the alert, or scheduling comes back with an
    // opaque com.apple.AlarmKit.Alarm error 1 (seen from the focus-session
    // compound on 2026-08-19; the alert-only form had never fired on device).
    let presentation = AlarmPresentation(
      alert: AlarmPresentation.Alert(title: title),
      countdown: AlarmPresentation.Countdown(
        title: title,
        pauseButton: AlarmButton(text: "Pause", textColor: .white, systemImageName: "pause.fill")),
      paused: AlarmPresentation.Paused(
        title: title,
        resumeButton: AlarmButton(
          text: "Resume", textColor: .white, systemImageName: "play.fill")))
    let attributes = AlarmAttributes<EmptyAlarmMetadata>(
      presentation: presentation, tintColor: .green)
    let configuration = AlarmManager.AlarmConfiguration<EmptyAlarmMetadata>.timer(
      duration: TimeInterval(seconds), attributes: attributes)
    do {
      _ = try await manager.schedule(id: UUID(), configuration: configuration)
    } catch {
      return try await ringAsNotification(seconds: seconds, label: arguments.label, why: "scheduling failed")
    }
    ArtifactBox.shared.post(.timer(seconds: seconds, label: arguments.label))
    return "timer set for \(seconds)s — it will ring in the system, not in this app"
  }

  /// A timer that cannot ring in the system rings as a notification instead —
  /// the beat still lands. AlarmKit has refused every way it has been asked
  /// on this build (error 1 from authorization and from scheduling, with and
  /// without countdown faces); the likely missing piece is a widget extension
  /// hosting the alarm's Live Activity, which this single-target app lacks.
  private func ringAsNotification(seconds: Int, label: String, why: String) async throws -> String {
    let scheduled = try await NotificationTool().call(
      arguments: .init(title: label, body: "Timer done — \(max(1, seconds / 60)) min.", seconds: seconds))
    ArtifactBox.shared.post(.timer(seconds: seconds, label: label))
    return "timer set for \(seconds)s as a notification (system alarm \(why)) — \(scheduled)"
  }
}

/// AlarmKit requires a metadata type even when there is nothing to carry.
@available(iOS 27.0, *)
struct EmptyAlarmMetadata: AlarmMetadata, Codable, Hashable {
  init() {}
}

@available(iOS 27.0, *)
struct TranslateTool: Tool {
  let name = "translate"
  let description = "Translate text to another language. Use for every translation request."

  @Generable struct Arguments {
    @Guide(description: "Text to translate, exactly as written.")
    var source: String
    @Guide(description: "Target language code; default en.", .anyOf(["en", "ja", "fr", "es", "de", "zh", "ko"]))
    var to: String?
  }

  func call(arguments: Arguments) async throws -> String {
    // The session needs the language pair already downloaded; `installedSource`
    // is the initializer that does not require a SwiftUI view to host it.
    // Absent means English: a missing required field makes FM refuse the call
    // outright, and "translate this" without naming a target is the common case.
    let targetCode = arguments.to ?? "en"
    let target = Locale.Language(identifier: targetCode)
    let sourceCode = detectSource(arguments.source)
    guard sourceCode != targetCode else {
      return "that text is already \(targetCode) — pass the original wording instead"
    }
    let source = Locale.Language(identifier: sourceCode)
    let session = TranslationSession(installedSource: source, target: target)
    do {
      // Fails fast and clearly if the pair is not on the device, instead of
      // surfacing as a translation that never arrives.
      try await session.prepareTranslation()
      let response = try await session.translate(arguments.source)
      ArtifactBox.shared.post(
        .translation(
          source: arguments.source, target: response.targetText, language: targetCode))
      return response.targetText
    } catch {
      return "cannot translate \(source.languageCode?.identifier ?? "?") to "
        + "\(targetCode) on this device: the language pair has not been downloaded "
        + "(Settings → Apps → Translate → Downloaded Languages)."
    }
  }

  private func detectSource(_ text: String) -> String {
    // Enough to pick a source: the translator needs one, and the model's own
    // guess about its input is not something to trust here.
    let japanese = text.unicodeScalars.contains { (0x3040...0x30FF).contains($0.value) }
      || text.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
    return japanese ? "ja" : "en"
  }
}

@available(iOS 27.0, *)
struct EditPhotoTool: Tool {
  let name = "edit_latest_photo"
  let description = "Filter the latest photo and save it."

  @Generable struct Arguments {
    @Guide(description: "Which filter to apply.", .anyOf(["mono", "sepia", "invert", "blur", "vivid"]))
    var filter: String
  }

  func call(arguments: Arguments) async throws -> String {
    guard let source = try await PhotoBox.latestImage() else {
      return "no photo to edit (or permission was refused)"
    }
    let names = [
      "mono": "CIPhotoEffectMono", "sepia": "CISepiaTone", "invert": "CIColorInvert",
      "blur": "CIGaussianBlur", "vivid": "CIPhotoEffectChrome",
    ]
    guard let filterName = names[arguments.filter.lowercased()] else {
      return "unknown filter; try \(names.keys.sorted().joined(separator: ", "))"
    }
    let context = CIContext()
    let input = CIImage(cgImage: source)
    guard let filter = CIFilter(name: filterName) else { return "filter unavailable" }
    filter.setValue(input, forKey: kCIInputImageKey)
    if filterName == "CIGaussianBlur" { filter.setValue(12.0, forKey: kCIInputRadiusKey) }
    guard let output = filter.outputImage,
      let cgImage = context.createCGImage(output, from: input.extent)
    else { return "could not render the filter" }

    let edited = UIImage(cgImage: cgImage)
    try await PHPhotoLibrary.shared().performChanges {
      PHAssetChangeRequest.creationRequestForAsset(from: edited)
    }
    ArtifactBox.shared.post(.photo(edited, caption: "\(arguments.filter) copy, saved"))
    return "saved a \(arguments.filter) copy to the photo library"
  }
}

@available(iOS 27.0, *)
struct StepChartTool: Tool {
  let name = "chart_steps"
  let description = "Chart steps over recent days."

  @Generable struct Arguments {
    @Guide(description: "How many days back to chart, 2 to 14.")
    var days: Int
  }

  func call(arguments: Arguments) async throws -> String {
    let days = min(14, max(2, arguments.days))
    let counts = try await MotionBox.shared.stepsByDay(days)
    guard counts.contains(where: { $0.steps > 0 }) else {
      return "no step history on this device"
    }
    ArtifactBox.shared.post(.steps(counts))
    let total = counts.reduce(0) { $0 + $1.steps }
    return "charted \(days) days, \(total) steps in total — the chart is on screen"
  }
}

public struct DayStepCount: Identifiable, Sendable {
  public let id = UUID()
  public let day: Date
  public let steps: Int
}

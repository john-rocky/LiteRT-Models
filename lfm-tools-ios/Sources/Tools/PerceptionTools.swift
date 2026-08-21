// The parts of iOS that are themselves models: Vision reads and classifies
// images, NaturalLanguage reads text, SoundAnalysis names a sound, CoreMotion
// says what the body carrying the phone is doing.
//
// All of it ships with the OS and runs offline, which is the point: a 1.2B model
// that cannot see or hear can still answer "what does this say" by calling
// something that can.
import AVFoundation
import CoreMotion
import Foundation
import FoundationModels
import NaturalLanguage
import Photos
import UIKit
import Vision

@available(iOS 27.0, *)
struct ReadPhotoTextTool: Tool {
  let name = "read_text_in_latest_photo"
  let description = "Read the text in the latest photo."

  func call(arguments: NoArguments) async throws -> String {
    // "The photo" is whatever is on the stage or attached to the message when
    // there is one; the newest library photo otherwise.
    let staged = PhotoEditBox.shared.currentCGImage()
    guard let image = try await (staged != nil ? staged : PhotoBox.latestImage()) else {
      return "no photo to read (or permission was refused)"
    }
    var request = RecognizeTextRequest()
    // Vision only looks for the languages it is told about. Left at the default
    // it read a Japanese label as "59" — the digits it could match — and the
    // whole OCR-then-translate chain had nothing to work with.
    request.recognitionLanguages = [
      Locale.Language(identifier: "ja-JP"), Locale.Language(identifier: "en-US"),
    ]
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let observations = try await request.perform(on: image)
    let lines = observations.compactMap { $0.topCandidates(1).first?.string }
    ArtifactBox.shared.post(
      .photo(UIImage(cgImage: image), caption: lines.isEmpty ? "no text found" : "read by Vision"))
    return lines.isEmpty ? "no text found in the photo" : lines.prefix(20).joined(separator: "\n")
  }
}

@available(iOS 27.0, *)
struct ClassifyPhotoTool: Tool {
  let name = "identify_latest_photo"
  let description = "Say what the latest photo shows."

  func call(arguments: NoArguments) async throws -> String {
    // "The photo" is whatever is on the stage or attached to the message when
    // there is one; the newest library photo otherwise.
    let staged = PhotoEditBox.shared.currentCGImage()
    guard let image = try await (staged != nil ? staged : PhotoBox.latestImage()) else {
      return "no photo to look at (or permission was refused)"
    }
    let request = ClassifyImageRequest()
    let observations = try await request.perform(on: image)
    let top = observations.filter { $0.hasMinimumRecall(0.01, forPrecision: 0.9) }.prefix(5)
    guard !top.isEmpty else { return "nothing recognized" }
    ArtifactBox.shared.post(
      .photo(UIImage(cgImage: image), caption: top.first.map { $0.identifier } ?? ""))
    return top.map { "\($0.identifier) \(Int($0.confidence * 100))%" }.joined(separator: ", ")
  }
}

@available(iOS 27.0, *)
enum PhotoBox {
  /// The newest asset, decoded once. Vision wants pixels, and the library hands
  /// them over asynchronously.
  static func latestImage() async throws -> CGImage? {
    let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    guard status == .authorized || status == .limited else { return nil }
    let options = PHFetchOptions()
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = 1
    guard let asset = PHAsset.fetchAssets(with: .image, options: options).firstObject else {
      return nil
    }
    let request = PHImageRequestOptions()
    request.isSynchronous = false
    request.deliveryMode = .highQualityFormat
    request.isNetworkAccessAllowed = false
    return await withCheckedContinuation { continuation in
      let once = OnceBox()
      PHImageManager.default().requestImage(
        for: asset, targetSize: CGSize(width: 1600, height: 1600),
        contentMode: .aspectFit, options: request
      ) { image, _ in
        guard once.claim() else { return }
        continuation.resume(returning: image?.cgImage)
      }
    }
  }
}

@available(iOS 27.0, *)
struct DetectLanguageTool: Tool {
  let name = "detect_language"
  let description = "Identify what language a piece of text is written in."

  @Generable struct Arguments {
    @Guide(description: "The text to identify.")
    var text: String
  }

  func call(arguments: Arguments) async throws -> String {
    let recognizer = NLLanguageRecognizer()
    recognizer.processString(arguments.text)
    let ranked = recognizer.languageHypotheses(withMaximum: 3)
    guard !ranked.isEmpty else { return "could not identify the language" }
    return ranked.sorted { $0.value > $1.value }
      .map { "\(Locale.current.localizedString(forIdentifier: $0.key.rawValue) ?? $0.key.rawValue) \(Int($0.value * 100))%" }
      .joined(separator: ", ")
  }
}

@available(iOS 27.0, *)
struct SentimentTool: Tool {
  let name = "analyze_sentiment"
  let description = "Score how positive or negative a piece of text is."

  @Generable struct Arguments {
    @Guide(description: "The text to score.")
    var text: String
  }

  func call(arguments: Arguments) async throws -> String {
    let tagger = NLTagger(tagSchemes: [.sentimentScore])
    tagger.string = arguments.text
    let (tag, _) = tagger.tag(
      at: arguments.text.startIndex, unit: .paragraph, scheme: .sentimentScore)
    guard let score = tag.flatMap({ Double($0.rawValue) }) else { return "no sentiment score" }
    let word = score > 0.25 ? "positive" : (score < -0.25 ? "negative" : "neutral")
    return String(format: "%@ (%.2f on -1…1)", word, score)
  }
}

@available(iOS 27.0, *)
struct SoundLevelTool: Tool {
  let name = "measure_sound_level"
  let description = "Measure how loud the room is."

  func call(arguments: NoArguments) async throws -> String {
    try await withDeadline(10, "the microphone") { try await MicBox.shared.level() }
  }
}

/// One audio engine, started and stopped around a short measurement. Built here
/// rather than in the tool so the engine outlives the call that starts it.
@available(iOS 27.0, *)
final class MicBox: @unchecked Sendable {
  static let shared = MicBox()
  private let engine = AVAudioEngine()

  func level() async throws -> String {
    guard await AVAudioApplication.requestRecordPermission() else {
      return "microphone permission was refused"
    }
    let input = engine.inputNode
    let format = input.inputFormat(forBus: 0)
    let peak = PeakBox()
    input.installTap(onBus: 0, bufferSize: 2048, format: format) { buffer, _ in
      guard let channel = buffer.floatChannelData?[0] else { return }
      var sum: Float = 0
      for i in 0..<Int(buffer.frameLength) { sum += channel[i] * channel[i] }
      peak.record(sqrtf(sum / Float(max(1, buffer.frameLength))))
    }
    engine.prepare()
    try engine.start()
    try await Task.sleep(for: .milliseconds(700))
    input.removeTap(onBus: 0)
    engine.stop()
    let rms = peak.value
    guard rms > 0 else { return "silence (no signal)" }
    let db = 20 * log10(rms)
    let room: String
    switch db {
    case ..<(-50): room = "very quiet"
    case ..<(-35): room = "quiet"
    case ..<(-20): room = "conversational"
    default: room = "loud"
    }
    return String(format: "%.0f dBFS — %@", db, room)
  }
}

/// The audio tap runs on its own thread; the peak has to be crossed safely.
final class PeakBox: @unchecked Sendable {
  private let lock = NSLock()
  private var peak: Float = 0
  func record(_ sample: Float) {
    lock.lock()
    peak = max(peak, sample)
    lock.unlock()
  }
  var value: Float {
    lock.lock()
    defer { lock.unlock() }
    return peak
  }
}

@available(iOS 27.0, *)
struct AttitudeTool: Tool {
  let name = "get_tilt"
  let description = "How the phone is tilted right now, in degrees."

  func call(arguments: NoArguments) async throws -> String {
    try await withDeadline(8, "the gyroscope") { try await MotionBox.shared.attitude() }
  }
}

@available(iOS 27.0, *)
struct MotionActivityTool: Tool {
  let name = "get_motion_activity"
  let description = "Whether the user is moving right now: still, walking, running or driving."

  func call(arguments: NoArguments) async throws -> String {
    guard CMMotionActivityManager.isActivityAvailable() else {
      return "activity tracking unavailable on this device"
    }
    return try await withDeadline(8, "the motion coprocessor") {
      try await MotionBox.shared.activity()
    }
  }
}

// Voice input: speech as the vaguest interface.
//
// SpeechAnalyzer (the on-device speech stack new in iOS 26) streams the mic
// into live text. The words land in the composer as they are spoken, and the
// final transcript goes through exactly the same send path as typing — the
// agent does not know or care that the sentence was said out loud.
import AVFoundation
import Foundation
import Observation
import Speech

@available(iOS 27.0, *)
@MainActor
@Observable
final class VoiceInput {
  /// Everything heard this take. Volatile results replace the tail; final
  /// results accumulate — so the text on screen edits itself as the
  /// recognizer changes its mind, which is half the demo.
  private(set) var heard = ""
  private(set) var listening = false
  private(set) var problem: String?

  private var finalized = ""
  private var analyzer: SpeechAnalyzer?
  private var transcriber: SpeechTranscriber?
  private var inputBuilder: AsyncStream<AnalyzerInput>.Continuation?
  private var reader: Task<Void, Never>?
  private var feeder: Feeder?
  private let engine = AVAudioEngine()

  func start() async {
    guard !listening else { return }
    problem = nil
    heard = ""
    finalized = ""
    guard await AVAudioApplication.requestRecordPermission() else {
      problem = "microphone permission was refused"
      return
    }
    do {
      let transcriber = SpeechTranscriber(
        locale: await Self.bestLocale(),
        transcriptionOptions: [],
        reportingOptions: [.volatileResults],
        attributeOptions: [])
      self.transcriber = transcriber
      // The locale's model may not be on the device yet; the request is nil
      // when it already is. First use can take a download.
      if let installation = try await AssetInventory.assetInstallationRequest(
        supporting: [transcriber])
      {
        try await installation.downloadAndInstall()
      }
      let analyzer = SpeechAnalyzer(modules: [transcriber])
      self.analyzer = analyzer
      guard
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(
          compatibleWith: [transcriber])
      else {
        problem = "no usable transcription format on this device"
        return
      }
      let (sequence, builder) = AsyncStream<AnalyzerInput>.makeStream()
      inputBuilder = builder
      try await analyzer.start(inputSequence: sequence)

      // Results arrive as an async sequence; this task ends on its own when
      // the analyzer is finished at stop().
      reader = Task { [weak self] in
        do {
          for try await result in transcriber.results {
            let text = String(result.text.characters)
            guard let self else { return }
            if result.isFinal {
              self.finalized += text
              self.heard = self.finalized
            } else {
              self.heard = self.finalized + text
            }
          }
        } catch {
          self?.problem = "transcription stopped: \(error.localizedDescription)"
        }
      }

      let audioSession = AVAudioSession.sharedInstance()
      try audioSession.setCategory(
        .playAndRecord, mode: .spokenAudio, options: [.duckOthers, .defaultToSpeaker])
      try audioSession.setActive(true)
      let input = engine.inputNode
      let micFormat = input.outputFormat(forBus: 0)
      // The tap fires on an audio thread; everything it needs lives in the
      // Feeder so no actor state is touched off the main actor.
      let feeder = Feeder(
        converter: AVAudioConverter(from: micFormat, to: format),
        format: format, builder: builder)
      self.feeder = feeder
      input.installTap(onBus: 0, bufferSize: 4096, format: micFormat) { buffer, _ in
        feeder.feed(buffer)
      }
      engine.prepare()
      try engine.start()
      listening = true
    } catch {
      problem = "voice input failed: \(error.localizedDescription)"
    }
  }

  /// Stops the take and returns the final transcript. Finalizing before the
  /// reader is awaited is what turns the last volatile tail into text.
  func stop() async -> String {
    guard listening else { return heard }
    listening = false
    engine.stop()
    engine.inputNode.removeTap(onBus: 0)
    inputBuilder?.finish()
    try? await analyzer?.finalizeAndFinishThroughEndOfInput()
    _ = await reader?.value
    try? AVAudioSession.sharedInstance().setActive(
      false, options: .notifyOthersOnDeactivation)
    analyzer = nil
    transcriber = nil
    inputBuilder = nil
    feeder = nil
    reader = nil
    return heard.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// The device's language when the recognizer supports it, English
  /// otherwise — the same two languages every scenario pack measures.
  private static func bestLocale() async -> Locale {
    let supported = await SpeechTranscriber.supportedLocales
    let current = Locale.current
    if supported.contains(where: { $0.identifier(.bcp47) == current.identifier(.bcp47) }) {
      return current
    }
    return supported.first { $0.identifier(.bcp47).hasPrefix("en") }
      ?? Locale(identifier: "en-US")
  }
}

/// Owns the audio-thread side of a take: format conversion and handing
/// buffers to the analyzer. Immutable after construction, so sharing it with
/// the tap callback is safe.
private final class Feeder: @unchecked Sendable {
  private let converter: AVAudioConverter?
  private let format: AVAudioFormat
  private let builder: AsyncStream<AnalyzerInput>.Continuation

  init(
    converter: AVAudioConverter?, format: AVAudioFormat,
    builder: AsyncStream<AnalyzerInput>.Continuation
  ) {
    self.converter = converter
    self.format = format
    self.builder = builder
  }

  func feed(_ buffer: AVAudioPCMBuffer) {
    if buffer.format == format {
      builder.yield(AnalyzerInput(buffer: buffer))
      return
    }
    guard let converter,
      let out = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(
          (Double(buffer.frameLength) * format.sampleRate / buffer.format.sampleRate)
            .rounded(.up)) + 16)
    else { return }
    // The input block must hand the buffer over exactly once — returning it
    // again spins the converter forever.
    var handed = false
    var conversionError: NSError?
    converter.convert(to: out, error: &conversionError) { _, status in
      if handed {
        status.pointee = .noDataNow
        return nil
      }
      handed = true
      status.pointee = .haveData
      return buffer
    }
    if conversionError == nil, out.frameLength > 0 {
      builder.yield(AnalyzerInput(buffer: out))
    }
  }
}

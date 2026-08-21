// The video-editing pack: the menus of a CapCut / LumaFusion, said out loud.
//
// The model never sees a frame. What it gets, at the top of every message, is
// the app's state — the clips on the timeline with their times, which one is
// selected, where the playhead sits, the frame size — and it turns "cut the
// first two seconds, make it vertical, fade out at the end" into
// trim_clip → crop_video → add_fade. State in, tools out. The tool names and
// argument shapes are the ones those apps' menus use, so the pack is a
// stand-in for the category, not a toy.
//
// The edit is real: an AVMutableComposition rebuilt from the clip list on
// every change (trim, split, delete, speed), a Core Image video composition
// for crop / captions / fades / the stabiliser's crop-in, an audio mix for
// volume and fades, AVAssetExportSession to write it out. The stage renders
// the frame at the moment each edit touched, through the same composition, so
// what the viewer sees is what would export.
import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import FoundationModels
import Photos
@preconcurrency import Speech
import UIKit
import Vision

// The CLIP rung (playbook spec E). CoreAI.framework ships in the iphoneos 27
// SDK but is absent from the Mac Catalyst iOSSupport tree, so this import
// resolves on the device target and nowhere else — the same shape as
// #if canImport(LiteRTLM) for the engine with no macabi slice. Catalyst (the
// whole Mac bench/stage lane) keeps the label shelf and behaves exactly as it
// did before this file learned the word "embedding".
#if canImport(CoreAIKitVision)
  import CoreAIKitVision
#endif

/// The timeline. One shared instance: the model operates "the video", and
/// which asset and which clips that means is this class's problem, not the
/// model's.
@available(iOS 27.0, *)
final class VideoEditBox: @unchecked Sendable {
  static let shared = VideoEditBox()

  /// One piece of the source on the timeline. Times are source seconds; the
  /// clip occupies `sourceDuration / speed` seconds of timeline.
  struct Clip: Sendable {
    var sourceStart: Double
    var sourceDuration: Double
    var speed: Double = 1
    var timelineDuration: Double { sourceDuration / speed }
  }

  struct Caption: Sendable {
    var text: String
    var position: String  // top | bottom
    var start: Double
    var duration: Double
  }

  /// Everything an edit can change. A value, so a snapshot for rendering is
  /// a copy and the CI handler can hold one without a lock.
  struct EditState: Sendable {
    var clips: [Clip] = []
    var selected = 0
    /// Timeline seconds. Set by the app at load (40 % in), never by an
    /// edit: it is a number the model is told and asked to copy back
    /// ("split it at the playhead"), and it stays put in timeline
    /// coordinates the way an editor's playhead does when content under it
    /// changes.
    var playhead: Double = 0
    var aspect: String?
    var fadeIn: Double = 0
    var fadeOut: Double = 0
    var volume = 100
    var stabilize: String?
    var captions: [Caption] = []
    /// Background music style, or nil. The track itself is synthesized on
    /// demand (Tools/AudioTools.swift's Synth) — no assets in the repo.
    var music: String?

    var duration: Double { clips.reduce(0) { $0 + $1.timelineDuration } }

    /// Timeline start of each clip.
    var starts: [Double] {
      var out: [Double] = []
      var cursor = 0.0
      for clip in clips {
        out.append(cursor)
        cursor += clip.timelineDuration
      }
      return out
    }
  }

  /// What the stage draws under the frame: the timeline as blocks.
  struct Snapshot: @unchecked Sendable {
    struct Block: Sendable, Identifiable {
      let id: Int
      let start: Double
      let end: Double
      let selected: Bool
      let speed: Double
    }
    let blocks: [Block]
    let duration: Double
    let playhead: Double
    let preview: Double
    let frame: String
    let captions: [(start: Double, end: Double)]
    let fadeIn: Double
    let fadeOut: Double
    let volume: Int
    let thumbnails: [UIImage]
  }

  enum Failure: LocalizedError {
    case noVideo
    var errorDescription: String? { "no video to edit (or permission was refused)" }
  }

  private let lock = NSLock()
  private var source: AVAsset?
  private var videoTrack: AVAssetTrack?
  private var audioTrack: AVAssetTrack?
  /// The frame as displayed (preferredTransform applied): a portrait phone
  /// video is 1080×1920 here even though its track says 1920×1080.
  private var sourceSize = CGSize.zero
  private var sourceDuration = 0.0
  private var state = EditState()
  private var original = EditState()
  /// The moment the stage shows: the point the last edit touched. Not the
  /// playhead — see EditState.playhead.
  private var previewTime = 0.0
  private var thumbnails: [UIImage] = []
  private var thumbnailJob: Task<Void, Never>?

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  var isLoaded: Bool { sync { source != nil } }

  // MARK: Loading

  /// `--video <path>` loads a file straight from disk instead of the
  /// library — no Photos permission, so a Mac run driven from a shell can
  /// load a fixture without a TCC dialog. Export follows the same flag:
  /// file in, file out.
  static var filePath: URL? {
    guard let flag = CommandLine.arguments.firstIndex(of: "--video"),
      CommandLine.arguments.indices.contains(flag + 1)
    else { return nil }
    return URL(fileURLWithPath: CommandLine.arguments[flag + 1])
  }

  /// The newest library video becomes the timeline: one clip, the whole
  /// thing, playhead 40 % in, nothing applied. Called by the stage before
  /// the first beat so the permission prompt fires there and not mid-run.
  func preload() async throws {
    if isLoaded { return }
    if let file = Self.filePath {
      try await load(AVURLAsset(url: file))
      return
    }
    guard let asset = try await VideoLibrary.latestAsset() else { throw Failure.noVideo }
    try await load(asset)
  }

  func load(_ asset: AVAsset) async throws {
    let video = try await asset.loadTracks(withMediaType: .video).first
    guard let video else { throw Failure.noVideo }
    let audio = try await asset.loadTracks(withMediaType: .audio).first
    let (natural, transform) = try await video.load(.naturalSize, .preferredTransform)
    let duration = try await asset.load(.duration).seconds
    let oriented = natural.applying(transform)
    let size = CGSize(width: abs(oriented.width), height: abs(oriented.height))
    var fresh = EditState()
    fresh.clips = [Clip(sourceStart: 0, sourceDuration: duration)]
    fresh.playhead = (duration * 0.4 * 10).rounded() / 10
    sync {
      source = asset
      videoTrack = video
      audioTrack = audio
      sourceSize = size
      sourceDuration = duration
      state = fresh
      original = fresh
      previewTime = fresh.playhead
      thumbnails = []
    }
    MomentIndexBox.shared.reset()
    refreshThumbnails()
  }

  // MARK: The state the model reads

  /// The line at the top of every message. Seconds with one decimal — the
  /// model copies these into arguments, and "4.9" survives that better than
  /// "0:04.9".
  func describe() -> String {
    let (s, size) = sync { (state, sourceSize) }
    guard !s.clips.isEmpty else { return "No video loaded." }
    let (w, h) = frameSize(for: s, source: size)
    let shape = w > h ? "landscape" : (w < h ? "portrait" : "square")
    var lines: [String] = []
    lines.append(
      "Timeline: \(s.clips.count) clip\(s.clips.count == 1 ? "" : "s"), "
        + "\(Self.f(s.duration)) s total, frame \(w)×\(h) (\(shape)\(s.aspect.map { ", \($0)" } ?? "")).")
    let starts = s.starts
    let clipLines = s.clips.enumerated().map { index, clip -> String in
      var text = "clip \(index + 1): \(Self.f(starts[index]))–\(Self.f(starts[index] + clip.timelineDuration)) s"
      if clip.speed != 1 { text += " at \(Self.f(clip.speed))× speed" }
      if index == s.selected { text += " (selected)" }
      return text
    }
    lines.append("Clips: " + clipLines.joined(separator: "; ") + ".")
    lines.append("Playhead: \(Self.f(s.playhead)) s.")
    var applied: [String] = []
    if s.fadeIn > 0 { applied.append("fade in \(Self.f(s.fadeIn)) s") }
    if s.fadeOut > 0 { applied.append("fade out \(Self.f(s.fadeOut)) s") }
    if s.volume != 100 { applied.append("volume \(s.volume)%") }
    if let level = s.stabilize { applied.append("stabilized (\(level))") }
    for caption in s.captions {
      applied.append(
        "caption \"\(caption.text)\" at \(caption.position), \(Self.f(caption.start))–\(Self.f(caption.start + caption.duration)) s")
    }
    if let music = s.music { applied.append("\(music) music") }
    if !applied.isEmpty { lines.append("Applied: " + applied.joined(separator: "; ") + ".") }
    return lines.joined(separator: " ")
  }

  static func f(_ seconds: Double) -> String {
    let rounded = (seconds * 10).rounded() / 10
    return rounded == rounded.rounded() ? String(Int(rounded)) : String(format: "%.1f", rounded)
  }

  /// The output frame in pixels after aspect crop and stabiliser crop-in.
  private func frameSize(for s: EditState, source: CGSize) -> (Int, Int) {
    let rect = cropRect(for: s, source: source)
    return (Int(rect.width.rounded()), Int(rect.height.rounded()))
  }

  private static let ratios: [String: CGFloat] = [
    "9:16": 9.0 / 16.0, "1:1": 1, "4:5": 4.0 / 5.0, "16:9": 16.0 / 9.0, "4:3": 4.0 / 3.0,
  ]

  private func cropRect(for s: EditState, source: CGSize) -> CGRect {
    var size = source
    if let aspect = s.aspect, let ratio = Self.ratios[aspect] {
      if source.width / source.height > ratio {
        size.width = source.height * ratio
      } else {
        size.height = source.width / ratio
      }
    }
    if let level = s.stabilize {
      let keep: CGFloat = level == "strong" ? 0.86 : (level == "light" ? 0.96 : 0.92)
      size = CGSize(width: size.width * keep, height: size.height * keep)
    }
    // Even pixel counts: encoders want them, and a half-pixel origin blurs.
    size = CGSize(
      width: (size.width / 2).rounded(.down) * 2, height: (size.height / 2).rounded(.down) * 2)
    return CGRect(
      x: ((source.width - size.width) / 2).rounded(.down),
      y: ((source.height - size.height) / 2).rounded(.down),
      width: size.width, height: size.height)
  }

  // MARK: Edits (each returns what the model is told)

  private let history = UndoStack<EditState>()

  private func mutate(_ what: String = "edit", _ body: (inout EditState) -> String) -> String {
    let result = sync {
      history.push(state, what)
      return body(&state)
    }
    refreshThumbnails()
    return result
  }

  func undoLast() -> String {
    guard let (snapshot, what) = history.pop() else { return "nothing to undo" }
    sync {
      state = snapshot
      previewTime = min(previewTime, max(0, snapshot.duration - 0.1))
    }
    refreshThumbnails()
    return "undid the last \(what) — the timeline is back to \(Self.f(snapshot.duration)) s"
  }

  func trim(edge: String, seconds: Double) -> String {
    mutate("trim") { s in
      guard s.clips.indices.contains(s.selected) else { return "no clip selected" }
      var clip = s.clips[s.selected]
      let cut = max(0, seconds) * clip.speed  // timeline seconds → source seconds
      guard cut < clip.sourceDuration - 0.2 else {
        return "clip \(s.selected + 1) is only \(Self.f(clip.timelineDuration)) s long — cannot trim \(Self.f(seconds)) s"
      }
      if edge.lowercased() == "end" {
        clip.sourceDuration -= cut
        previewTime = s.starts[s.selected] + clip.timelineDuration - 0.1
      } else {
        clip.sourceStart += cut
        clip.sourceDuration -= cut
        previewTime = s.starts[s.selected]
      }
      s.clips[s.selected] = clip
      s.playhead = min(s.playhead, max(0, s.duration - 0.1))
      return "trimmed \(Self.f(seconds)) s off the \(edge.lowercased() == "end" ? "end" : "start") of clip \(s.selected + 1); it now runs \(Self.f(s.starts[s.selected]))–\(Self.f(s.starts[s.selected] + clip.timelineDuration)) s, timeline \(Self.f(s.duration)) s"
    }
  }

  func split(at seconds: Double) -> String {
    mutate("split") { s in
      let starts = s.starts
      guard let index = s.clips.indices.first(where: {
        seconds > starts[$0] + 0.05 && seconds < starts[$0] + s.clips[$0].timelineDuration - 0.05
      }) else {
        return "\(Self.f(seconds)) s is not inside a clip (timeline is 0–\(Self.f(s.duration)) s, clip edges are cuts already)"
      }
      let clip = s.clips[index]
      let offset = (seconds - starts[index]) * clip.speed
      let first = Clip(sourceStart: clip.sourceStart, sourceDuration: offset, speed: clip.speed)
      let second = Clip(
        sourceStart: clip.sourceStart + offset, sourceDuration: clip.sourceDuration - offset,
        speed: clip.speed)
      s.clips.replaceSubrange(index...index, with: [first, second])
      s.selected = index
      previewTime = seconds
      return "split at \(Self.f(seconds)) s: clip \(index + 1) is now \(Self.f(starts[index]))–\(Self.f(seconds)) s and clip \(index + 2) is \(Self.f(seconds))–\(Self.f(starts[index] + clip.timelineDuration)) s; clip \(index + 1) is selected"
    }
  }

  func select(clip number: Int) -> String {
    mutate { s in
      guard s.clips.indices.contains(number - 1) else {
        return "there is no clip \(number); the timeline has \(s.clips.count)"
      }
      s.selected = number - 1
      let start = s.starts[number - 1]
      previewTime = start + s.clips[number - 1].timelineDuration / 2
      return "selected clip \(number) (\(Self.f(start))–\(Self.f(start + s.clips[number - 1].timelineDuration)) s)"
    }
  }

  func delete(clip number: Int) -> String {
    mutate { s in
      guard s.clips.indices.contains(number - 1) else {
        return "there is no clip \(number); the timeline has \(s.clips.count)"
      }
      guard s.clips.count > 1 else { return "cannot delete the only clip on the timeline" }
      s.clips.remove(at: number - 1)
      s.selected = min(s.selected, s.clips.count - 1)
      s.playhead = min(s.playhead, max(0, s.duration - 0.1))
      let index = min(number - 1, s.clips.count - 1)
      previewTime = s.starts[index]
      return "deleted clip \(number); \(s.clips.count) clip\(s.clips.count == 1 ? "" : "s") left, timeline \(Self.f(s.duration)) s"
    }
  }

  /// Navigation, not an edit: no history entry, nothing to undo. The
  /// playhead's "no edit moves it" rule stands — this is the user asking
  /// to move it, the one thing that does.
  func seek(to seconds: Double) -> String {
    sync {
      let clamped = min(max(0, seconds), max(0, state.duration - 0.1))
      state.playhead = (clamped * 10).rounded() / 10
      previewTime = state.playhead
      return "playhead at \(Self.f(state.playhead)) s — the frame is on screen"
    }
  }

  /// "Just the goal moment": keep only the timeline between two times. The
  /// timeline re-bases to 0, so the playhead goes home rather than pointing
  /// into footage that is gone.
  func keepRange(start: Double, end: Double) -> String {
    mutate("cut") { s in
      let duration = s.duration
      let from = max(0, min(start, end))
      let to = min(duration, max(start, end))
      guard to - from >= 0.2 else {
        return "nothing between \(Self.f(from)) and \(Self.f(to)) s (timeline is 0–\(Self.f(duration)) s)"
      }
      let starts = s.starts
      var kept: [Clip] = []
      for (index, clip) in s.clips.enumerated() {
        let clipStart = starts[index]
        let overlapStart = max(clipStart, from)
        let overlapEnd = min(clipStart + clip.timelineDuration, to)
        guard overlapEnd - overlapStart > 0.05 else { continue }
        kept.append(
          Clip(
            sourceStart: clip.sourceStart + (overlapStart - clipStart) * clip.speed,
            sourceDuration: (overlapEnd - overlapStart) * clip.speed,
            speed: clip.speed))
      }
      guard !kept.isEmpty else { return "nothing between \(Self.f(from)) and \(Self.f(to)) s" }
      s.clips = kept
      s.selected = 0
      s.playhead = 0
      previewTime = 0
      return "kept \(Self.f(from))–\(Self.f(to)) s; the timeline is now \(Self.f(s.duration)) s (was \(Self.f(duration)) s)"
    }
  }

  func setSpeed(_ multiplier: Double) -> String {
    mutate { s in
      guard s.clips.indices.contains(s.selected) else { return "no clip selected" }
      let speed = min(4, max(0.25, multiplier))
      s.clips[s.selected].speed = speed
      let start = s.starts[s.selected]
      previewTime = start + s.clips[s.selected].timelineDuration / 2
      return "clip \(s.selected + 1) now plays at \(Self.f(speed))× (\(Self.f(s.clips[s.selected].timelineDuration)) s long); timeline \(Self.f(s.duration)) s"
    }
  }

  func crop(aspect: String) -> String {
    let size = sync { sourceSize }
    guard Self.ratios[aspect] != nil else {
      return "unknown aspect; try \(Self.ratios.keys.sorted().joined(separator: ", "))"
    }
    return mutate { s in
      s.aspect = aspect
      let (w, h) = frameSize(for: s, source: size)
      return "cropped to \(aspect) — the frame is now \(w)×\(h)"
    }
  }

  func addCaption(_ text: String, position: String, start: Double, duration: Double) -> String {
    mutate("caption") { s in
      let start = min(max(0, start), max(0, s.duration - 0.5))
      let duration = min(max(0.5, duration), s.duration - start)
      let where_ = position.lowercased() == "top" ? "top" : "bottom"
      s.captions.append(Caption(text: text, position: where_, start: start, duration: duration))
      previewTime = start + min(0.5, duration / 2)
      return "caption \"\(text)\" at the \(where_), \(Self.f(start))–\(Self.f(start + duration)) s"
    }
  }

  func fade(_ which: String, seconds: Double) -> String {
    mutate("fade") { s in
      let length = min(max(0.2, seconds), max(0.2, s.duration / 2))
      switch which.lowercased() {
      case "in":
        s.fadeIn = length
        previewTime = length / 2
      case "out":
        s.fadeOut = length
        previewTime = max(0, s.duration - length / 2)
      default:
        s.fadeIn = length
        s.fadeOut = length
        previewTime = max(0, s.duration - length / 2)
      }
      return "fade \(which.lowercased()) over \(Self.f(length)) s (picture and sound)"
    }
  }

  /// Synthesize a music loop and lay it under the whole timeline. The audio
  /// pack's Synth makes the notes; here they are rendered to a file once and
  /// the composition loops it.
  func addMusic(style: String) -> String {
    let want = ["calm", "upbeat"].contains(style.lowercased()) ? style.lowercased() : "calm"
    do {
      let url = try Self.renderMusic(style: want)
      sync { musicFile = url }
      return mutate { s in
        s.music = want
        return "\(want) music added under the whole video"
      }
    } catch {
      return "could not make the music: \(error.localizedDescription)"
    }
  }

  func removeMusic() -> String {
    sync { musicFile = nil }
    return mutate { s in
      guard s.music != nil else { return "there is no music to remove" }
      s.music = nil
      return "music removed"
    }
  }

  private var musicFile: URL?

  /// Two bars of the audio pack's synth, mixed and written once per style.
  private static func renderMusic(style: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("music-\(style).m4a")
    if FileManager.default.fileExists(atPath: url.path) { return url }
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    let tempo = style == "upbeat" ? 120 : 88
    let kinds: [AudioBox.Kind] = style == "upbeat" ? [.drums, .bass] : [.keys, .bass]
    let buffers = try kinds.map { try Synth.loop($0, tempo: tempo, format: format) }
    let frames = buffers.map(\.frameLength).min() ?? 0
    guard frames > 0, let mixed = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
      throw NSError(domain: "Music", code: 1)
    }
    mixed.frameLength = frames
    let out = mixed.floatChannelData![0]
    for i in 0..<Int(frames) { out[i] = 0 }
    for buffer in buffers {
      let src = buffer.floatChannelData![0]
      for i in 0..<Int(frames) { out[i] += src[i] * 0.7 }
    }
    let file = try AVAudioFile(
      forWriting: url,
      settings: [
        AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44100, AVNumberOfChannelsKey: 1,
        AVEncoderBitRateKey: 128_000,
      ])
    try file.write(from: mixed)
    return file.url
  }

  func setVolume(_ percent: Int) -> String {
    let hasAudio = sync { audioTrack != nil }
    return mutate { s in
      s.volume = min(200, max(0, percent))
      guard hasAudio else { return "volume set to \(s.volume)% — this video has no audio track" }
      return s.volume == 0 ? "muted" : "volume set to \(s.volume)%"
    }
  }

  func stabilize(level: String) -> String {
    let size = sync { sourceSize }
    return mutate { s in
      let level = ["light", "standard", "strong"].contains(level.lowercased()) ? level.lowercased() : "standard"
      s.stabilize = level
      let (w, h) = frameSize(for: s, source: size)
      return "stabilization on (\(level)) — the frame crops in to \(w)×\(h)"
    }
  }

  func revert() -> String {
    let fresh = sync { original }
    sync { musicFile = nil }
    return mutate { s in
      guard fresh.clips.count > 0 else { return "no video loaded" }
      s = fresh
      previewTime = fresh.playhead
      return "discarded all edits — back to the original video, \(Self.f(fresh.duration)) s"
    }
  }

  /// CapCut's hero feature, on the phone's own recognizer: export the
  /// timeline's audio, transcribe it on the device, and lay the words in as
  /// captions where they were said.
  func autoCaptions() async -> String {
    do {
      let (composition, _, mix) = try await build()
      guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetAppleM4A)
      else { return "could not read the audio track" }
      session.audioMix = mix
      let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("captions-\(Int(Date().timeIntervalSince1970)).m4a")
      try await session.export(to: url, as: .m4a)
      let chunks = try await Self.transcribe(url)
      guard !chunks.isEmpty else { return "no speech found in the video" }
      var added = 0
      for chunk in chunks.prefix(6) {
        _ = addCaption(chunk.text, position: "bottom", start: chunk.start, duration: max(1.2, chunk.duration))
        added += 1
      }
      return "transcribed the speech on the device and added \(added) caption\(added == 1 ? "" : "s") where it was said"
    } catch {
      return "could not transcribe: \(error.localizedDescription)"
    }
  }

  /// File-based on-device speech recognition, grouped into caption-sized
  /// chunks (~3.5 s or 7 words). Internal, not private: the moment index
  /// builds its transcript through the same recognizer.
  static func transcribe(_ url: URL) async throws -> [(text: String, start: Double, duration: Double)] {
    let auth = await withCheckedContinuation { continuation in
      SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
    }
    guard auth == .authorized else {
      throw NSError(
        domain: "Captions", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "speech recognition permission refused"])
    }
    // The device locale is not always a recognition locale ("en_JP" is
    // English-in-Japan, which Speech does not ship): fall back to en-US
    // rather than to silence.
    let chosen = [SFSpeechRecognizer(), SFSpeechRecognizer(locale: Locale(identifier: "en-US"))]
      .compactMap { $0 }
      .first { $0.isAvailable && $0.supportsOnDeviceRecognition }
    guard let recognizer = chosen else {
      throw NSError(
        domain: "Captions", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "the speech recognizer is not available"])
    }
    let request = SFSpeechURLRecognitionRequest(url: url)
    request.requiresOnDeviceRecognition = true
    // Partials on, and every callback's segments accumulate: with partials
    // off, a file with pauses came back as its LAST utterance window only —
    // the recognizer commits and resets across silences, the intermediate
    // commits ride partial results, and the single final carries just the
    // tail (measured: five commentary lines in, one out, 2026-08-21).
    request.shouldReportPartialResults = true
    // The recognizer and its request cross into the deadline race's
    // @Sendable closure; Speech predates Sendable and the box carries them.
    struct Recognition: @unchecked Sendable {
      let recognizer: SFSpeechRecognizer
      let request: SFSpeechURLRecognitionRequest
    }
    /// Segments from every callback, deduplicated by their timestamp — a
    /// later window's partials never rewrite an earlier window's words.
    final class Collector: @unchecked Sendable {
      private let lock = NSLock()
      private var byTime: [Int: (String, Double, Double)] = [:]
      func merge(_ segments: [SFTranscriptionSegment]) {
        lock.lock()
        defer { lock.unlock() }
        for segment in segments where !segment.substring.isEmpty {
          byTime[Int(segment.timestamp * 10)] = (
            segment.substring, segment.timestamp, segment.duration
          )
        }
      }
      var sorted: [(String, Double, Double)] {
        lock.lock()
        defer { lock.unlock() }
        return byTime.values.sorted { $0.1 < $1.1 }
      }
    }
    let recognition = Recognition(recognizer: recognizer, request: request)
    let collector = Collector()
    let segments: [(String, Double, Double)] = try await firstToFinish(within: 45) {
      try await withCheckedThrowingContinuation { continuation in
        let once = OnceBox()
        recognition.recognizer.recognitionTask(with: recognition.request) { result, error in
          if let result { collector.merge(result.bestTranscription.segments) }
          if let error {
            guard once.claim() else { return }
            let kept = collector.sorted
            if kept.isEmpty {
              continuation.resume(throwing: error)
            } else {
              continuation.resume(returning: kept)
            }
            return
          }
          guard let result, result.isFinal, once.claim() else { return }
          continuation.resume(returning: collector.sorted)
        }
      }
    }
    var out: [(String, Double, Double)] = []
    var words: [String] = []
    var start = 0.0
    var end = 0.0
    for (word, at, length) in segments {
      if words.isEmpty { start = at }
      words.append(word)
      end = at + length
      if end - start > 3.5 || words.count >= 7 {
        out.append((words.joined(separator: " "), start, end - start))
        words = []
      }
    }
    if !words.isEmpty { out.append((words.joined(separator: " "), start, max(1.0, end - start))) }
    return out
  }

  // MARK: Rendering

  /// A plan the CI handler can hold: everything the frame needs, as values.
  private struct RenderPlan: Sendable {
    let crop: CGRect
    let fadeIn: Double
    let fadeOut: Double
    let duration: Double
    let captions: [Caption]

    func factor(at t: Double) -> Double {
      var k = 1.0
      if fadeIn > 0, t < fadeIn { k = min(k, max(0, t / fadeIn)) }
      if fadeOut > 0, t > duration - fadeOut { k = min(k, max(0, (duration - t) / fadeOut)) }
      return k
    }
  }

  /// The composition, the video composition and the audio mix for the state
  /// as it is now. Rebuilt from the clip list every time — cheap (no pixels
  /// move until something renders) and never stale.
  private func build() async throws -> (AVMutableComposition, AVMutableVideoComposition, AVAudioMix?) {
    let (asset, video, audio, size, s) = sync { (source, videoTrack, audioTrack, sourceSize, state) }
    guard asset != nil, let video else { throw Failure.noVideo }
    let composition = AVMutableComposition()
    guard let videoOut = composition.addMutableTrack(
      withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
    else { throw Failure.noVideo }
    let audioOut = audio == nil ? nil : composition.addMutableTrack(
      withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
    videoOut.preferredTransform = try await video.load(.preferredTransform)

    var cursor = CMTime.zero
    for clip in s.clips {
      let range = CMTimeRange(
        start: CMTime(seconds: clip.sourceStart, preferredTimescale: 600),
        duration: CMTime(seconds: clip.sourceDuration, preferredTimescale: 600))
      try videoOut.insertTimeRange(range, of: video, at: cursor)
      if let audio, let audioOut { try audioOut.insertTimeRange(range, of: audio, at: cursor) }
      let placed = CMTimeRange(start: cursor, duration: range.duration)
      let scaled = CMTime(seconds: clip.timelineDuration, preferredTimescale: 600)
      if clip.speed != 1 {
        videoOut.scaleTimeRange(placed, toDuration: scaled)
        audioOut?.scaleTimeRange(placed, toDuration: scaled)
      }
      cursor = cursor + scaled
    }

    let plan = RenderPlan(
      crop: cropRect(for: s, source: size), fadeIn: s.fadeIn, fadeOut: s.fadeOut,
      duration: s.duration, captions: s.captions)
    let videoComposition = try await AVMutableVideoComposition.videoComposition(
      with: composition,
      applyingCIFiltersWithHandler: { request in
        request.finish(with: Self.render(request.sourceImage, at: request.compositionTime.seconds, plan: plan), context: nil)
      })
    videoComposition.renderSize = plan.crop.size

    var mixParams: [AVMutableAudioMixInputParameters] = []
    func fadeRamps(_ params: AVMutableAudioMixInputParameters, volume: Float) {
      params.setVolume(volume, at: .zero)
      if s.fadeIn > 0 {
        params.setVolumeRamp(
          fromStartVolume: 0, toEndVolume: volume,
          timeRange: CMTimeRange(start: .zero, duration: CMTime(seconds: s.fadeIn, preferredTimescale: 600)))
      }
      if s.fadeOut > 0 {
        params.setVolumeRamp(
          fromStartVolume: volume, toEndVolume: 0,
          timeRange: CMTimeRange(
            start: CMTime(seconds: max(0, s.duration - s.fadeOut), preferredTimescale: 600),
            duration: CMTime(seconds: s.fadeOut, preferredTimescale: 600)))
      }
    }
    if let audioOut {
      let params = AVMutableAudioMixInputParameters(track: audioOut)
      fadeRamps(params, volume: Float(s.volume) / 100)
      mixParams.append(params)
    }
    // The music loops under the whole timeline on its own track, quiet enough
    // to sit under the original sound, and rides the same fades.
    if s.music != nil, let file = sync({ musicFile }) {
      let musicAsset = AVURLAsset(url: file)
      if let musicTrack = try await musicAsset.loadTracks(withMediaType: .audio).first,
        let musicOut = composition.addMutableTrack(
          withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
      {
        let loopSeconds = try await musicAsset.load(.duration).seconds
        var at = 0.0
        while at < s.duration, loopSeconds > 0.1 {
          let length = min(loopSeconds, s.duration - at)
          try musicOut.insertTimeRange(
            CMTimeRange(
              start: .zero, duration: CMTime(seconds: length, preferredTimescale: 600)),
            of: musicTrack, at: CMTime(seconds: at, preferredTimescale: 600))
          at += length
        }
        let params = AVMutableAudioMixInputParameters(track: musicOut)
        fadeRamps(params, volume: 0.5)
        mixParams.append(params)
      }
    }
    var mix: AVMutableAudioMix?
    if !mixParams.isEmpty {
      let audioMix = AVMutableAudioMix()
      audioMix.inputParameters = mixParams
      mix = audioMix
    }
    return (composition, videoComposition, mix)
  }

  /// One frame: crop, captions in their windows, then the fade.
  private static func render(_ source: CIImage, at t: Double, plan: RenderPlan) -> CIImage {
    var image = source
      .transformed(by: CGAffineTransform(translationX: -plan.crop.minX, y: -plan.crop.minY))
      .cropped(to: CGRect(origin: .zero, size: plan.crop.size))
    for caption in plan.captions where t >= caption.start && t <= caption.start + caption.duration {
      if let text = captionImage(caption, frame: plan.crop.size) {
        image = text.composited(over: image)
      }
    }
    let k = plan.factor(at: t)
    if k < 1 {
      let matrix = CIFilter.colorMatrix()
      matrix.inputImage = image
      matrix.rVector = CIVector(x: CGFloat(k), y: 0, z: 0, w: 0)
      matrix.gVector = CIVector(x: 0, y: CGFloat(k), z: 0, w: 0)
      matrix.bVector = CIVector(x: 0, y: 0, z: CGFloat(k), w: 0)
      matrix.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
      image = matrix.outputImage ?? image
    }
    return image.cropped(to: CGRect(origin: .zero, size: plan.crop.size))
  }

  /// White text with a soft shadow, sized to the frame, centred at the top or
  /// bottom. Core Image draws it, so the same caption reaches the preview and
  /// the export.
  private static func captionImage(_ caption: Caption, frame: CGSize) -> CIImage? {
    let generator = CIFilter.textImageGenerator()
    generator.text = caption.text
    generator.fontName = "HelveticaNeue-Bold"
    generator.fontSize = Float(frame.height * 0.055)
    generator.scaleFactor = 1
    guard var text = generator.outputImage else { return nil }
    // Force white regardless of what the generator paints, keeping its alpha.
    let white = CIFilter.colorMatrix()
    white.inputImage = text
    white.rVector = CIVector(x: 0, y: 0, z: 0, w: 0)
    white.gVector = CIVector(x: 0, y: 0, z: 0, w: 0)
    white.bVector = CIVector(x: 0, y: 0, z: 0, w: 0)
    white.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
    white.biasVector = CIVector(x: 1, y: 1, z: 1, w: 0)
    text = white.outputImage ?? text
    let box = text.extent
    let maxWidth = frame.width * 0.9
    if box.width > maxWidth {
      let scale = maxWidth / box.width
      text = text.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }
    let bounds = text.extent
    let x = (frame.width - bounds.width) / 2 - bounds.minX
    let y = caption.position == "top"
      ? frame.height - bounds.height - frame.height * 0.07 - bounds.minY
      : frame.height * 0.07 - bounds.minY
    text = text.transformed(by: CGAffineTransform(translationX: x, y: y))
    let shadow = text
      .applyingFilter("CIColorMatrix", parameters: [
        "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 0),
        "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
        "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
        "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0.8),
      ])
      .applyingGaussianBlur(sigma: Double(frame.height) * 0.004)
    return text.composited(over: shadow)
  }

  /// The frame at the point the last edit touched, through the whole
  /// pipeline — what the stage keeps on screen.
  func currentFrame() async -> UIImage? {
    guard isLoaded else { return nil }
    let t = sync { previewTime }
    return await frame(at: t, maxSize: 1280)
  }

  /// The source asset and its duration, for the moment index build.
  var sourceAsset: (asset: AVAsset, duration: Double)? {
    sync { source.map { ($0, sourceDuration) } }
  }

  /// A raw source frame, no edits applied — what the index and check_moment
  /// look at. Source seconds, which equal timeline seconds until a cut
  /// re-bases the timeline (the pack's order is find, then cut).
  func sourceFrame(at seconds: Double, maxSize: CGFloat = 640) async -> CGImage? {
    guard let (asset, duration) = sourceAsset else { return nil }
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: maxSize, height: maxSize)
    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.3, preferredTimescale: 600)
    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.3, preferredTimescale: 600)
    let t = min(max(0, seconds), max(0, duration - 0.05))
    return try? await generator.image(at: CMTime(seconds: t, preferredTimescale: 600)).image
  }

  private func frame(at seconds: Double, maxSize: CGFloat) async -> UIImage? {
    do {
      let (composition, videoComposition, _) = try await build()
      let generator = AVAssetImageGenerator(asset: composition)
      generator.videoComposition = videoComposition
      generator.maximumSize = CGSize(width: maxSize, height: maxSize)
      generator.requestedTimeToleranceBefore = CMTime(seconds: 0.15, preferredTimescale: 600)
      generator.requestedTimeToleranceAfter = CMTime(seconds: 0.15, preferredTimescale: 600)
      let duration = try await composition.load(.duration).seconds
      let t = min(max(0, seconds), max(0, duration - 0.05))
      let (image, _) = try await generator.image(at: CMTime(seconds: t, preferredTimescale: 600))
      return UIImage(cgImage: image)
    } catch {
      return nil
    }
  }

  /// Eight small frames across the timeline, for the strip. Off the calling
  /// thread; the snapshot carries whatever is ready.
  private func refreshThumbnails() {
    let job = Task.detached(priority: .utility) { [weak self] in
      guard let self else { return }
      do {
        let (composition, videoComposition, _) = try await self.build()
        let duration = try await composition.load(.duration).seconds
        guard duration > 0 else { return }
        let generator = AVAssetImageGenerator(asset: composition)
        generator.videoComposition = videoComposition
        generator.maximumSize = CGSize(width: 240, height: 240)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.5, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)
        let count = 8
        let times = (0..<count).map {
          CMTime(seconds: duration * (Double($0) + 0.5) / Double(count), preferredTimescale: 600)
        }
        var images: [UIImage] = []
        for await result in generator.images(for: times) {
          if Task.isCancelled { return }
          if let cg = try? result.image { images.append(UIImage(cgImage: cg)) }
        }
        self.sync { self.thumbnails = images }
      } catch {
        // No strip is not a failure the model needs to hear about.
      }
    }
    let previous = sync { let old = thumbnailJob; thumbnailJob = job; return old }
    previous?.cancel()
  }

  func snapshot() -> Snapshot? {
    let (s, size, preview, thumbs) = sync { (state, sourceSize, previewTime, thumbnails) }
    guard !s.clips.isEmpty else { return nil }
    let starts = s.starts
    let (w, h) = frameSize(for: s, source: size)
    return Snapshot(
      blocks: s.clips.enumerated().map { index, clip in
        Snapshot.Block(
          id: index, start: starts[index], end: starts[index] + clip.timelineDuration,
          selected: index == s.selected, speed: clip.speed)
      },
      duration: s.duration, playhead: s.playhead, preview: preview,
      frame: "\(w)×\(h)" + (s.aspect.map { " · \($0)" } ?? ""),
      captions: s.captions.map { ($0.start, $0.start + $0.duration) },
      fadeIn: s.fadeIn, fadeOut: s.fadeOut, volume: s.volume, thumbnails: thumbs)
  }

  // MARK: Export

  func export() async throws -> String {
    let (composition, videoComposition, mix) = try await build()
    guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality)
    else { return "could not start an export" }
    session.videoComposition = videoComposition
    session.audioMix = mix
    session.shouldOptimizeForNetworkUse = true
    // File mode writes next to the app's own files instead of the photo
    // library — the Photos add-permission dialog would hang a shell-driven
    // Mac run just like the read one.
    let fileMode = Self.filePath != nil
    let url =
      fileMode
      ? AppFiles.documents.appendingPathComponent("export-\(Int(Date().timeIntervalSince1970)).mp4")
      : FileManager.default.temporaryDirectory
        .appendingPathComponent("edit-\(Int(Date().timeIntervalSince1970)).mp4")
    try await session.export(to: url, as: .mp4)
    if !fileMode {
      try await PHPhotoLibrary.shared().performChanges {
        PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
      }
    }
    let s = sync { state }
    let (w, h) = frameSize(for: s, source: sync { sourceSize })
    if let poster = await frame(at: min(0.5, s.duration / 2), maxSize: 800) {
      ArtifactBox.shared.post(.photo(poster, caption: "exported \(Self.f(s.duration)) s, \(w)×\(h)"))
    }
    return fileMode
      ? "exported \(Self.f(s.duration)) s at \(w)×\(h) to \(url.lastPathComponent)"
      : "exported \(Self.f(s.duration)) s at \(w)×\(h) to the photo library"
  }
}

/// The newest video in the library, as an asset the composition can read.
@available(iOS 27.0, *)
enum VideoLibrary {
  static func latestAsset() async throws -> AVAsset? {
    let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    guard status == .authorized || status == .limited else { return nil }
    let options = PHFetchOptions()
    options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
    options.fetchLimit = 1
    guard let asset = PHAsset.fetchAssets(with: .video, options: options).firstObject else {
      return nil
    }
    let request = PHVideoRequestOptions()
    request.deliveryMode = .highQualityFormat
    request.isNetworkAccessAllowed = false
    // AVAsset is not Sendable in the compiler's eyes; the box carries it
    // out of the callback, where nothing else holds it.
    struct Handoff: @unchecked Sendable { let asset: AVAsset? }
    let handoff = await withCheckedContinuation { (continuation: CheckedContinuation<Handoff, Never>) in
      let once = OnceBox()
      PHImageManager.default().requestAVAsset(forVideo: asset, options: request) { avAsset, _, _ in
        guard once.claim() else { return }
        continuation.resume(returning: Handoff(asset: avAsset))
      }
    }
    return handoff.asset
  }
}

/// How the app's state rides into the prompt: first, as a labelled block,
/// then the person's words. Shared by the stage and the bench so a case
/// measures the same message the demo sends.
enum AppState {
  static func compose(state: String, request: String) -> String {
    "[App state] \(state)\n\n\(request)"
  }
}

// MARK: - Tools (the menu, in the words the menu uses)

@available(iOS 27.0, *)
struct TrimClipTool: Tool {
  let name = "trim_clip"
  let description = "Trim seconds off the start or the end of the selected clip."
  @Generable struct Arguments {
    @Guide(description: "Which end to shorten.", .anyOf(["start", "end"])) var edge: String
    @Guide(description: "How many seconds to remove.") var seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.trim(edge: arguments.edge, seconds: arguments.seconds)
  }
}

@available(iOS 27.0, *)
struct SplitClipTool: Tool {
  let name = "split_clip"
  let description = "Split the timeline into two clips at a point in time."
  @Generable struct Arguments {
    @Guide(description: "Timeline time in seconds to cut at. The playhead position is in the app state.")
    var seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.split(at: arguments.seconds)
  }
}

@available(iOS 27.0, *)
struct SelectClipTool: Tool {
  let name = "select_clip"
  let description = "Select a clip on the timeline so the next edit applies to it."
  @Generable struct Arguments {
    @Guide(description: "Clip number, counting from 1 at the left.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.select(clip: arguments.number)
  }
}

@available(iOS 27.0, *)
struct DeleteClipTool: Tool {
  let name = "delete_clip"
  let description = "Remove a clip from the timeline."
  @Generable struct Arguments {
    @Guide(description: "Clip number, counting from 1 at the left.") var number: Int
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.delete(clip: arguments.number)
  }
}

@available(iOS 27.0, *)
struct ClipSpeedTool: Tool {
  let name = "set_clip_speed"
  // Takes the clip too: asked to slow "the second clip", Apple FM called
  // this directly without selecting first (Mac, 2026-08-19) — and a tool
  // that silently acted on the selected clip would slow the wrong one. The
  // argument lets the direct call be the right call.
  let description = "Change the playback speed of a clip."
  @Generable struct Arguments {
    // "speed", not "multiplier": named multiplier, "half speed" arrived as 2
    // — the model reasoned about duration, not speed (Mac, 2026-08-19). The
    // argument's name is part of the contract.
    @Guide(description: "The new playback speed: 0.5 plays at half speed (slow motion means 0.5 unless a number is given), 1 is normal, 2 is double speed.")
    var speed: Double
    @Guide(description: "Which clip, from 1. Omit for the selected clip.")
    var clip: Int?
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    if let clip = arguments.clip {
      let selected = VideoEditBox.shared.select(clip: clip)
      if selected.hasPrefix("there is no clip") { return selected }
    }
    return VideoEditBox.shared.setSpeed(arguments.speed)
  }
}

@available(iOS 27.0, *)
struct CropVideoTool: Tool {
  let name = "crop_video"
  let description = "Crop the frame to an aspect ratio, for example 9:16 for a vertical video."
  @Generable struct Arguments {
    @Guide(description: "Target aspect.", .anyOf(["9:16", "1:1", "4:5", "16:9", "4:3"])) var aspect: String
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.crop(aspect: arguments.aspect)
  }
}

@available(iOS 27.0, *)
struct AddCaptionTool: Tool {
  let name = "add_caption"
  let description = "Put a text caption on the video for a span of time."
  @Generable struct Arguments {
    // The ask lives on the argument: the instructions-level "ask when a
    // caption has no words" both over- and under-fired; at the decision
    // point it is harder to miss.
    @Guide(description: "The words to show, exactly as the user gave them. If the user did not say what the caption should say, do not call this — ask them what it should say.")
    var text: String
    @Guide(description: "Where on the frame.", .anyOf(["top", "bottom"])) var position: String
    @Guide(description: "When it appears, in timeline seconds. 0 is the beginning.") var start_seconds: Double
    @Guide(description: "How long it stays, in seconds. 3 is typical.") var duration_seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.addCaption(
      arguments.text, position: arguments.position,
      start: arguments.start_seconds, duration: arguments.duration_seconds)
  }
}

@available(iOS 27.0, *)
struct AddFadeTool: Tool {
  let name = "add_fade"
  let description = "Fade the video from black at the start, to black at the end, or both."
  @Generable struct Arguments {
    @Guide(description: "Which end.", .anyOf(["in", "out", "both"])) var which: String
    @Guide(description: "Length of the fade in seconds. 1 is typical.") var seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.fade(arguments.which, seconds: arguments.seconds)
  }
}

@available(iOS 27.0, *)
struct StabilizeVideoTool: Tool {
  let name = "stabilize_video"
  let description = "Steady shaky footage. Stabilizing crops the frame in a little."
  @Generable struct Arguments {
    @Guide(description: "How hard to work.", .anyOf(["light", "standard", "strong"])) var level: String
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.stabilize(level: arguments.level)
  }
}

@available(iOS 27.0, *)
struct VideoVolumeTool: Tool {
  let name = "set_volume"
  let description = "Set the video's sound level. 0 mutes it."
  @Generable struct Arguments {
    @Guide(description: "Volume as a percentage, 0 to 200. 100 is unchanged.") var percent: Int
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.setVolume(arguments.percent)
  }
}

@available(iOS 27.0, *)
struct AutoCaptionsTool: Tool {
  let name = "auto_captions"
  let description = "Transcribe the video's speech on the device and add it as subtitles where it was said."
  func call(arguments: NoArguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return await VideoEditBox.shared.autoCaptions()
  }
}

@available(iOS 27.0, *)
struct AddMusicTool: Tool {
  let name = "add_music"
  let description = "Put background music under the video."
  @Generable struct Arguments {
    @Guide(description: "The mood.", .anyOf(["calm", "upbeat"])) var style: String
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.addMusic(style: arguments.style)
  }
}

@available(iOS 27.0, *)
struct RemoveMusicTool: Tool {
  let name = "remove_music"
  // The "not for muting" clause is load-bearing: without it, 「音を消して」
  // (kill the sound) routed here instead of to set_volume(0) — twice.
  let description = "Remove the background music track that add_music added. Not for muting the video's own sound — set_volume 0 does that."
  func call(arguments: NoArguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.removeMusic()
  }
}

@available(iOS 27.0, *)
struct MakeReelTool: Tool {
  // The compound: one name to say, one call, the app walks the steps — for
  // the job that is always the same three, and for the models that cannot
  // chain. "Make it vertical" alone should still route to crop_video; this
  // is for the person who says what they want, not how.
  let name = "make_reel"
  // "no other calls needed" is load-bearing: without it the model cropped
  // first and then called this, or walked the steps by hand and forgot the
  // export (Mac, 2026-08-19).
  let description =
    "When the user asks for a Reel, call only this: one call does the 9:16 crop, the fade-out and the export itself. Never call crop_video, add_fade or export_video alongside it."
  @Generable struct Arguments {
    @Guide(description: "Optional caption shown at the bottom for the first seconds; omit for none.")
    var caption: String?
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    var steps: [String] = []
    steps.append(VideoEditBox.shared.crop(aspect: "9:16"))
    if let text = arguments.caption, !text.isEmpty {
      steps.append(VideoEditBox.shared.addCaption(text, position: "bottom", start: 0, duration: 3))
    }
    steps.append(VideoEditBox.shared.fade("out", seconds: 1))
    steps.append(try await VideoEditBox.shared.export())
    return "made a Reel — " + steps.joined(separator: "; ")
  }
}

@available(iOS 27.0, *)
struct RevertVideoTool: Tool {
  // Same name as the photo pack's, on purpose: it is the phrase the user
  // says, and the two packs never share a session.
  let name = "revert_to_original"
  let description = "Throw away all edits and go back to the original video."
  func call(arguments: NoArguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.revert()
  }
}

@available(iOS 27.0, *)
struct ExportVideoTool: Tool {
  let name = "export_video"
  let description = "Export: render the edited video and save it to the photo library."
  func call(arguments: NoArguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return try await VideoEditBox.shared.export()
  }
}

// MARK: - The moment index (find → seek → trim → export)

/// The retrieval side of the room, built from the loaded video with what
/// the OS already ships — the orchestrator thesis's cheapest rungs, no
/// extra models: ~1 fps source frames through VNClassifyImageRequest
/// (what is seen) and VNRecognizeTextRequest (what is written), the audio
/// track through the same on-device recognizer auto_captions uses (what
/// is said). A CLIP rung slots in above the classifier when the model
/// repo's embedding build lands — today "find the goal" is answered by
/// the scoreboard's OCR and the commentator's words, not by an embedding,
/// and that gap is the demo's honest edge. The bench never reaches this:
/// it runs the canned MomentEcho (Bench/RecordingTool.swift).
///
/// Times are source seconds, which equal timeline seconds until a cut
/// re-bases the timeline — index at load, search before editing: the
/// pack's find → cut order, load-bearing here.
@available(iOS 27.0, *)
final class MomentIndexBox: @unchecked Sendable {
  static let shared = MomentIndexBox()

  struct Row: Sendable {
    var start: Double
    var end: Double
    var text: String
  }
  enum Kind { case frames, transcript, screenText }
  /// AVAsset predates Sendable; the box carries it into the deadline race,
  /// where nothing else holds it (the VideoLibrary Handoff pattern).
  struct UncheckedBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
  }
  private enum Status: Equatable { case notBuilt, building, ready(String) }

  private let lock = NSLock()
  private var frames: [Row] = []
  private var transcript: [Row] = []
  private var screenText: [Row] = []
  private var status = Status.notBuilt
  private var buildJob: Task<Void, Never>?
  /// The sample cadence this index was built on — the CLIP side merges its
  /// runs with the same 3.2×step rule the label side uses.
  private var sampleStep = 1.0

  #if canImport(CoreAIKitVision)
    /// The embedding rung: one L2-normalized vector per sampled frame, filled
    /// in the same loop that runs the OS judges so both sides see one decode.
    private var clipVectors: [(time: Double, vector: [Float])] = []

    /// Cosine floor for a CLIP hit. Measured rather than guessed:
    /// `ios/bench/cliprung` (edge-agent-lab) swept it on journey.mp4 with 11
    /// queries and the four scenes as ground truth. Labels-first, CLIP where
    /// the shelf is silent: 0.27 answers every query the labels miss and adds
    /// no false positive, while 0.24 lets two of the three should-miss queries
    /// through. The scale is per-query, not per-corpus — "a person walking
    /// away" peaks at 0.265 on footage with no person in it — so this floor is
    /// a property of this rung, not a universal constant.
    static let clipThreshold: Float = 0.27

    /// Loaded once per process: the graph compile is the expensive part and a
    /// second video must not pay it again. Never downloads — a 305 MB fetch
    /// inside an index build is a demo-killer, so the rung stays dark until
    /// the bundle is in the store (prime it once with `primeCLIP()`).
    private actor ClipLoader {
      private var encoder: ImageTextEncoder?
      private var tried = false

      func ready(download: Bool) async -> ImageTextEncoder? {
        if let encoder { return encoder }
        if tried && !download { return nil }
        tried = true
        let local = ModelStore.default.localURL(for: .clipViTB32)
        guard download || local != nil else {
          RunLog.write("MOMENTS CLIP bundle not in the store — labels only")
          return nil
        }
        do {
          let made =
            local == nil
            ? try await ImageTextEncoder(model: .clipViTB32)
            : try await ImageTextEncoder(bundleAt: local!)
          encoder = made
          return made
        } catch {
          RunLog.write("MOMENTS CLIP unavailable: \(error.localizedDescription)")
          return nil
        }
      }
    }
    private static let clipLoader = ClipLoader()

    /// Downloads the CLIP bundle if it is not already in the store. Explicit
    /// and off the index's path: call it before a take, never during one.
    static func primeCLIP() async -> Bool {
      await clipLoader.ready(download: true) != nil
    }
  #endif

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  /// One line for the state block — the mirror rule: the state claims the
  /// index only when it exists, and names which sides of it do.
  func describe() -> String {
    switch sync({ status }) {
    case .notBuilt: return "Index: not built."
    case .building: return "Index: building…"
    case .ready(let what): return "Index: ready (\(what))."
    }
  }

  func reset() {
    let job = sync { () -> Task<Void, Never>? in
      let old = buildJob
      buildJob = nil
      status = .notBuilt
      frames = []
      transcript = []
      screenText = []
      #if canImport(CoreAIKitVision)
        clipVectors = []
      #endif
      return old
    }
    job?.cancel()
  }

  /// Build once; a caller that arrives mid-build waits for the same build.
  func ensureBuilt() async {
    let job: Task<Void, Never>? = sync {
      if status == .notBuilt {
        status = .building
        let job = Task.detached(priority: .userInitiated) { await self.build() }
        buildJob = job
        return job
      }
      return buildJob
    }
    await job?.value
  }

  private func build() async {
    guard let (asset, duration) = VideoEditBox.shared.sourceAsset, duration > 0 else {
      sync { status = .notBuilt }
      return
    }
    let started = Date()
    // ≤ ~90 samples however long the video: a 10-minute video indexes at
    // one frame per ~7 s, a 40-second clip at 1 fps — the ROADMAP's
    // "10-minute video indexes in ~10 s, once" budget, kept honest.
    let step = max(1.0, duration / 90)
    var sampleTimes: [Double] = []
    var t = step / 2
    while t < duration {
      sampleTimes.append(t)
      t += step
    }
    RunLog.write("MOMENTS indexing \(sampleTimes.count) frames at every \(VideoEditBox.f(step)) s…")
    // Per-label and per-text runs: which samples saw them, merged into
    // ranges afterwards (a gap of one missed sample keeps the run alive —
    // classifiers flicker frame to frame).
    var labelTimes: [String: [Double]] = [:]
    var textTimes: [String: [Double]] = [:]
    /// Per-sample scene signature (classifier labels only) and which
    /// labels came from a detector — the scene-snap below needs both.
    var samples: [(time: Double, scene: Set<String>)] = []
    var detectorLabels = Set<String>()
    #if canImport(CoreAIKitVision)
      // Same frames, same loop: the rung above the classifier, if its bundle
      // is already on the device. Absent, the whole side is skipped and the
      // index is exactly the one every round up to r42 measured.
      let clip = await Self.clipLoader.ready(download: false)
      var vectors: [(time: Double, vector: [Float])] = []
    #endif
    for time in sampleTimes {
      if Task.isCancelled { return }
      guard let cg = await VideoEditBox.shared.sourceFrame(at: time, maxSize: 512) else { continue }
      let classify = VNClassifyImageRequest()
      let read = VNRecognizeTextRequest()
      read.recognitionLevel = .fast
      read.usesLanguageCorrection = false
      // The classifier answers in scene nouns (outdoor, ocean, street) and
      // missed a backlit puppy filling half the frame — the specialized
      // detector row of the OS shelf exists for exactly that (measured
      // 2026-08-21, the Pexels beach take).
      let animals = VNRecognizeAnimalsRequest()
      let handler = VNImageRequestHandler(cgImage: cg)
      try? handler.perform([classify, read, animals])
      var scene = Set<String>()
      for result in (classify.results ?? []).filter({ $0.confidence > 0.35 }).prefix(8) {
        let label = result.identifier.lowercased().replacingOccurrences(of: "_", with: " ")
        labelTimes[label, default: []].append(time)
        scene.insert(label)
      }
      samples.append((time, scene))
      for animal in animals.results ?? [] {
        for label in animal.labels where label.confidence > 0.5 {
          labelTimes[label.identifier.lowercased(), default: []].append(time)
          detectorLabels.insert(label.identifier.lowercased())
        }
      }
      for text in (read.results ?? []).compactMap({ $0.topCandidates(1).first?.string }) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else { continue }
        textTimes[trimmed, default: []].append(time)
      }
      #if canImport(CoreAIKitVision)
        if let clip, let vector = try? await clip.encode(image: cg) {
          vectors.append((time, vector))
        }
      #endif
    }
    func runs(_ times: [Double], label: String) -> [Row] {
      var out: [Row] = []
      for time in times.sorted() {
        // A generous gap: the animal detector dropped a mid-run frame or
        // two while the dog turned, and a split row made "that moment"
        // ambiguous downstream (measured 2026-08-21).
        if var last = out.last, time - last.end <= step * 3.2 {
          last.end = time + step / 2
          out[out.count - 1] = last
        } else {
          out.append(Row(start: max(0, time - step / 2), end: time + step / 2, text: label))
        }
      }
      return out
    }
    // Detector rows start late — the animal detector first fired 2 s after
    // the cut that opens the dog scene, and a seek to the row start
    // visibly missed the scene's head (user, 2026-08-21). Snap a detector
    // row's start backward through samples whose scene signature still
    // matches (Jaccard ≥ 0.5 on classifier labels): "the moment" a person
    // means is the scene containing the detection, and the scene starts
    // at the cut, not at the detector's first confident frame.
    func snapped(_ rows: [Row]) -> [Row] {
      rows.map { row in
        var row = row
        let firstSample = row.start + step / 2
        guard var i = samples.firstIndex(where: { abs($0.time - firstSample) < step / 4 })
        else { return row }
        let base = samples[i].scene
        guard !base.isEmpty else { return row }
        var budget = 4.0
        while i > 0, budget > 0 {
          let prev = samples[i - 1].scene
          let union = base.union(prev).count
          guard union > 0, Double(base.intersection(prev).count) / Double(union) >= 0.5
          else { break }
          i -= 1
          budget -= step
        }
        row.start = max(0, samples[i].time - step / 2)
        return row
      }
    }
    let visual = labelTimes.flatMap { entry -> [Row] in
      let rows = runs(entry.value, label: entry.key)
      return detectorLabels.contains(entry.key) ? snapped(rows) : rows
    }.sorted { $0.start < $1.start }
    let written = textTimes.flatMap { runs($0.value, label: "\"\($0.key)\"") }
      .sorted { $0.start < $1.start }
    RunLog.write(
      "MOMENTS visual side done — \(labelTimes.count) labels, \(textTimes.count) texts seen")
    // The rows themselves, longest-covered labels first: a take's beat
    // words have to name what the classifier actually said.
    let coverage = labelTimes.sorted { $0.value.count > $1.value.count }.prefix(10)
      .map { "\($0.key)×\($0.value.count)" }
    RunLog.write("MOMENTS visual labels: " + coverage.joined(separator: ", "))

    // The spoken side: the source's audio through the caption recognizer.
    // The whole side races a deadline — the permission prompt inside
    // transcribe() has no timeout of its own, and an unclicked dialog must
    // cost the transcript, not the demo (the withDeadline lesson).
    var spoken: [Row] = []
    if (try? await asset.loadTracks(withMediaType: .audio))?.first != nil {
      RunLog.write("MOMENTS transcribing the audio track…")
      let sendableAsset = UncheckedBox(asset)
      let chunks =
        (try? await firstToFinish(within: 75) { () -> [(String, Double, Double)] in
          guard
            let session = AVAssetExportSession(
              asset: sendableAsset.value, presetName: AVAssetExportPresetAppleM4A)
          else { return [] }
          let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("index-\(Int(Date().timeIntervalSince1970)).m4a")
          try await session.export(to: url, as: .m4a)
          defer { try? FileManager.default.removeItem(at: url) }
          return try await VideoEditBox.transcribe(url).map { ($0.text, $0.start, $0.duration) }
        }) ?? []
      spoken = chunks.map {
        Row(start: $0.1, end: $0.1 + max(1, $0.2), text: "\"\($0.0)\"")
      }
      if spoken.isEmpty {
        RunLog.write("MOMENTS transcript empty (no speech, or no permission)")
      } else {
        RunLog.write(
          "MOMENTS transcript rows: "
            + spoken.prefix(3).map { "\(VideoEditBox.f($0.start))s \($0.text)" }
            .joined(separator: " | "))
      }
    }

    var sides = ["frames"]
    if !spoken.isEmpty { sides.append("transcript") }
    if !written.isEmpty { sides.append("screen text") }
    #if canImport(CoreAIKitVision)
      if !vectors.isEmpty {
        sides.append("clip")
        RunLog.write("MOMENTS CLIP embedded \(vectors.count) frames")
      }
    #endif
    sync {
      frames = visual
      transcript = spoken
      screenText = written
      sampleStep = step
      #if canImport(CoreAIKitVision)
        clipVectors = vectors
      #endif
      status = .ready(sides.joined(separator: ", "))
    }
    print(
      "MOMENTS indexed — \(visual.count) visual, \(spoken.count) spoken, \(written.count) text rows in \(Int(Date().timeIntervalSince(started))) s"
    )
  }

  func search(_ kind: Kind, query: String) -> String {
    let (rows, what): ([Row], String) = sync {
      switch kind {
      case .frames: return (frames, "the picture")
      case .transcript: return (transcript, "the speech")
      case .screenText: return (screenText, "the on-screen text")
      }
    }
    guard case .ready = sync({ status }) else { return "the index is not ready yet" }
    if rows.isEmpty && kind == .transcript { return "this video has no recognized speech" }
    // Any-word match, three letters or a number up ("to" and "the" match
    // everything; "1-0" must match). Query words are the model's — expanding
    // a phrase into index-friendly words is its half of the deal.
    var tokens = query.lowercased()
      .split(whereSeparator: { " ,.!?'\"「」『』、。".contains($0) })
      .map(String.init)
      .filter { $0.count >= 3 || $0.contains(where: \.isNumber) }
    // Detector nouns in the languages a voice take can utter: the labels
    // are English, a JA query names them in JA, and data must be findable
    // in every language the pack tests (the findability rule; playbook
    // spec D chose the alias table over failing the JA case honestly).
    for (ja, en) in [("犬", "dog"), ("いぬ", "dog"), ("猫", "cat"), ("ねこ", "cat")]
    where query.contains(ja) {
      tokens.append(en)
    }
    let hits = rows.filter { row in
      let text = row.text.lowercased()
      return tokens.contains { text.contains($0) }
    }
    guard !hits.isEmpty else { return "no moments found for \"\(query)\" in \(what)" }
    // "found" leads: the model's final answer follows the strongest
    // verdict word in recent results, and a bare count lost to two
    // later "no moments found" sweeps — the hit itself must carry the
    // verdict (demo-playbook: answers follow the verdict word).
    let lines = hits.prefix(8).map { "\(VideoEditBox.f($0.start))–\(VideoEditBox.f($0.end)) s — \($0.text)" }
    return "found \(hits.count) moment\(hits.count == 1 ? "" : "s"):\n" + lines.joined(separator: "\n")
      + (hits.count > 8 ? "\n(and \(hits.count - 8) more)" : "")
  }

  /// The frames index, answered labels-first: the OS shelf keeps every case it
  /// already wins, and the CLIP rung speaks only where the shelf is silent —
  /// "a puppy running on the sand", "tall buildings seen from above", the
  /// queries with no detector, no scene noun and no on-screen text. Measured
  /// labels-first rather than blended because blending is where a rung starts
  /// costing you rounds you had already won (ios/bench/cliprung).
  ///
  /// On Mac Catalyst there is no CoreAI framework, so this is the label search
  /// and nothing else — the bench and the stage behave byte-for-byte as they
  /// did before the rung existed.
  func searchFrames(query: String) async -> String {
    let labels = search(.frames, query: query)
    #if canImport(CoreAIKitVision)
      guard labels.hasPrefix("no moments found") else { return labels }
      if let found = await clipSearch(query: query) { return found }
    #endif
    return labels
  }

  #if canImport(CoreAIKitVision)
    /// Text embedding of the model's phrase, cosine against every sampled
    /// frame, the over-threshold samples merged into windows — strongest
    /// first, because a scored index has a ranking the label side never had.
    /// It is rendered as a ranked candidate rather than as a find: the row
    /// shape is the label side's, the verdict word is not (see below).
    private func clipSearch(query: String) async -> String? {
      let (vectors, step) = sync { (clipVectors, sampleStep) }
      guard !vectors.isEmpty, let encoder = await Self.clipLoader.ready(download: false),
        let wanted = try? await encoder.encode(text: query)
      else { return nil }
      let scored = vectors.map {
        ($0.time, ImageTextEncoder.cosineSimilarity($0.vector, wanted))
      }
      let hitTimes = scored.filter { $0.1 >= Self.clipThreshold }.map { $0.0 }
      guard !hitTimes.isEmpty else {
        RunLog.write(
          "MOMENTS CLIP \"\(query)\" best \(String(format: "%.3f", scored.map(\.1).max() ?? 0)) — under \(Self.clipThreshold)"
        )
        return nil
      }
      var windows: [(row: Row, peak: Float)] = []
      for time in hitTimes.sorted() {
        let score = scored.first { $0.0 == time }?.1 ?? 0
        if var last = windows.last, time - last.row.end <= step * 3.2 {
          last.row.end = time + step / 2
          last.peak = max(last.peak, score)
          windows[windows.count - 1] = last
        } else {
          windows.append(
            (Row(start: max(0, time - step / 2), end: time + step / 2, text: query), score))
        }
      }
      windows.sort { $0.peak > $1.peak }
      RunLog.write(
        "MOMENTS CLIP \"\(query)\" → "
          + windows.prefix(3).map {
            "\(VideoEditBox.f($0.row.start))–\(VideoEditBox.f($0.row.end)) s \(String(format: "%.3f", $0.peak))"
          }.joined(separator: ", "))
      // The rung may rank, so it must not claim (playbook spec E). The A/B
      // found no cosine that separates a true hit (0.232–0.329 on this
      // footage) from a query the footage never contained (0.199–0.265),
      // because the scale is per-query rather than per-corpus, and no
      // calibration tried rescued it — so presence is not buyable with a
      // number and the threshold above stays what it is, a noise gate.
      // The verdict word decides it instead (answers follow the verdict
      // word): ranking and detection are different competences, and a
      // ranker that says "found" launders one into the other. So a CLIP row
      // opens with the label shelf's own negative verdict — the model must
      // still be able to answer "there isn't one" — and then offers the
      // best window as what it is, a candidate with a similarity attached.
      // Never a presence verb, whatever the score.
      let best = windows[0]
      var answer =
        "no labelled moment matches \"\(query)\" — the closest-looking window is "
        + "\(VideoEditBox.f(best.row.start))–\(VideoEditBox.f(best.row.end)) s "
        + "(visual similarity \(String(format: "%.2f", best.peak)), not a confirmed sighting)"
      if windows.count > 1 {
        answer +=
          "\nother close-looking windows: "
          + windows.dropFirst().prefix(4).map {
            "\(VideoEditBox.f($0.row.start))–\(VideoEditBox.f($0.row.end)) s "
              + "(\(String(format: "%.2f", $0.peak)))"
          }.joined(separator: ", ")
      }
      return answer
    }
  #endif

  /// The forced-choice check: one real frame, the OS's judges, an answer
  /// from the options given — never an open judgment (the judge-study
  /// ruling holds on the real pixels too).
  func check(at seconds: Double, question: String, options: [String]) async -> String {
    // A "moment" is a range and detectors flicker frame to frame — a check
    // pinned to one exact frame vetoed a correct search hit (the dog the
    // index saw at 14.5 s was absent from the 14.0 s frame, 2026-08-21).
    // Three frames around the asked time, truths merged, detectors first.
    /// One frame read: its truths, plus the scene signature the indexer
    /// builds (classifier labels over 0.35, first eight — see the indexing
    /// loop above), so the window below can tell a cut from a flicker with
    /// the same instrument scene-snap uses.
    func look(at time: Double) async -> (scene: Set<String>, truths: [String])? {
      guard let cg = await VideoEditBox.shared.sourceFrame(at: time, maxSize: 640)
      else { return nil }
      let classify = VNClassifyImageRequest()
      let read = VNRecognizeTextRequest()
      read.recognitionLevel = .accurate
      let animals = VNRecognizeAnimalsRequest()
      let handler = VNImageRequestHandler(cgImage: cg)
      try? handler.perform([classify, read, animals])
      var truths: [String] = []
      truths += (animals.results ?? []).flatMap { $0.labels.map { $0.identifier.lowercased() } }
      truths += (read.results ?? []).compactMap { $0.topCandidates(1).first?.string.lowercased() }
      truths += (classify.results ?? []).filter { $0.confidence > 0.2 }
        .map { $0.identifier.lowercased().replacingOccurrences(of: "_", with: " ") }
      let scene = Set(
        (classify.results ?? []).filter { $0.confidence > 0.35 }.prefix(8)
          .map { $0.identifier.lowercased().replacingOccurrences(of: "_", with: " ") })
      return (scene, truths)
    }
    /// The indexer's scene test at the same threshold (`snapped`, above),
    /// with its empty-set guard: no signature is no evidence of a cut.
    func agrees(_ a: Set<String>, _ b: Set<String>) -> Bool {
      let union = a.union(b).count
      guard union > 0 else { return true }
      return Double(a.intersection(b).count) / Double(union) >= 0.5
    }
    var reads: [(offset: Double, scene: Set<String>, truths: [String])] = []
    for offset in [-0.6, 0.0, 0.6] {
      guard let frame = await look(at: seconds + offset) else { continue }
      reads.append((offset, frame.scene, frame.truths))
    }
    // A ±0.6 s window on a cut reads two scenes at once and answers about
    // whichever one spoke loudest — the check at the dog scene's head came
    // back describing the forest before it. Scene-snap moves a detector
    // row's start back to the cut, so a check *at* a boundary time asks
    // about the scene that starts there: the forward side is the asked
    // scene. Keep the frames whose signature agrees with +0.6 (the centre
    // only if it agrees) and walk further forward for replacements.
    let back = reads.first { $0.offset == -0.6 }
    let forward = reads.first { $0.offset == 0.6 }
    if let back, let forward, !back.scene.isEmpty, !forward.scene.isEmpty,
      !agrees(back.scene, forward.scene)
    {
      reads = reads.filter { agrees($0.scene, forward.scene) }
      for offset in [1.2, 1.8] where reads.count < 3 {
        guard let frame = await look(at: seconds + offset), agrees(frame.scene, forward.scene)
        else { continue }
        reads.append((offset, frame.scene, frame.truths))
      }
      RunLog.write(
        "MOMENTS check at \(VideoEditBox.f(seconds)) s sits on a cut — keeping the forward scene: "
          + reads.map { "\(VideoEditBox.f(seconds + $0.offset)) s" }.joined(separator: ", "))
    }
    var truths = reads.flatMap { $0.truths }
    guard !truths.isEmpty else { return "no frame at \(VideoEditBox.f(seconds)) s" }
    var seen = Set<String>()
    truths = truths.filter { seen.insert($0).inserted }
    let shows = " — around \(VideoEditBox.f(seconds)) s the frame shows: "

    // The model words its options freely — "yes"/"no", "appears"/"does
    // not appear", "dog"/"no dog" — and a matcher that only compared
    // option text to labels answered "none of those" while its own
    // message listed the dog; the model then read the verdict word, not
    // the evidence list (measured 2026-08-21; the semantics here are
    // demo-playbook spec A). Negation partition first, then a direct
    // label match on the positive options, then presence decided by the
    // content words of the question and positive options together.
    func negated(_ option: String) -> Bool {
      // JA carries the negation inside the word and writes no spaces, so
      // these match as bare substrings. Without them a JA option pair is
      // all-positive: a miss falls through to "none of those" and the
      // verdict word vetoes a correct retrieval — the findability rule
      // (playbook spec D) applies to the verdict, not only to the index.
      // The second row is spec D2's: 「いいえ」 is the no half of the
      // plainest pair a model can offer, and without it はい/いいえ reads
      // all-positive (measured, r38 m-ja-check-2). 「ません」 is the polite
      // negation ありません/いません are two instances of, so it catches
      // 写っていません too; 無い/無し are the kanji spellings.
      for marker in [
        "なし", "ない", "いない", "ありません", "いません",
        "いいえ", "ません", "無い", "無し",
      ] where option.contains(marker) { return true }
      let o = " " + option.lowercased() + " "
      return o.contains(" no ") || o.contains(" not ") || o.contains("n't ")
        || o.contains(" none ") || o.contains(" without ") || o.contains(" nothing ")
    }
    let positives = options.filter { !negated($0) }
    let negatives = options.filter { negated($0) }
    if let hit = positives.first(where: { option in
      let o = option.lowercased()
      return truths.contains { $0.contains(o) || o.contains($0) }
    }) {
      return hit + shows + truths.prefix(6).joined(separator: ", ")
    }
    let stop: Set<String> = [
      "the", "there", "this", "that", "does", "did", "is", "are", "was", "were", "and",
      "not", "moment", "frame", "video", "clip", "show", "shows", "shown", "appear",
      "appears", "visible", "have", "has", "any", "still", "you", "can", "see", "around",
      "second", "seconds", "present", "yes", "true", "what", "which", "contain", "contains",
    ]
    // The two tokenizers must agree on short tokens: this one and the
    // search tokenizer above both decide what of the model's wording ever
    // reaches the truths, and search already keeps a 2-character token that
    // carries a digit ("1-0"). A bare 3-character floor here does not.
    // Measured (r39, m-ja-check-2): the model asked "Is the PK scene at 460
    // seconds?" of a frame the penalty row covers — and the floor dropped
    // "PK", the one word that row holds by name, leaving the wrapper noun
    // "scene" to test with, so the check reported a real absence of the
    // wrong word. Keep a token that is long enough, carries a digit, or is
    // written as an acronym (2+ uppercase letters as the model typed it);
    // compare lowercased as before.
    //
    // The boundary has to agree too, and did not: search splits on spaces
    // and punctuation and so keeps "1-0" whole, while this one split on
    // every non-alphanumeric and handed back "1" and "0" — a one-digit
    // content word, which is a looser presence test than the search side
    // would ever run. Same separator set as search, so a question naming
    // "1-0" meets the truths' own "1-0" directly.
    func contentWords(_ text: String) -> [String] {
      text.split(whereSeparator: { " ,.!?'\"「」『』、。".contains($0) })
        .map { (typed: String($0), word: String($0).lowercased()) }
        .filter { token in
          guard !stop.contains(token.word) else { return false }
          return token.word.count >= 3 || token.word.contains(where: \.isNumber)
            || (token.typed.count >= 2 && token.typed.allSatisfy { $0.isUppercase })
        }
        .map(\.word)
    }
    var words = contentWords(question) + positives.flatMap(contentWords)
    // JA writes no spaces, so contentWords hands back whole clauses that no
    // English label can ever contain. The same JA→EN detector-noun aliases
    // search_frames carries (above) carry the presence test across — the
    // findability rule, applied to the check (playbook spec D).
    let asked = ([question] + positives).joined(separator: " ")
    for (ja, en) in [("犬", "dog"), ("いぬ", "dog"), ("猫", "cat"), ("ねこ", "cat")]
    where asked.contains(ja) {
      words.append(en)
    }
    // A check that cannot read the question must not veto (playbook spec
    // D2). contentWords splits on non-letters and JA writes no spaces, so
    // a Japanese question arrives as one whole-clause token — and the
    // presence test asks whether a truth *contains* that token, which no
    // English label can and no short JA key can either (the direct option
    // match above is where JA truths get their chance). What follows is
    // then not an absence but blindness, and blind, the code below used to
    // fall through to the negative option, where the model reads the
    // verdict word as "not there": a confident no from a tool that never
    // read the question is the failure this lane keeps rediscovering
    // (verification vetoes retrieval). So test what the code can see — the
    // truths are written in Latin letters and digits (Vision labels, OCR
    // lines, scores), and a content word holding no ASCII letter or digit
    // could not appear in them whatever the frame held. No such word, the
    // empty list included, means the check could not evaluate: it says so
    // and leaves the model the search hit it already has. A word that does
    // hold one is testable, and an absence found with it is a real one.
    //
    // The same ruling has a second edge, and there the check *can* evaluate
    // — wrongly. A wrapper noun builds the frame of a question without
    // naming what it asks about, so a question left holding only wrapper
    // nouns is being tested for a word it never asked about: the absence is
    // real and the verdict is not. Measured (r39, m-ja-check-2): "Is the PK
    // scene at 460 seconds?" lost "PK" to the old floor, tested "scene"
    // against a frame the penalty row covers, found it absent honestly, and
    // told the model the PK was not there. The floor is fixed above; this
    // is the case that survives it, because a question can also simply be
    // worded that way. Wrappers are their own list, not stopwords — a
    // stopword is dropped and the rest of the question still carries it,
    // while a question that is *all* wrapper has nothing left to carry.
    let wrapper: Set<String> = [
      "scene", "moment", "part", "section", "place", "spot", "thing", "area",
      "場面", "瞬間", "部分", "箇所", "ところ",
    ]
    let named = words.filter { !wrapper.contains($0) }
    let testable = named.contains { word in
      word.contains { $0.isASCII && ($0.isLetter || $0.isNumber) }
    }
    guard testable else {
      return "cannot tell from this frame" + shows + truths.prefix(8).joined(separator: ", ")
    }
    let present = words.contains { word in truths.contains { $0.contains(word) } }
    if present {
      let verdict = positives.first { !["yes", "true"].contains($0.lowercased()) } ?? "yes"
      return verdict + shows + truths.prefix(6).joined(separator: ", ")
    }
    if let no = negatives.first ?? options.first(where: { ["no", "false"].contains($0.lowercased()) }) {
      return no + shows + truths.prefix(6).joined(separator: ", ")
    }
    return "none of those" + shows + truths.prefix(8).joined(separator: ", ")
  }
}

@available(iOS 27.0, *)
struct SearchFramesTool: Tool {
  let name = "search_frames"
  let description =
    "Find moments in the video by what is visible in the picture — objects, actions, scenes. Not for spoken words or for text shown on screen. Returns candidate moments with their times in seconds."
  @Generable struct Arguments {
    @Guide(description: "What to look for, as a short visual phrase.") var query: String
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    await MomentIndexBox.shared.ensureBuilt()
    return await MomentIndexBox.shared.searchFrames(query: arguments.query)
  }
}

@available(iOS 27.0, *)
struct SearchTranscriptTool: Tool {
  let name = "search_transcript"
  let description =
    "Find moments by the words spoken in the video. Returns the line and when it was said, in seconds."
  @Generable struct Arguments {
    @Guide(description: "The spoken words to search for.") var query: String
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    await MomentIndexBox.shared.ensureBuilt()
    return MomentIndexBox.shared.search(.transcript, query: arguments.query)
  }
}

@available(iOS 27.0, *)
struct SearchScreenTextTool: Tool {
  let name = "search_screen_text"
  let description =
    "Find moments where text is visible in the frame — signs, scoreboards, slides, banners. Not for the spoken words. Returns the text and when it is on screen, in seconds."
  @Generable struct Arguments {
    @Guide(description: "The on-screen text to search for.") var query: String
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    await MomentIndexBox.shared.ensureBuilt()
    return MomentIndexBox.shared.search(.screenText, query: arguments.query)
  }
}

@available(iOS 27.0, *)
struct CheckMomentTool: Tool {
  let name = "check_moment"
  // Forced choice on the argument, the judge-study ruling: the check takes
  // its candidates and answers with one of them, never an open judgment.
  // The "never to check a search result" clause is r33's: without it the
  // model called this after nearly every successful search, verifying its
  // own retrieval (the remove_music "not for muting" pattern).
  let description =
    "Look at the frame at one moment and answer a question the user asked about it. Always give the possible answers — the reply is one of them. Never call this to check a search result you already have."
  @Generable struct Arguments {
    @Guide(description: "The moment to look at, in timeline seconds — from a search result or the state.")
    var seconds: Double
    @Guide(description: "The question about that frame.") var question: String
    @Guide(description: "The possible answers to choose from.") var options: [String]
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return await MomentIndexBox.shared.check(
      at: arguments.seconds, question: arguments.question, options: arguments.options)
  }
}

@available(iOS 27.0, *)
struct SeekTool: Tool {
  let name = "seek"
  // "Edits do not need a seek first": r34 grew a seek-seek-edit ritual
  // (seek 30, seek 60, keep_range 30–60) — the edits take their own times.
  let description =
    "Move the playhead to a time in seconds and show that frame. Edits do not need a seek first — they take their own times."
  @Generable struct Arguments {
    @Guide(description: "Timeline time in seconds — from the request, the state, or a search result.")
    var seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.seek(to: arguments.seconds)
  }
}

@available(iOS 27.0, *)
struct KeepRangeTool: Tool {
  let name = "keep_range"
  // r33 taught this tool its three clauses: "no split_clip first" (the
  // model prepended a split in three cases), the start guide's "the
  // moment's own start, not 0" (it kept 0–226 for "just the goal
  // moment"), and the ask on the argument (asked to cut "that one
  // moment", it invented 0–240 from the playhead — the add_caption
  // lesson: the ask lives where the decision is made).
  // "nothing after" is gone (r34): the model obeyed it into dropping the
  // export the request asked for — the clause meant "no cleanup calls"
  // and was read as "the turn ends here".
  let description =
    "Keep only the part of the timeline between two times and cut away everything else. 'Just the goal moment' is this one call with the moment's start and end — no split_clip first. Not for shaving seconds off an edge; trim_clip does that."
  @Generable struct Arguments {
    @Guide(description: "Where the kept part starts, in timeline seconds — the moment's own start, from the request or a search result; not 0 unless the user means from the beginning. If no moment or times are named anywhere, do not call this — ask which moment.")
    var start_seconds: Double
    @Guide(description: "Where the kept part ends, in timeline seconds — the moment's own end.")
    var end_seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return VideoEditBox.shared.keepRange(start: arguments.start_seconds, end: arguments.end_seconds)
  }
}

@available(iOS 27.0, *)
struct DoneTool: Tool {
  let name = "done"
  // The ritual sink (r45). Three lanes measured the same habit: after the
  // model answers, it calls something. Teaching check_moment to answer
  // doubled its own call rate and the floor ratcheted; taking check_moment
  // out of a take's room did not remove the ritual — the first beat grew a
  // seek instead, 9 of 9. So the hypothesis under test is that the slot
  // wants an occupant rather than a particular tool, and this is the
  // harmless occupant: no world, no state, no verdict, nothing a later
  // call can be built on. It is a tool in the room and never a clause in
  // the instructions — r34 measured what a stop sentence costs, when
  // keep_range's "nothing after" was read as "the turn ends here" and the
  // export the request asked for was dropped.
  let description =
    "Call when the request has been carried out and nothing is left to do. Takes no arguments and changes nothing."
  func call(arguments: NoArguments) async throws -> String {
    "acknowledged"
  }
}

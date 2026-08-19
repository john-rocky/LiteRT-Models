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
import UIKit

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

  /// The newest library video becomes the timeline: one clip, the whole
  /// thing, playhead 40 % in, nothing applied. Called by the stage before
  /// the first beat so the permission prompt fires there and not mid-run.
  func preload() async throws {
    if isLoaded { return }
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

  private func mutate(_ body: (inout EditState) -> String) -> String {
    let result = sync { body(&state) }
    refreshThumbnails()
    return result
  }

  func trim(edge: String, seconds: Double) -> String {
    mutate { s in
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
    mutate { s in
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
    mutate { s in
      let start = min(max(0, start), max(0, s.duration - 0.5))
      let duration = min(max(0.5, duration), s.duration - start)
      let where_ = position.lowercased() == "top" ? "top" : "bottom"
      s.captions.append(Caption(text: text, position: where_, start: start, duration: duration))
      previewTime = start + min(0.5, duration / 2)
      return "caption \"\(text)\" at the \(where_), \(Self.f(start))–\(Self.f(start + duration)) s"
    }
  }

  func fade(_ which: String, seconds: Double) -> String {
    mutate { s in
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
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("edit-\(Int(Date().timeIntervalSince1970)).mp4")
    try await session.export(to: url, as: .mp4)
    try await PHPhotoLibrary.shared().performChanges {
      PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
    }
    let s = sync { state }
    let (w, h) = frameSize(for: s, source: sync { sourceSize })
    if let poster = await frame(at: min(0.5, s.duration / 2), maxSize: 800) {
      ArtifactBox.shared.post(.photo(poster, caption: "exported \(Self.f(s.duration)) s, \(w)×\(h)"))
    }
    return "exported \(Self.f(s.duration)) s at \(w)×\(h) to the photo library"
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
    @Guide(description: "Speed multiplier, 0.25 to 4. 0.5 is slow motion, 2 is twice as fast.")
    var multiplier: Double
    @Guide(description: "Which clip, from 1. Omit for the selected clip.")
    var clip: Int?
  }
  func call(arguments: Arguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    if let clip = arguments.clip {
      let selected = VideoEditBox.shared.select(clip: clip)
      if selected.hasPrefix("there is no clip") { return selected }
    }
    return VideoEditBox.shared.setSpeed(arguments.multiplier)
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
    @Guide(description: "The words to show.") var text: String
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
  // "(added with add_music)" is load-bearing: without it, 「音を消して」
  // (kill the sound) routed here instead of to set_volume(0).
  let description = "Take the background music (added with add_music) off the video."
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
    "Turn the video into a ready-to-post vertical Reel: this one call does the 9:16 crop, the fade-out and the export itself — no other calls needed."
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
  let description = "Render the edited video and save it to the photo library."
  func call(arguments: NoArguments) async throws -> String {
    try await VideoEditBox.shared.preload()
    return try await VideoEditBox.shared.export()
  }
}

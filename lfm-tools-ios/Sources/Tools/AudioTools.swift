// The audio pack: a GarageBand mixer, said out loud.
//
// Four tracks — drums, bass, keys, lead — synthesized on the phone from a
// chord progression (no audio assets in the repo), looped through an
// AVAudioEngine graph: player → effects → per-track mixer (volume, pan,
// mute) → main mixer. The tools are the mixer's own controls in its own
// words: track volume and pan, mute and solo, an effect on a track,
// duplicate and delete, tempo, a fade, play / stop, export. State in, tools
// out again: every message opens with the track list as it is — names,
// levels, pans, effects, what is muted, the tempo, whether it is playing —
// so "turn the keys down a bit" is a number the model reads (70) and lowers
// (55), and "solo the drums" names a track that exists.
import AVFoundation
import Foundation
import FoundationModels

@available(iOS 27.0, *)
final class AudioBox: @unchecked Sendable {
  static let shared = AudioBox()

  enum Kind: String, Sendable, CaseIterable { case drums, bass, keys, lead }

  struct Track: Sendable {
    let id: Int
    var name: String
    var kind: Kind
    var volume: Int  // 0–100
    var pan: Int  // -100 (left) … 100 (right)
    var muted = false
    var solo = false
    var effects: [String] = []
  }

  struct SongState: Sendable {
    var tracks: [Track]
    var tempo: Int
    var bars = 8
    var fadeIn = 0.0
    var fadeOut = 0.0
    var playing = false
    var playhead = 0.0
    var nextID = 5

    var duration: Double { Double(bars) * 4 * 60 / Double(tempo) }
  }

  static let effects = ["reverb", "echo", "distortion", "lowpass"]

  private let lock = NSLock()
  private var state = SongState(
    tracks: [
      Track(id: 1, name: "Drums", kind: .drums, volume: 80, pan: 0),
      Track(id: 2, name: "Bass", kind: .bass, volume: 75, pan: 0),
      Track(id: 3, name: "Keys", kind: .keys, volume: 70, pan: -25),
      Track(id: 4, name: "Lead", kind: .lead, volume: 65, pan: 30),
    ], tempo: 110)
  private let original: SongState
  private var engine: AVAudioEngine?
  private var trackMixers: [Int: AVAudioMixerNode] = [:]
  private var playbackJob: Task<Void, Never>?
  private var startedAt: Date?

  private init() { original = state }

  private func sync<T>(_ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
  }

  // MARK: The state the model reads

  func describe() -> String {
    let s = sync { state }
    var line = "Song: \(s.tracks.count) track\(s.tracks.count == 1 ? "" : "s"), \(s.bars) bars at \(s.tempo) bpm (\(Self.f(s.duration)) s), "
    line += s.playing ? "playing at \(Self.f(s.playhead)) s." : "stopped."
    let rows = s.tracks.enumerated().map { index, t -> String in
      var text = "\(index + 1) \(t.name): volume \(t.volume), pan \(Self.panText(t.pan))"
      if !t.effects.isEmpty { text += ", effects \(t.effects.joined(separator: "+"))" }
      if t.muted { text += ", muted" }
      if t.solo { text += ", solo" }
      return text
    }
    line += " Tracks: " + rows.joined(separator: "; ") + "."
    var applied: [String] = []
    if s.fadeIn > 0 { applied.append("fade in \(Self.f(s.fadeIn)) s") }
    if s.fadeOut > 0 { applied.append("fade out \(Self.f(s.fadeOut)) s") }
    if !applied.isEmpty { line += " Applied: " + applied.joined(separator: "; ") + "." }
    return line
  }

  static func f(_ seconds: Double) -> String {
    let rounded = (seconds * 10).rounded() / 10
    return rounded == rounded.rounded() ? String(Int(rounded)) : String(format: "%.1f", rounded)
  }

  static func panText(_ pan: Int) -> String {
    pan == 0 ? "center" : (pan < 0 ? "L\(-pan)" : "R\(pan)")
  }

  // MARK: Finding a track by what the user calls it

  private func index(of name: String, in s: SongState) -> Int? {
    let needle = name.lowercased().trimmingCharacters(in: .whitespaces)
    if let n = Int(needle), s.tracks.indices.contains(n - 1) { return n - 1 }
    if let exact = s.tracks.firstIndex(where: { $0.name.lowercased() == needle }) { return exact }
    if let partial = s.tracks.firstIndex(where: {
      $0.name.lowercased().contains(needle) || needle.contains($0.name.lowercased())
        || $0.kind.rawValue == needle
    }) { return partial }
    return nil
  }

  private func noTrack(_ name: String, _ s: SongState) -> String {
    "no track called \"\(name)\"; the tracks are \(s.tracks.map(\.name).joined(separator: ", "))"
  }

  // MARK: Edits (each returns what the model is told)

  private func mutate(_ body: (inout SongState) -> String) -> String {
    let (result, snapshot) = sync { (body(&state), state) }
    applyLevels(snapshot)
    postMixer(snapshot)
    return result
  }

  func setVolume(track: String, percent: Int) -> String {
    mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      let was = s.tracks[i].volume
      s.tracks[i].volume = min(100, max(0, percent))
      return "\(s.tracks[i].name) volume \(was) → \(s.tracks[i].volume)"
    }
  }

  func setPan(track: String, pan: Int) -> String {
    mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      s.tracks[i].pan = min(100, max(-100, pan))
      return "\(s.tracks[i].name) panned \(Self.panText(s.tracks[i].pan))"
    }
  }

  func mute(track: String, on: Bool) -> String {
    mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      s.tracks[i].muted = on
      return "\(s.tracks[i].name) \(on ? "muted" : "unmuted")"
    }
  }

  func solo(track: String, on: Bool) -> String {
    mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      s.tracks[i].solo = on
      let soloed = s.tracks.filter(\.solo).map(\.name)
      return on
        ? "\(s.tracks[i].name) soloed — you hear only \(soloed.joined(separator: " and "))"
        : "\(s.tracks[i].name) solo off" + (soloed.isEmpty ? " — all tracks audible again" : "")
    }
  }

  func addEffect(track: String, effect: String) -> String {
    let want = effect.lowercased()
    guard Self.effects.contains(want) else { return "unknown effect; try \(Self.effects.joined(separator: ", "))" }
    let result = mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      if !s.tracks[i].effects.contains(want) { s.tracks[i].effects.append(want) }
      return "\(want) on \(s.tracks[i].name) (effects: \(s.tracks[i].effects.joined(separator: " → ")))"
    }
    rebuildIfPlaying()
    return result
  }

  func removeEffect(track: String, effect: String) -> String {
    let want = effect.lowercased()
    let result = mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      guard let at = s.tracks[i].effects.firstIndex(of: want) else {
        return "\(s.tracks[i].name) has no \(want)"
      }
      s.tracks[i].effects.remove(at: at)
      return "\(want) removed from \(s.tracks[i].name)"
    }
    rebuildIfPlaying()
    return result
  }

  func duplicate(track: String) -> String {
    let result = mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      let source = s.tracks[i]
      let copy = Track(
        id: s.nextID, name: "\(source.name) copy", kind: source.kind, volume: source.volume,
        pan: -source.pan, muted: source.muted, solo: false, effects: source.effects)
      s.nextID += 1
      s.tracks.insert(copy, at: i + 1)
      return "duplicated \(s.tracks[i].name) as track \(i + 2) \"\(copy.name)\" (panned \(Self.panText(copy.pan)) to widen it)"
    }
    rebuildIfPlaying()
    return result
  }

  func delete(track: String) -> String {
    let result = mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      guard s.tracks.count > 1 else { return "cannot delete the only track" }
      let name = s.tracks[i].name
      s.tracks.remove(at: i)
      return "deleted \(name); \(s.tracks.count) tracks left"
    }
    rebuildIfPlaying()
    return result
  }

  func rename(track: String, to name: String) -> String {
    mutate { s in
      guard let i = index(of: track, in: s) else { return noTrack(track, s) }
      let was = s.tracks[i].name
      s.tracks[i].name = name.trimmingCharacters(in: .whitespaces)
      return "renamed \(was) to \(s.tracks[i].name)"
    }
  }

  func setTempo(_ bpm: Int) -> String {
    let result = mutate { s in
      s.tempo = min(200, max(60, bpm))
      return "tempo \(s.tempo) bpm — the song is now \(Self.f(s.duration)) s"
    }
    rebuildIfPlaying()
    return result
  }

  func fade(_ which: String, seconds: Double) -> String {
    mutate { s in
      let length = min(max(0.5, seconds), s.duration / 2)
      switch which.lowercased() {
      case "in": s.fadeIn = length
      case "out": s.fadeOut = length
      default:
        s.fadeIn = length
        s.fadeOut = length
      }
      return "fade \(which.lowercased()) over \(Self.f(length)) s"
    }
  }

  func revert() -> String {
    stop()
    return mutate { s in
      s = original
      return "back to the original mix — \(s.tracks.count) tracks at \(s.tempo) bpm"
    }
  }

  // MARK: Playback

  func play(from seconds: Double = 0) -> String {
    stop()
    let s = sync { state }
    let start = min(max(0, seconds), max(0, s.duration - 0.5))
    do {
      let engine = AVAudioEngine()
      let players = try buildGraph(in: engine, state: s)
      try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
      try AVAudioSession.sharedInstance().setActive(true)
      try engine.start()
      // Every player loops its two-bar buffer; the loop is the same length
      // on every track, so they stay in step. Starting mid-song is a
      // matter of scheduling from the right offset once, then looping.
      for (player, buffer) in players {
        let loop = Double(buffer.frameLength) / buffer.format.sampleRate
        let offset = start.truncatingRemainder(dividingBy: loop)
        let frame = AVAudioFrameCount(offset * buffer.format.sampleRate)
        if frame > 0, frame < buffer.frameLength, let tail = Self.tail(of: buffer, from: frame) {
          player.scheduleBuffer(tail, at: nil, options: [])
        }
        player.scheduleBuffer(buffer, at: nil, options: .loops)
        player.play()
      }
      sync {
        self.engine = engine
        state.playing = true
        state.playhead = start
        startedAt = Date().addingTimeInterval(-start)
      }
      applyLevels(sync { state })
      let job = Task.detached(priority: .utility) { [weak self] in
        while !Task.isCancelled {
          try? await Task.sleep(for: .milliseconds(200))
          guard let self, let began = self.sync({ self.startedAt }) else { return }
          let now = Date().timeIntervalSince(began)
          let (duration, fadeIn, fadeOut) = self.sync {
            self.state.playhead = min(now, self.state.duration)
            return (self.state.duration, self.state.fadeIn, self.state.fadeOut)
          }
          // The song-level fades ride on the main mixer while playing.
          var gain = 1.0
          if fadeIn > 0, now < fadeIn { gain = min(gain, now / fadeIn) }
          if fadeOut > 0, now > duration - fadeOut { gain = min(gain, max(0, (duration - now) / fadeOut)) }
          self.sync { self.engine?.mainMixerNode.outputVolume = Float(gain) }
          if now >= duration {
            self.stop()
            return
          }
        }
      }
      let previous = sync { let old = playbackJob; playbackJob = job; return old }
      previous?.cancel()
      postMixer(sync { state })
      return "playing from \(Self.f(start)) s (\(Self.f(s.duration)) s song, \(s.tempo) bpm)"
    } catch {
      return "could not start playback: \(error.localizedDescription)"
    }
  }

  /// The rest of a loop from a frame: what plays first when the song starts
  /// mid-bar, before the whole loop takes over.
  private static func tail(of buffer: AVAudioPCMBuffer, from frame: AVAudioFrameCount) -> AVAudioPCMBuffer? {
    let count = buffer.frameLength - frame
    guard let out = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: count),
      let src = buffer.floatChannelData, let dst = out.floatChannelData
    else { return nil }
    out.frameLength = count
    for channel in 0..<Int(buffer.format.channelCount) {
      dst[channel].update(from: src[channel] + Int(frame), count: Int(count))
    }
    return out
  }

  @discardableResult
  func stop() -> String {
    let (engine, job, wasPlaying, at) = sync {
      let out = (self.engine, playbackJob, state.playing, state.playhead)
      self.engine = nil
      playbackJob = nil
      trackMixers = [:]
      startedAt = nil
      state.playing = false
      return out
    }
    job?.cancel()
    engine?.stop()
    postMixer(sync { state })
    return wasPlaying ? "stopped at \(Self.f(at)) s" : "already stopped"
  }

  /// A structural change (effects, tracks, tempo) while playing: rebuild the
  /// graph from where the song is.
  private func rebuildIfPlaying() {
    let (playing, at) = sync { (state.playing, state.playhead) }
    guard playing else { return }
    _ = play(from: at)
  }

  /// Volume, pan, mute and solo apply live to the per-track mixers.
  private func applyLevels(_ s: SongState) {
    let anySolo = s.tracks.contains(where: \.solo)
    sync {
      for track in s.tracks {
        guard let mixer = trackMixers[track.id] else { continue }
        let audible = !track.muted && (!anySolo || track.solo)
        mixer.outputVolume = audible ? Float(track.volume) / 100 : 0
        mixer.pan = Float(track.pan) / 100
      }
    }
  }

  private func postMixer(_ s: SongState) {
    let anySolo = s.tracks.contains(where: \.solo)
    ArtifactBox.shared.post(.table(
      title: "\(s.tracks.count) tracks · \(s.tempo) bpm · \(s.playing ? "playing" : "stopped")",
      columns: ["Track", "Vol", "Pan", "FX", ""],
      rows: s.tracks.map { t in
        let audible = !t.muted && (!anySolo || t.solo)
        return [
          t.name, String(t.volume), Self.panText(t.pan),
          t.effects.isEmpty ? "—" : t.effects.joined(separator: "+"),
          t.solo ? "SOLO" : (t.muted ? "muted" : (audible ? "" : "silent")),
        ]
      }))
  }

  // MARK: The graph

  /// player → effects → per-track mixer → main mixer, one chain per track.
  /// Returns each player with its loop buffer so the caller can start them.
  private func buildGraph(in engine: AVAudioEngine, state s: SongState) throws
    -> [(AVAudioPlayerNode, AVAudioPCMBuffer)]
  {
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
    var players: [(AVAudioPlayerNode, AVAudioPCMBuffer)] = []
    var mixers: [Int: AVAudioMixerNode] = [:]
    for track in s.tracks {
      let buffer = try Synth.loop(track.kind, tempo: s.tempo, format: format)
      let player = AVAudioPlayerNode()
      let mixer = AVAudioMixerNode()
      engine.attach(player)
      engine.attach(mixer)
      var upstream: AVAudioNode = player
      for effect in track.effects {
        let node = Self.effectNode(effect)
        engine.attach(node)
        engine.connect(upstream, to: node, format: format)
        upstream = node
      }
      engine.connect(upstream, to: mixer, format: format)
      engine.connect(mixer, to: engine.mainMixerNode, format: nil)
      mixers[track.id] = mixer
      players.append((player, buffer))
    }
    sync { trackMixers = mixers }
    return players
  }

  private static func effectNode(_ effect: String) -> AVAudioNode {
    switch effect {
    case "reverb":
      let node = AVAudioUnitReverb()
      node.loadFactoryPreset(.mediumHall)
      node.wetDryMix = 40
      return node
    case "echo":
      let node = AVAudioUnitDelay()
      node.delayTime = 0.3
      node.feedback = 35
      node.wetDryMix = 35
      return node
    case "distortion":
      let node = AVAudioUnitDistortion()
      node.loadFactoryPreset(.multiDistortedSquared)
      node.wetDryMix = 40
      return node
    default:  // lowpass
      let node = AVAudioUnitEQ(numberOfBands: 1)
      node.bands[0].filterType = .lowPass
      node.bands[0].frequency = 600
      node.bands[0].bypass = false
      return node
    }
  }

  // MARK: Export

  /// The whole song, offline, to an .m4a in Documents — the same graph the
  /// speaker plays, rendered faster than real time.
  func export() async throws -> String {
    stop()
    let s = sync { state }
    let engine = AVAudioEngine()
    let players = try buildGraph(in: engine, state: s)
    let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!
    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
    try engine.start()
    for (player, buffer) in players {
      // The completion-handler form: in an async context the bare call
      // resolves to the awaiting overload, which would wait for a loop that
      // never ends.
      player.scheduleBuffer(buffer, at: nil, options: .loops, completionHandler: nil)
      player.play()
    }
    applyLevels(s)
    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("mix-\(Int(Date().timeIntervalSince1970)).m4a")
    let file = try AVAudioFile(
      forWriting: url,
      settings: [
        AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 44100, AVNumberOfChannelsKey: 2,
        AVEncoderBitRateKey: 192_000,
      ])
    let total = AVAudioFramePosition(s.duration * format.sampleRate)
    let chunk = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4096)!
    var rendered: AVAudioFramePosition = 0
    while rendered < total {
      let want = AVAudioFrameCount(min(4096, total - rendered))
      let t = Double(rendered) / format.sampleRate
      var gain = 1.0
      if s.fadeIn > 0, t < s.fadeIn { gain = min(gain, t / s.fadeIn) }
      if s.fadeOut > 0, t > s.duration - s.fadeOut { gain = min(gain, max(0, (s.duration - t) / s.fadeOut)) }
      engine.mainMixerNode.outputVolume = Float(gain)
      let status = try engine.renderOffline(want, to: chunk)
      guard status == .success else { break }
      try file.write(from: chunk)
      rendered += AVAudioFramePosition(chunk.frameLength)
    }
    engine.stop()
    sync { trackMixers = [:] }
    return "exported \(Self.f(s.duration)) s of \(s.tracks.count) tracks to \(url.lastPathComponent) in the app's Documents"
  }
}

/// A tiny synthesizer: two bars of a I–vi–IV–V loop per instrument, at the
/// song's tempo. Nothing here is music worth keeping; it is enough that
/// muting the drums, panning the keys or an echo on the lead is something
/// a viewer hears change.
enum Synth {
  private static func hz(_ midi: Int) -> Double { 440 * pow(2, Double(midi - 69) / 12) }

  /// Chord roots (MIDI) for C, Am, F, G — two beats each across two bars.
  private static let roots = [60, 57, 53, 55]

  static func loop(_ kind: AudioBox.Kind, tempo: Int, format: AVAudioFormat) throws -> AVAudioPCMBuffer {
    let rate = format.sampleRate
    let beat = 60.0 / Double(tempo)
    let frames = Int(beat * 8 * rate)
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)) else {
      throw NSError(domain: "Synth", code: 1)
    }
    buffer.frameLength = AVAudioFrameCount(frames)
    let out = buffer.floatChannelData![0]
    for i in 0..<frames { out[i] = 0 }

    func add(freq: Double, start: Double, length: Double, gain: Double, timbre: (Double) -> Double, decay: Double) {
      let from = Int(start * rate)
      let count = Int(length * rate)
      for n in 0..<count where from + n < frames {
        let t = Double(n) / rate
        let env = exp(-t * decay) * min(1, t * 200)  // fast attack, exponential release
        out[from + n] += Float(gain * env * timbre(2 * .pi * freq * t))
      }
    }
    func noise(start: Double, length: Double, gain: Double, decay: Double) {
      let from = Int(start * rate)
      let count = Int(length * rate)
      var seed: UInt32 = 12345
      for n in 0..<count where from + n < frames {
        seed = seed &* 1_664_525 &+ 1_013_904_223
        let white = Double(seed) / Double(UInt32.max) * 2 - 1
        let t = Double(n) / rate
        out[from + n] += Float(gain * exp(-t * decay) * white)
      }
    }
    let saw: (Double) -> Double = { x in
      // Three harmonics of a sawtooth: bright enough to hear a lowpass work.
      sin(x) + sin(2 * x) / 2 + sin(3 * x) / 3
    }
    let soft: (Double) -> Double = { x in sin(x) + sin(2 * x) * 0.15 }
    let square: (Double) -> Double = { x in sin(x) + sin(3 * x) / 3 + sin(5 * x) / 5 }

    switch kind {
    case .drums:
      for b in 0..<8 {
        let t = Double(b) * beat
        if b % 2 == 0 {  // kick on 1 and 3: a sine sweeping down
          let from = Int(t * rate)
          let count = Int(0.18 * rate)
          for n in 0..<count where from + n < frames {
            let tt = Double(n) / rate
            let f = 40 + 110 * exp(-tt * 30)
            out[from + n] += Float(0.9 * exp(-tt * 14) * sin(2 * .pi * f * tt))
          }
        } else {  // snare on 2 and 4
          noise(start: t, length: 0.15, gain: 0.5, decay: 22)
        }
        noise(start: t, length: 0.04, gain: 0.18, decay: 90)  // hat on the beat
        noise(start: t + beat / 2, length: 0.03, gain: 0.12, decay: 110)  // and the off-beat
      }
    case .bass:
      for (chord, root) in roots.enumerated() {
        for eighth in 0..<4 {
          let t = (Double(chord) * 2 + Double(eighth) / 2) * beat
          let note = eighth == 3 ? root - 5 : root  // a fifth below on the last eighth
          add(freq: hz(note - 24), start: t, length: beat / 2, gain: 0.5, timbre: saw, decay: 3)
        }
      }
    case .keys:
      for (chord, root) in roots.enumerated() {
        let third = chord == 1 ? 3 : 4  // Am is minor
        for interval in [0, third, 7] {
          add(freq: hz(root + interval), start: Double(chord) * 2 * beat, length: 2 * beat, gain: 0.16, timbre: soft, decay: 0.8)
        }
      }
    case .lead:
      let melody = [[4, 7, 12, 7], [3, 7, 12, 7], [4, 7, 12, 7], [4, 7, 11, 7]]
      for (chord, root) in roots.enumerated() {
        for (q, interval) in melody[chord].enumerated() {
          let t = (Double(chord) * 2 + Double(q) / 2) * beat
          add(freq: hz(root + 12 + interval), start: t, length: beat / 2, gain: 0.25, timbre: square, decay: 5)
        }
      }
    }
    // Keep peaks under 1.0 whatever stacked.
    var peak: Float = 0
    for i in 0..<frames { peak = max(peak, abs(out[i])) }
    if peak > 0.95 { for i in 0..<frames { out[i] *= 0.95 / peak } }
    return buffer
  }
}

// MARK: - Tools (the mixer's controls, in its words)

@available(iOS 27.0, *)
struct TrackVolumeTool: Tool {
  let name = "set_track_volume"
  let description = "Set a track's volume fader."
  @Generable struct Arguments {
    @Guide(description: "The track, by name (Drums, Bass, Keys, Lead) or number.") var track: String
    @Guide(description: "New level 0–100. The current level is in the song state; 'a bit quieter' is about 15 less.")
    var percent: Int
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.setVolume(track: arguments.track, percent: arguments.percent)
  }
}

@available(iOS 27.0, *)
struct TrackPanTool: Tool {
  let name = "set_track_pan"
  let description = "Pan a track left or right."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
    @Guide(description: "-100 (hard left) to 100 (hard right); 0 is centre, -40 is a little left.") var pan: Int
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.setPan(track: arguments.track, pan: arguments.pan)
  }
}

@available(iOS 27.0, *)
struct MuteTrackTool: Tool {
  let name = "mute_track"
  let description = "Mute or unmute a track."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
    @Guide(description: "true to mute, false to unmute.") var muted: Bool
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.mute(track: arguments.track, on: arguments.muted)
  }
}

@available(iOS 27.0, *)
struct SoloTrackTool: Tool {
  let name = "solo_track"
  let description = "Solo a track (hear only it), or take it out of solo."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
    @Guide(description: "true to solo, false to un-solo.") var solo: Bool
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.solo(track: arguments.track, on: arguments.solo)
  }
}

@available(iOS 27.0, *)
struct AddEffectTool: Tool {
  let name = "add_effect"
  let description = "Put an effect on a track."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
    @Guide(description: "Which effect.", .anyOf(AudioBox.effects)) var effect: String
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.addEffect(track: arguments.track, effect: arguments.effect)
  }
}

@available(iOS 27.0, *)
struct RemoveEffectTool: Tool {
  let name = "remove_effect"
  let description = "Take an effect off a track."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
    @Guide(description: "Which effect.", .anyOf(AudioBox.effects)) var effect: String
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.removeEffect(track: arguments.track, effect: arguments.effect)
  }
}

@available(iOS 27.0, *)
struct DuplicateTrackTool: Tool {
  let name = "duplicate_track"
  let description = "Duplicate a track (the copy is panned opposite, to widen the sound)."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.duplicate(track: arguments.track)
  }
}

@available(iOS 27.0, *)
struct DeleteTrackTool: Tool {
  let name = "delete_track"
  let description = "Remove a track from the song."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.delete(track: arguments.track)
  }
}

@available(iOS 27.0, *)
struct RenameTrackTool: Tool {
  let name = "rename_track"
  let description = "Rename a track."
  @Generable struct Arguments {
    @Guide(description: "The track, by name or number.") var track: String
    @Guide(description: "The new name.") var name: String
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.rename(track: arguments.track, to: arguments.name)
  }
}

@available(iOS 27.0, *)
struct SetTempoTool: Tool {
  let name = "set_tempo"
  let description = "Change the song's tempo."
  @Generable struct Arguments {
    @Guide(description: "Beats per minute, 60–200. The current tempo is in the song state.") var bpm: Int
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.setTempo(arguments.bpm)
  }
}

@available(iOS 27.0, *)
struct SongFadeTool: Tool {
  let name = "add_fade"
  let description = "Fade the whole song in at the start, out at the end, or both."
  @Generable struct Arguments {
    @Guide(description: "Which end.", .anyOf(["in", "out", "both"])) var which: String
    @Guide(description: "Length of the fade in seconds. 2 is typical.") var seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.fade(arguments.which, seconds: arguments.seconds)
  }
}

@available(iOS 27.0, *)
struct PlaySongTool: Tool {
  let name = "play"
  let description = "Play the song from a point in time."
  @Generable struct Arguments {
    @Guide(description: "Where to start, in seconds. 0 is the top.") var from_seconds: Double
  }
  func call(arguments: Arguments) async throws -> String {
    AudioBox.shared.play(from: arguments.from_seconds)
  }
}

@available(iOS 27.0, *)
struct StopSongTool: Tool {
  let name = "stop"
  let description = "Stop playback."
  func call(arguments: NoArguments) async throws -> String {
    AudioBox.shared.stop()
  }
}

@available(iOS 27.0, *)
struct ExportSongTool: Tool {
  let name = "export_song"
  let description = "Mix the song down to an audio file."
  func call(arguments: NoArguments) async throws -> String {
    try await AudioBox.shared.export()
  }
}

@available(iOS 27.0, *)
struct RevertSongTool: Tool {
  let name = "revert_to_original"
  let description = "Throw away all changes and go back to the original mix."
  func call(arguments: NoArguments) async throws -> String {
    AudioBox.shared.revert()
  }
}

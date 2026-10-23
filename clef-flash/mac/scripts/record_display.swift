// record_display.swift: record the main display, showing only the windows of the apps named here, to a silent H.264
// mp4 at a constant frame rate, with the time of every frame; or take one screenshot through the same filter.
// ScreenCaptureKit's display filter excludes every other app that is running when it starts, so their windows, their
// menu bar items and their notifications are not in the picture (what is behind them is drawn instead); an app that
// starts later (a draft's TextEdit, opened during a scene) is in it. No cursor.
//
//   xcrun swiftc -O -o out/record_display scripts/record_display.swift      (take.sh does this)
//   out/record_display --keep <pid,...> --keep-bundles <id,...> --shot <out.png>
//   out/record_display --keep <pid,...> --keep-bundles <id,...> <out.mp4> <frames.jsonl> <stop file> [fps 30] [max s 50]
//
// The display is captured at 1920 x 1080 (scaled from its native pixels), 60 captures per second at most. Output frame
// k (presentation time k / fps) holds the newest complete capture at its tick; the clock starts at the first capture.
// frames.jsonl, one line per output frame:
//   {"i": k, "pts": k / fps, "tick_epoch": the tick's time, "src_epoch": the capture's display time, "src_seq": n}
// (epoch seconds, the clock of the app's time.time(); the display time is ScreenCaptureKit's mach time + one offset).
// Stops when the stop file appears, on SIGINT / SIGTERM, or after max seconds; then prints one JSON summary line,
// which lists the apps kept and the apps left out (by name). stdout "RECORDING <epoch>" once the first frame is
// written. Exit 2 usage, 3 capture refused (TCC), 4 no display, 5 the stream failed, 6 the writer failed. Never
// overwrites its output.
import AVFoundation
import AppKit
import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit
import UniformTypeIdentifiers

setvbuf(stdout, nil, _IOLBF, 0)
// A command-line tool has no window-server connection until AppKit makes one; ScreenCaptureKit asserts without it
// (CGS_REQUIRE_INIT). Prohibited policy: no Dock icon, no menu bar, no windows of its own.
NSApplication.shared.setActivationPolicy(.prohibited)

let width = 1920
let height = 1080
var keepPids: Set<Int32> = []
var keepBundles: Set<String> = []
var shotPath: String? = nil
var rest: [String] = []
var argi = 1
let argv = CommandLine.arguments
while argi < argv.count {
  switch argv[argi] {
  case "--keep":
    argi += 1
    keepPids = Set((argi < argv.count ? argv[argi] : "").split(separator: ",").compactMap { Int32($0) })
  case "--keep-bundles":
    argi += 1
    keepBundles = Set((argi < argv.count ? argv[argi] : "").split(separator: ",").map(String.init))
  case "--shot":
    argi += 1
    shotPath = argi < argv.count ? argv[argi] : nil
  default:
    rest.append(argv[argi])
  }
  argi += 1
}
func usage() -> Never {
  fputs("usage: record_display --keep <pids> --keep-bundles <ids> (--shot <png> | <out.mp4> <frames.jsonl> <stop file> [fps] [max s])\n", stderr)
  exit(2)
}
if shotPath == nil && rest.count < 3 { usage() }
let outURL = URL(fileURLWithPath: shotPath ?? rest[0])
if FileManager.default.fileExists(atPath: outURL.path) {
  fputs("\(outURL.path) exists: not overwritten\n", stderr)
  exit(2)
}
let framesPath = shotPath == nil ? rest[1] : ""
let stopPath = shotPath == nil ? rest[2] : ""
let fps = rest.count > 3 ? (Int32(rest[3]) ?? 30) : 30
let maxSeconds = rest.count > 4 ? (Double(rest[4]) ?? 50) : 50

var timebase = mach_timebase_info_data_t()
mach_timebase_info(&timebase)
func machSeconds(_ t: UInt64) -> Double { Double(t) * Double(timebase.numer) / Double(timebase.denom) / 1e9 }
let epochOffset = Date().timeIntervalSince1970 - machSeconds(mach_absolute_time())

func streamConfig() -> SCStreamConfiguration {
  let cfg = SCStreamConfiguration()
  cfg.width = width
  cfg.height = height
  cfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)
  cfg.pixelFormat = kCVPixelFormatType_32BGRA
  cfg.colorSpaceName = CGColorSpace.sRGB
  cfg.showsCursor = false
  cfg.capturesAudio = false
  cfg.queueDepth = 8
  cfg.scalesToFit = true
  return cfg
}

final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate {
  let captureQueue = DispatchQueue(label: "record.capture")
  let tickQueue = DispatchQueue(label: "record.ticks")
  let lock = NSLock()
  var stream: SCStream?
  var writer: AVAssetWriter?
  var input: AVAssetWriterInput?
  var adaptor: AVAssetWriterInputPixelBufferAdaptor?
  var timer: DispatchSourceTimer?
  var stopWatch: DispatchSourceTimer?
  var frames: FileHandle?
  var latest: CVPixelBuffer?
  var latestEpoch = 0.0
  var latestSeq = 0
  var captures = 0
  var written: Int64 = 0
  var skipped = 0
  var started = false
  var finishing = false
  var firstEpoch = 0.0
  var info: [String: Any] = [:]

  func start(filter: SCContentFilter) {
    let cfg = streamConfig()
    do {
      let w = try AVAssetWriter(outputURL: outURL, fileType: .mp4)
      let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        AVVideoColorPropertiesKey: [
          AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
          AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
          AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2],
        AVVideoCompressionPropertiesKey: [
          AVVideoAverageBitRateKey: 16_000_000, AVVideoExpectedSourceFrameRateKey: Int(fps),
          AVVideoMaxKeyFrameIntervalKey: Int(fps), AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
      ]
      let inp = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
      inp.expectsMediaDataInRealTime = true
      let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp, sourcePixelBufferAttributes: nil)
      guard w.canAdd(inp) else { fail(6, "the writer refused the video input") }
      w.add(inp)
      writer = w
      input = inp
      adaptor = ad
    } catch {
      fail(6, "writer: \(error)")
    }
    FileManager.default.createFile(atPath: framesPath, contents: nil)
    frames = FileHandle(forWritingAtPath: framesPath)
    let s = SCStream(filter: filter, configuration: cfg, delegate: self)
    do {
      try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
    } catch {
      fail(5, "stream output: \(error)")
    }
    stream = s
    s.startCapture { error in
      if let error = error {
        let refused = (error as NSError).code == SCStreamError.Code.userDeclined.rawValue
        self.fail(refused ? 3 : 5, "startCapture: \(error)")
      }
    }
    tickQueue.asyncAfter(deadline: .now() + maxSeconds) { self.finish("max seconds") }
    let watch = DispatchSource.makeTimerSource(queue: tickQueue)
    watch.schedule(deadline: .now() + 0.05, repeating: 0.05)
    watch.setEventHandler { if FileManager.default.fileExists(atPath: stopPath) { self.finish("stop file") } }
    watch.resume()
    stopWatch = watch
  }

  func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
    guard type == .screen, sb.isValid,
      let atts = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
      let a = atts.first, let raw = a[.status] as? Int, let status = SCFrameStatus(rawValue: raw),
      status == .complete, let pb = CMSampleBufferGetImageBuffer(sb)
    else { return }
    let shown = (a[.displayTime] as? UInt64).map { machSeconds($0) + epochOffset } ?? Date().timeIntervalSince1970
    lock.lock()
    latest = pb
    latestEpoch = shown
    latestSeq += 1
    captures += 1
    let first = !started
    started = true
    lock.unlock()
    if first { tickQueue.async { self.startTicks() } }
  }

  func stream(_ stream: SCStream, didStopWithError error: Error) {
    fail(5, "the stream stopped: \(error)")
  }

  func startTicks() {
    guard let w = writer, w.startWriting() else { fail(6, "startWriting: \(String(describing: writer?.error))") }
    w.startSession(atSourceTime: .zero)
    firstEpoch = Date().timeIntervalSince1970
    let t = DispatchSource.makeTimerSource(flags: .strict, queue: tickQueue)
    t.schedule(deadline: .now(), repeating: 1.0 / Double(fps), leeway: .microseconds(500))
    t.setEventHandler { self.tick() }
    timer = t
    t.resume()
  }

  func tick() {
    guard !finishing, let inp = input, let ad = adaptor else { return }
    lock.lock()
    let pb = latest
    let src = latestEpoch
    let seq = latestSeq
    lock.unlock()
    guard let buffer = pb else { return }
    let k = written + Int64(skipped)
    let now = Date().timeIntervalSince1970
    if !inp.isReadyForMoreMediaData {
      skipped += 1
      return
    }
    if !ad.append(buffer, withPresentationTime: CMTime(value: k, timescale: fps)) {
      fail(6, "append: \(String(describing: writer?.error))")
    }
    written += 1
    let line = String(format: "{\"i\": %lld, \"pts\": %.6f, \"tick_epoch\": %.6f, \"src_epoch\": %.6f, \"src_seq\": %d}\n",
                      k, Double(k) / Double(fps), now, src, seq)
    frames?.write(line.data(using: .utf8)!)
    if written == 1 { print(String(format: "RECORDING %.6f", now)) }
  }

  func finish(_ why: String) {
    if finishing { return }
    finishing = true
    timer?.cancel()
    stopWatch?.cancel()
    let done = DispatchSemaphore(value: 0)
    if let s = stream {
      s.stopCapture { _ in done.signal() }
      _ = done.wait(timeout: .now() + 5)
    }
    try? frames?.close()
    guard let w = writer, w.status == .writing else {
      summary(why, ok: false, note: "no frame was written (status \(writer?.status.rawValue ?? -1))")
      exit(5)
    }
    input?.markAsFinished()
    let end = CMTime(value: written + Int64(skipped), timescale: fps)
    w.endSession(atSourceTime: end)
    w.finishWriting {
      let ok = w.status == .completed
      self.summary(why, ok: ok, note: ok ? "" : "writer: \(String(describing: w.error))")
      exit(ok ? 0 : 6)
    }
  }

  func summary(_ why: String, ok: Bool, note: String) {
    var s = info
    s["stopped_by"] = why
    s["ok"] = ok
    s["out"] = outURL.path
    s["frames_written"] = written
    s["ticks_skipped"] = skipped
    s["captures"] = captures
    s["first_frame_epoch"] = firstEpoch
    s["end_epoch"] = Date().timeIntervalSince1970
    if !note.isEmpty { s["note"] = note }
    if let data = try? JSONSerialization.data(withJSONObject: s, options: [.sortedKeys]),
      let text = String(data: data, encoding: .utf8)
    {
      print("SUMMARY \(text)")
    }
  }

  func fail(_ code: Int32, _ message: String) -> Never {
    fputs("record_display: \(message)\n", stderr)
    exit(code)
  }
}

let recorder = Recorder()
signal(SIGINT, SIG_IGN)
signal(SIGTERM, SIG_IGN)
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM] {
  let src = DispatchSource.makeSignalSource(signal: sig, queue: recorder.tickQueue)
  src.setEventHandler { recorder.finish("signal \(sig)") }
  src.resume()
  signalSources.append(src)
}

func writePNG(_ image: CGImage, _ url: URL) -> Bool {
  guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
    return false
  }
  CGImageDestinationAddImage(dest, image, nil)
  return CGImageDestinationFinalize(dest)
}

SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { content, error in
  if let error = error {
    let refused = (error as NSError).code == SCStreamError.Code.userDeclined.rawValue
    fputs("record_display: shareable content: \(error)\n", stderr)
    exit(refused ? 3 : 5)
  }
  guard let content = content else { exit(5) }
  let mainID = CGMainDisplayID()
  guard let display = content.displays.first(where: { $0.displayID == mainID }) ?? content.displays.first else {
    fputs("record_display: no display\n", stderr)
    exit(4)
  }
  var kept: [String] = []
  var leftOut: [String] = []
  let excluded = content.applications.filter { app in
    let keep = keepPids.contains(app.processID) || keepBundles.contains(app.bundleIdentifier)
    let name = app.applicationName.isEmpty ? app.bundleIdentifier : app.applicationName
    if keep { kept.append(name) } else { leftOut.append(name) }
    return !keep
  }
  let filter = SCContentFilter(display: display, excludingApplications: excluded, exceptingWindows: [])
  recorder.info = ["display_id": Int(display.displayID), "display_pt": [display.width, display.height],
                   "px": [width, height], "fps": Int(fps), "kept": kept.sorted(), "left_out": leftOut.count]
  if let shot = shotPath {
    SCScreenshotManager.captureImage(contentFilter: filter, configuration: streamConfig()) { image, error in
      guard let image = image, error == nil else {
        fputs("record_display: screenshot: \(String(describing: error))\n", stderr)
        exit(5)
      }
      let ok = writePNG(image, URL(fileURLWithPath: shot))
      var s = recorder.info
      s["shot"] = shot
      s["ok"] = ok
      s["image_px"] = [image.width, image.height]
      if let data = try? JSONSerialization.data(withJSONObject: s, options: [.sortedKeys]),
        let text = String(data: data, encoding: .utf8)
      {
        print("SHOT \(text)")
      }
      exit(ok ? 0 : 6)
    }
    return
  }
  recorder.start(filter: filter)
}
dispatchMain()

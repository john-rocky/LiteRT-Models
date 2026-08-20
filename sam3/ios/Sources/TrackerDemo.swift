import SwiftUI
import UIKit

/// Hands-free video-tracking showcase: when Documents/trackerdemo/demo.json exists
/// (plus the tracker payload under Documents/tracker and frames under
/// Documents/trackerdemo/frames), the app compiles the tracker, types the prompt,
/// tracks it through the clip, then loops the composited result — no touches
/// needed, ideal for a screen recording. Delete Documents/trackerdemo to disable.
///
/// The tracker's hotstart postprocess finalizes per-frame outputs only at the end
/// of the run, so the live phase shows the raw clip advancing with progress chips;
/// the composited overlay playback starts once tracking finishes.
enum TrackerDemo {
    struct Config: Decodable {
        let prompt: String
        let fps: Double?
        let startDelay: Double?
    }

    static func config(modelsRoot: URL) -> Config? {
        let url = modelsRoot.appendingPathComponent("trackerdemo/demo.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(Config.self, from: data)
    }

    static func frameURLs(modelsRoot: URL) -> [URL] {
        let dir = modelsRoot.appendingPathComponent("trackerdemo/frames")
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        return files
            .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
            .sorted {
                (Int($0.deletingPathExtension().lastPathComponent) ?? 0)
                    < (Int($1.deletingPathExtension().lastPathComponent) ?? 0)
            }
    }

    /// Source frame + every tracked object's video-res mask tinted in its stable
    /// per-id color (one RGBA layer for all objects — the tracker's outputs are
    /// pixel-disjoint after the object-wise non-overlap pass).
    static func composite(frameURL: URL, result: Sam3Tracker.FrameResult,
                          vW: Int, vH: Int) -> UIImage? {
        guard let base = UIImage(contentsOfFile: frameURL.path) else { return nil }
        guard !result.ids.isEmpty else { return base }
        let px = vW * vH
        var layer = [UInt8](repeating: 0, count: px * 4)
        for (i, id) in result.ids.enumerated() {
            let color = OverlayRenderer.palette[id % OverlayRenderer.palette.count]
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            color.getRed(&r, green: &g, blue: &b, alpha: &a)
            let alpha: CGFloat = 0.45
            let pr = UInt8(r * alpha * 255), pg = UInt8(g * alpha * 255)
            let pb = UInt8(b * alpha * 255), pa = UInt8(alpha * 255)
            let words = result.masks[i]
            for w in 0..<words.count {
                var bits = words[w]
                if bits == 0 { continue }
                while bits != 0 {
                    let t = bits.trailingZeroBitCount
                    let p = (w << 6) | t
                    if p < px {
                        layer[p * 4] = pr
                        layer[p * 4 + 1] = pg
                        layer[p * 4 + 2] = pb
                        layer[p * 4 + 3] = pa
                    }
                    bits &= bits - 1
                }
            }
        }
        let data = Data(layer)
        guard let provider = CGDataProvider(data: data as CFData),
              let overlay = CGImage(
                width: vW, height: vH, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: vW * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true,
                intent: .defaultIntent)
        else { return base }
        let size = base.size
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            base.draw(in: CGRect(origin: .zero, size: size))
            ctx.cgContext.interpolationQuality = .medium
            ctx.cgContext.draw(overlay, in: CGRect(origin: .zero, size: size))
        }
    }
}

// MARK: - Demo model (drives the view; owned by Sam3ViewModel.State.trackerDemo)

@MainActor
final class TrackerDemoModel: ObservableObject {
    enum Phase {
        case intro
        case tracking
        case rendering
        case playback
    }

    @Published var phase: Phase = .intro
    @Published var frame: UIImage?
    @Published var typedPrompt = ""
    @Published var progressChip: String?
    @Published var speedChip: String?
    @Published var trackedChip: String?
    @Published var errorLine: String?

    private var driver: Task<Void, Never>?

    func begin(tracker: Sam3Tracker, modelsRoot: URL, config: TrackerDemo.Config) {
        let frames = TrackerDemo.frameURLs(modelsRoot: modelsRoot)
        guard !frames.isEmpty else {
            errorLine = "no frames under Documents/trackerdemo/frames"
            return
        }
        let clipDir = frames[0].deletingLastPathComponent()
        driver = Task { @MainActor [weak self] in
            // Intro: first frame + typed prompt.
            try? await Task.sleep(nanoseconds: UInt64((config.startDelay ?? 2.0) * 1e9))
            self?.frame = UIImage(contentsOfFile: frames[0].path)
            try? await Task.sleep(nanoseconds: 700_000_000)
            for ch in config.prompt {
                self?.typedPrompt.append(ch)
                try? await Task.sleep(nanoseconds: UInt64.random(in: 55_000_000...120_000_000))
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            self?.phase = .tracking

            // Tracking: run on a background task; progress lines are "f<i> <ms>ms …".
            let t0 = Date()
            let outcome: Result<[Int: Sam3Tracker.FrameResult], Error> =
                await Task.detached(priority: .userInitiated) {
                    do {
                        return .success(try tracker.track(clipDir: clipDir) { line in
                            guard let fStr = line.split(separator: " ").first,
                                  let fi = Int(fStr.dropFirst()) else { return }
                            Task { @MainActor [weak self] in
                                guard let self else { return }
                                self.frame = UIImage(contentsOfFile: frames[fi].path)
                                self.progressChip = "tracking · frame \(fi + 1)/\(frames.count)"
                                let sPerFrame = Date().timeIntervalSince(t0) / Double(fi + 1)
                                self.speedChip = String(format: "%.1f s/frame · GPU", sPerFrame)
                            }
                        })
                    } catch {
                        return .failure(error)
                    }
                }.value

            guard let self else { return }
            switch outcome {
            case .failure(let error):
                self.errorLine = "\(error)"
            case .success(let results):
                self.phase = .rendering
                self.progressChip = "rendering overlays…"
                let composited: [UIImage] = await Task.detached(priority: .userInitiated) {
                    let vW = tracker.vW
                    let vH = tracker.vH
                    return frames.indices.compactMap { fi in
                        guard let r = results[fi] else {
                            return UIImage(contentsOfFile: frames[fi].path)
                        }
                        return TrackerDemo.composite(
                            frameURL: frames[fi], result: r, vW: vW, vH: vH)
                    }
                }.value
                guard !composited.isEmpty else {
                    self.errorLine = "no frames rendered"
                    return
                }
                let tracked = Set(results.values.flatMap { $0.ids }).count
                self.trackedChip = "\(config.prompt) · \(tracked) tracked"
                self.progressChip = nil
                self.phase = .playback
                let dt = UInt64(1e9 / max(config.fps ?? 7.5, 1))
                var i = 0
                while !Task.isCancelled {
                    self.frame = composited[i % composited.count]
                    i += 1
                    try? await Task.sleep(nanoseconds: dt)
                }
            }
        }
    }

    deinit { driver?.cancel() }
}

// MARK: - Demo view (same look as the image app: header, card, prompt capsule)

struct TrackerDemoView: View {
    @ObservedObject var model: TrackerDemoModel

    var body: some View {
        VStack(spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text("SAM 3")
                    .font(.system(size: 34, weight: .heavy, design: .rounded))
                Text("on-device")
                    .font(.system(.subheadline, design: .rounded).weight(.semibold))
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(Color.purple.opacity(0.35)))
                Spacer()
            }

            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(Color.white.opacity(0.06))
                if let frame = model.frame {
                    Image(uiImage: frame)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .padding(6)
                }
            }
            .frame(maxHeight: .infinity)

            HStack(spacing: 8) {
                if case .playback = model.phase {
                    if let t = model.trackedChip { chip(t, system: "sparkles", emphasize: true) }
                    if let s = model.speedChip { chip(s, system: "bolt.fill") }
                } else {
                    if let p = model.progressChip { chip(p, system: "scope", emphasize: true) }
                    if let s = model.speedChip { chip(s, system: "bolt.fill") }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let error = model.errorLine {
                Text(error).font(.footnote).foregroundStyle(.red)
            }

            HStack(spacing: 10) {
                Text(model.typedPrompt.isEmpty ? " " : model.typedPrompt)
                    .font(.body)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .frame(height: 46)
                    .background(Capsule().fill(Color.white.opacity(0.10)))
                Image(systemName: phaseIcon)
                    .font(.title3.weight(.bold))
                    .frame(width: 46, height: 46)
                    .background(Circle().fill(Color.purple))
            }

            Text("Text-prompted video tracking · fully on-device · LiteRT GPU")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(16)
    }

    private var phaseIcon: String {
        switch model.phase {
        case .intro: return "arrow.up"
        case .tracking, .rendering: return "scope"
        case .playback: return "play.fill"
        }
    }

    private func chip(_ label: String, system: String, emphasize: Bool = false) -> some View {
        HStack(spacing: 4) {
            Image(systemName: system).font(.caption2)
            Text(label).font(.system(.caption, design: .rounded).weight(.semibold))
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(Capsule().fill(emphasize ? Color.purple.opacity(0.45) : Color.white.opacity(0.08)))
    }
}

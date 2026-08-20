import PhotosUI
import SwiftUI

@main
struct Sam3App: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

// MARK: - View model

@MainActor
final class Sam3ViewModel: ObservableObject {
    enum State {
        case loading(String)
        case needsFiles([String])
        case ready
        case failed(String)
        case trackerDone(String)
        case trackerDemo(TrackerDemoModel)
    }

    @Published var state: State = .loading("Starting…")
    @Published var image: UIImage?
    @Published var resultImage: UIImage?
    @Published var prompt = ""
    @Published var running = false
    @Published var detectionCount: Int?
    @Published var visionLabel: String?
    @Published var textLabel: String?
    @Published var headLabel: String?
    @Published var totalLabel: String?
    @Published var visionCached = false
    @Published var errorLine: String?

    private var detector: Sam3Detector?

    func start() {
        // Tracker autotest mode: when its fixtures are staged in Documents, run the
        // video tracker instead of the image pipeline (loading both graph sets at
        // once would double the GPU memory). Delete Documents/tracker to go back.
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        // Tracker demo mode: hands-free video-tracking showcase for screen
        // recordings (Documents/trackerdemo/demo.json). Takes precedence over the
        // autotest; delete Documents/trackerdemo to fall through.
        if let config = TrackerDemo.config(modelsRoot: docs) {
            state = .loading("Compiling SAM 3 on the Metal GPU…")
            let model = TrackerDemoModel()
            Task.detached(priority: .userInitiated) { [weak self] in
                do {
                    let tracker = try Sam3Tracker(
                        modelsRoot: docs, prompt: config.prompt, gpuAccel: .gpu
                    ) { line in
                        Task { @MainActor [weak self] in
                            let step = line.split(separator: " ").count > 1
                                ? String(line.split(separator: " ")[1]) : line
                            self?.state = .loading("Compiling SAM 3 on the Metal GPU…\n\(step)")
                        }
                    }
                    await MainActor.run { [weak self] in
                        self?.state = .trackerDemo(model)
                        model.begin(tracker: tracker, modelsRoot: docs, config: config)
                    }
                } catch {
                    await MainActor.run { [weak self] in
                        self?.state = .failed("tracker demo: \(error)")
                    }
                }
            }
            return
        }
        if TrackerAutotest.shouldRun(modelsRoot: docs) {
            state = .loading("Tracker autotest — compiling graphs…")
            Task.detached(priority: .userInitiated) { [weak self] in
                do {
                    let verdict = try TrackerAutotest.run(modelsRoot: docs, gpuAccel: .gpu) { line in
                        Task { @MainActor [weak self] in self?.state = .loading(line) }
                    }
                    await MainActor.run { [weak self] in self?.state = .trackerDone(verdict) }
                } catch {
                    await MainActor.run { [weak self] in
                        self?.state = .failed("tracker: \(error)")
                    }
                }
            }
            return
        }
        let missing = Sam3Detector.missingFiles()
        if !missing.isEmpty {
            state = .needsFiles(missing)
            return
        }
        state = .loading("Compiling graphs…")
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let detector = try Sam3Detector { stage in
                    Task { @MainActor [weak self] in self?.state = .loading(stage) }
                }
                let report = detector.compileReport
                await MainActor.run { [weak self] in
                    self?.detector = detector
                    self?.state = .ready
                    print("SAM3 compile report:\n\(report)")
                    self?.maybeRunDemo()
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.state = .failed("\(error)")
                }
            }
        }
    }

    func retryStart() { start() }

    func setImage(_ new: UIImage) {
        image = new
        clearResult()
    }

    /// Drop the overlay and chips but keep the current image instance — the detector's
    /// vision features are cached against it, so re-prompting stays instant.
    func clearResult() {
        resultImage = nil
        detectionCount = nil
        visionLabel = nil
        textLabel = nil
        headLabel = nil
        totalLabel = nil
        errorLine = nil
    }

    func detect() {
        guard let detector, let image, !running else { return }
        let query = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }
        running = true
        errorLine = nil
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let detections = try detector.detect(image: image, prompt: query)
                let composited = OverlayRenderer.render(image: image, detections: detections)
                let t = detector.timing
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    self.resultImage = composited
                    self.detectionCount = detections.count
                    self.visionCached = t.visionCached
                    self.totalLabel = t.visionCached
                        ? "re-prompt \(Self.ms(t.textMs + t.headMs))"
                        : Self.ms(t.visionMs + t.textMs + t.headMs)
                    self.visionLabel = t.visionCached
                        ? "Vision cached (same photo)"
                        : "Vision \(Self.ms(t.visionMs)) · \(detector.visionAccel)"
                    self.textLabel = "Text \(Self.ms(t.textMs)) · \(detector.textAccel)"
                    self.headLabel = "Head \(Self.ms(t.headMs)) · \(detector.headAccel)"
                    self.running = false
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.errorLine = "\(error)"
                    self?.running = false
                }
            }
        }
    }

    nonisolated private static func ms(_ v: Double) -> String {
        v >= 1000 ? String(format: "%.1f s", v / 1000) : String(format: "%.0f ms", v)
    }

    // MARK: Scripted demo — auto-runs when Documents/demo/demo.json exists, so a
    // screen recording needs no hands on the device. Delete Documents/demo to disable.

    private struct DemoStep: Decodable {
        let photo: String
        let prompt: String
        let dwell: Double?
    }
    private struct DemoScript: Decodable {
        let steps: [DemoStep]
        let loop: Bool?
        let startDelay: Double?
    }

    func maybeRunDemo() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("demo")
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("demo.json")),
              let script = try? JSONDecoder().decode(DemoScript.self, from: data)
        else { return }
        // One UIImage per file, reused across steps: the detector caches vision
        // features per image INSTANCE, so re-loading the file would silently re-run
        // the 5 s vision graph and hide the instant re-prompt this demo shows off.
        var loaded: [String: UIImage] = [:]
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64((script.startDelay ?? 1.5) * 1e9))
            repeat {
                for step in script.steps {
                    guard let self else { return }
                    let path = dir.appendingPathComponent(step.photo).path
                    let image: UIImage
                    if let cached = loaded[step.photo] {
                        image = cached
                    } else if let fresh = UIImage(contentsOfFile: path) {
                        loaded[step.photo] = fresh
                        image = fresh
                    } else {
                        continue
                    }
                    if self.image !== image { self.setImage(image) } else { self.clearResult() }
                    self.prompt = ""
                    try? await Task.sleep(nanoseconds: 700_000_000)
                    for ch in step.prompt {
                        self.prompt.append(ch)
                        try? await Task.sleep(nanoseconds: UInt64.random(in: 55_000_000...120_000_000))
                    }
                    try? await Task.sleep(nanoseconds: 400_000_000)
                    self.detect()
                    while self.running { try? await Task.sleep(nanoseconds: 100_000_000) }
                    try? await Task.sleep(nanoseconds: UInt64((step.dwell ?? 3.0) * 1e9))
                }
            } while script.loop == true && self != nil
        }
    }
}

// MARK: - Overlay rendering

enum OverlayRenderer {
    static let palette: [UIColor] = [
        UIColor(red: 1.00, green: 0.28, blue: 0.34, alpha: 1),
        UIColor(red: 0.18, green: 0.84, blue: 0.45, alpha: 1),
        UIColor(red: 0.12, green: 0.56, blue: 1.00, alpha: 1),
        UIColor(red: 1.00, green: 0.65, blue: 0.01, alpha: 1),
        UIColor(red: 0.65, green: 0.37, blue: 0.92, alpha: 1),
        UIColor(red: 0.00, green: 0.82, blue: 0.83, alpha: 1),
    ]

    /// Composite masks (50% tint), boxes, and score labels onto the image.
    /// Boxes/masks are normalized to the full (squashed-to-square) image extent,
    /// so drawing in the original image rect is geometrically consistent.
    static func render(image: UIImage, detections: [Sam3Detector.Detection]) -> UIImage {
        let maxSide: CGFloat = 2048
        let scale = min(1, maxSide / max(image.size.width, image.size.height))
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            image.draw(in: CGRect(origin: .zero, size: size))
            let cg = ctx.cgContext
            for (i, det) in detections.enumerated() {
                let color = palette[i % palette.count]
                if let maskImage = tintedMask(det.mask, color: color) {
                    cg.saveGState()
                    cg.interpolationQuality = .medium
                    cg.draw(maskImage, in: CGRect(origin: .zero, size: size))
                    cg.restoreGState()
                }
            }
            for (i, det) in detections.enumerated() {
                let color = palette[i % palette.count]
                let cx = CGFloat(det.box[0]) * size.width
                let cy = CGFloat(det.box[1]) * size.height
                let w = CGFloat(det.box[2]) * size.width
                let h = CGFloat(det.box[3]) * size.height
                let rect = CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
                let lineWidth = max(2, size.width / 300)
                cg.setStrokeColor(color.cgColor)
                cg.setLineWidth(lineWidth)
                cg.stroke(rect)

                let label = String(format: " %.2f ", det.score)
                let font = UIFont.systemFont(ofSize: max(12, size.width / 45), weight: .bold)
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: font, .foregroundColor: UIColor.white,
                    .backgroundColor: color.withAlphaComponent(0.85),
                ]
                let text = NSAttributedString(string: label, attributes: attrs)
                let textSize = text.size()
                var origin = CGPoint(x: rect.minX, y: rect.minY - textSize.height)
                if origin.y < 0 { origin.y = rect.minY }
                text.draw(at: origin)
            }
        }
    }

    /// 288x288 RGBA bitmap: `color` with alpha 0.5 where mask prob > 0.5.
    private static func tintedMask(_ mask: [Float], color: UIColor) -> CGImage? {
        let side = Sam3Detector.maskSize
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        color.getRed(&r, green: &g, blue: &b, alpha: &a)
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        // premultiplied alpha
        let pr = UInt8(r * 0.5 * 255), pg = UInt8(g * 0.5 * 255), pb = UInt8(b * 0.5 * 255)
        for i in 0..<(side * side) where mask[i] > 0.5 {
            pixels[i * 4] = pr
            pixels[i * 4 + 1] = pg
            pixels[i * 4 + 2] = pb
            pixels[i * 4 + 3] = 128
        }
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}

// MARK: - Views

struct ContentView: View {
    @StateObject private var vm = Sam3ViewModel()
    @State private var pickerItem: PhotosPickerItem?
    @FocusState private var promptFocused: Bool

    private let suggestions = ["person", "wheel", "shoe", "bottle", "window", "dog"]

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.05, green: 0.05, blue: 0.10), Color(red: 0.10, green: 0.08, blue: 0.18)],
                startPoint: .top, endPoint: .bottom
            ).ignoresSafeArea()

            switch vm.state {
            case .loading(let stage):
                loadingView(stage)
            case .needsFiles(let missing):
                needsFilesView(missing)
            case .failed(let message):
                failedView(message)
            case .trackerDone(let verdict):
                trackerDoneView(verdict)
            case .trackerDemo(let model):
                TrackerDemoView(model: model)
            case .ready:
                mainView
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            // Long unattended compiles/tracking die if the phone auto-locks
            // (backgrounded Metal work is terminated) — keep the screen awake.
            UIApplication.shared.isIdleTimerDisabled = true
            vm.start()
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("SAM 3")
                .font(.system(size: 34, weight: .heavy, design: .rounded))
            Text("on-device")
                .font(.system(.subheadline, design: .rounded).weight(.semibold))
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(Capsule().fill(Color.purple.opacity(0.35)))
            Spacer()
        }
    }

    private var mainView: some View {
        VStack(spacing: 14) {
            header

            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(Color.white.opacity(0.06))
                if let shown = vm.resultImage ?? vm.image {
                    Image(uiImage: shown)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 20))
                        .padding(6)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.on.rectangle.angled")
                            .font(.system(size: 44))
                            .foregroundStyle(.secondary)
                        Text("Pick a photo, type what to find")
                            .foregroundStyle(.secondary)
                    }
                }
                if vm.running {
                    RoundedRectangle(cornerRadius: 24).fill(.black.opacity(0.45))
                    ProgressView("Segmenting…")
                        .tint(.white)
                }
            }
            .frame(maxHeight: .infinity)

            if let count = vm.detectionCount {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        chip("\(count) found", system: "sparkles", emphasize: true)
                        if let total = vm.totalLabel {
                            chip(total, system: vm.visionCached ? "bolt.fill" : "timer",
                                 emphasize: vm.visionCached)
                        }
                    }
                    HStack(spacing: 8) {
                        if let v = vm.visionLabel { chip(v, system: "eye") }
                        if let t = vm.textLabel { chip(t, system: "textformat") }
                        if let h = vm.headLabel { chip(h, system: "square.stack.3d.up") }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            if let error = vm.errorLine {
                Text(error).font(.footnote).foregroundStyle(.red)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(suggestions, id: \.self) { s in
                        Button {
                            vm.prompt = s
                            vm.detect()
                        } label: {
                            Text(s)
                                .font(.system(.subheadline, design: .rounded).weight(.medium))
                                .padding(.horizontal, 12).padding(.vertical, 6)
                                .background(Capsule().fill(Color.white.opacity(0.10)))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            HStack(spacing: 10) {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Image(systemName: "photo.fill.on.rectangle.fill")
                        .font(.title3)
                        .frame(width: 46, height: 46)
                        .background(Circle().fill(Color.white.opacity(0.10)))
                }
                TextField("what should I find?", text: $vm.prompt)
                    .textFieldStyle(.plain)
                    .focused($promptFocused)
                    .submitLabel(.search)
                    .onSubmit { vm.detect() }
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .padding(.horizontal, 16)
                    .frame(height: 46)
                    .background(Capsule().fill(Color.white.opacity(0.10)))
                Button {
                    promptFocused = false
                    vm.detect()
                } label: {
                    Image(systemName: "arrow.up")
                        .font(.title3.weight(.bold))
                        .frame(width: 46, height: 46)
                        .background(Circle().fill(vm.running ? Color.gray : Color.purple))
                }
                .disabled(vm.running || vm.image == nil)
            }

            Text("ViT-L/14 @1008 · LiteRT CompiledModel · Metal GPU")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(16)
        .onChange(of: pickerItem) { item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    vm.setImage(image)
                }
            }
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

    private func loadingView(_ stage: String) -> some View {
        VStack(spacing: 16) {
            ProgressView().scaleEffect(1.4).tint(.white)
            Text(stage).font(.system(.body, design: .rounded))
            Text("First launch compiles ~1.6 GB of graphs on the Metal GPU — up to a minute.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }

    private func needsFilesView(_ missing: [String]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Model files missing").font(.title2.weight(.bold))
            Text("Copy these into the app's Documents folder (Finder → iPhone → Files → SAM3), then retry:")
                .foregroundStyle(.secondary)
            ForEach(missing, id: \.self) { f in
                Label(f, systemImage: "doc.badge.arrow.up").font(.system(.footnote, design: .monospaced))
            }
            Text("Files come from sam3/models/out/ (see sam3/ios/README.md).")
                .font(.caption).foregroundStyle(.tertiary)
            Button("Retry") { vm.retryStart() }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
        }
        .padding(24)
    }

    private func trackerDoneView(_ verdict: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: verdict.contains("ids-agree=true") ? "checkmark.seal.fill" : "xmark.seal.fill")
                .font(.largeTitle)
                .foregroundStyle(verdict.contains("ids-agree=true") ? .green : .red)
            Text("Tracker autotest").font(.title3.weight(.semibold))
            Text(verdict).font(.system(.footnote, design: .monospaced))
                .multilineTextAlignment(.center)
            Text("Full log: Documents/tracker_result.txt. Delete Documents/tracker to return to image mode.")
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(24)
    }

    private func failedView(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.largeTitle).foregroundStyle(.yellow)
            Text("Model load failed").font(.title3.weight(.semibold))
            Text(message).font(.footnote).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Retry") { vm.retryStart() }
                .buttonStyle(.borderedProminent)
                .tint(.purple)
        }
        .padding(24)
    }
}

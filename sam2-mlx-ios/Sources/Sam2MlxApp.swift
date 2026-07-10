import SwiftUI
import Foundation
import MLX
import MLXNN

@main
struct Sam2MlxApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// SAM2 (hiera-tiny) MLX-swift on-device benchmark — the MLX side of the LiteRT-vs-MLX
/// comparison, measured on the same iPhone as the LiteRT harness. Runs the ported model
/// (numerically corr 1.0 vs the Python MLX reference) and reports warm encoder / decoder
/// latency to the device console (NSLog "SAM2MLXBENCH") and on screen.
struct ContentView: View {
    @State private var status = "Loading MLX SAM2…"
    @State private var running = true

    var body: some View {
        VStack(spacing: 20) {
            Text("SAM2 · MLX-swift benchmark").font(.headline)
            Text(status).font(.system(.footnote, design: .monospaced))
                .multilineTextAlignment(.leading).padding()
            Button(running ? "Running…" : "Re-run") { run() }.disabled(running)
        }
        .padding()
        .onAppear { run() }
    }

    private func run() {
        running = true
        status = "Loading MLX SAM2…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result: String
            do { result = try Benchmark.run() } catch { result = "FAIL \(error)" }
            NSLog("SAM2MLXBENCH\n%@", result)
            DispatchQueue.main.async { status = result; running = false }
        }
    }
}

enum Benchmark {
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    static func circle() -> MLXArray {
        let size = 1024
        var arr = [Float](repeating: 0, count: 3 * size * size)
        let plane = size * size
        let (cx, cy, r2) = (size / 2, size / 2, 240 * 240)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let value: Float = ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) ? 1.0 : 0.0
                let i = y * size + x
                arr[i] = (value - mean[0]) / std[0]
                arr[plane + i] = (value - mean[1]) / std[1]
                arr[2 * plane + i] = (value - mean[2]) / std[2]
            }
        }
        return MLXArray(arr, [1, 3, size, size]).asType(.float16)
    }

    static func run() throws -> String {
        guard let url = Bundle.main.url(forResource: "sam2_tiny", withExtension: "safetensors") else {
            throw NSError(domain: "SAM2MLX", code: 1, userInfo: [NSLocalizedDescriptionKey: "safetensors not bundled"])
        }
        let prefixes = ["trunk.", "neck.", "sam_prompt_encoder.", "sam_mask_decoder.", "no_mem_embed"]
        let all = try loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (k, v) in all where prefixes.contains(where: { k.hasPrefix($0) }) { weights[k] = v }

        let model = Sam2ImageSegmenter(config: HieraCfg())
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(model)

        let pixels = circle()
        let coords = MLXArray([Float(512), Float(512)], [1, 1, 2])
        let labels = MLXArray([Int32(1)], [1, 1])

        func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
        let warmup = 5, runs = 20
        for _ in 0 ..< warmup {
            let (v, hr) = model.encodeImage(pixels)
            eval(model.predict(v, hr, coords, labels))
        }
        var encTimes: [Double] = []
        for _ in 0 ..< runs {
            let t = Date()
            let (v, _) = model.encodeImage(pixels); eval(v)
            encTimes.append(-t.timeIntervalSinceNow * 1000)
        }
        let (v0, hr0) = model.encodeImage(pixels); eval(v0)
        var decTimes: [Double] = []
        for _ in 0 ..< runs {
            let t = Date()
            eval(model.predict(v0, hr0, coords, labels))
            decTimes.append(-t.timeIntervalSinceNow * 1000)
        }
        let masks = model.predict(v0, hr0, coords, labels)
        eval(masks)
        let fg = (masks[0, 0] .> 0).sum().item(Int32.self)

        return String(format: """
            SAM2 hiera-tiny · MLX-swift (fp16)
            enc_median=%.1fms  dec_median=%.1fms  (runs=%d)
            mask_fg=%d/65536
            """, median(encTimes), median(decTimes), runs, fg)
    }
}

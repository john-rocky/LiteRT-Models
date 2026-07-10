import SwiftUI

@main
struct Sam2App: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// SAM2 (hiera-tiny) LiteRT-on-Apple-GPU benchmark. On launch it compiles the encoder and
/// mask decoder on the GPU, reports whether each is fully accelerated, and times the warm
/// encoder / decoder latency on a synthetic circle — the LiteRT side of the LiteRT-vs-MLX
/// comparison, measured on the same Apple silicon MLX runs on.
struct ContentView: View {
    @State private var status = "Compiling SAM2 on GPU…"
    @State private var running = true

    var body: some View {
        VStack(spacing: 20) {
            Text("SAM2 · LiteRT Apple-GPU benchmark")
                .font(.headline)
            Text(status)
                .font(.system(.footnote, design: .monospaced))
                .multilineTextAlignment(.leading)
                .padding()
            Button(running ? "Running…" : "Re-run benchmark") { run() }
                .disabled(running)
        }
        .padding()
        .onAppear { run() }
    }

    private func run() {
        running = true
        status = "Compiling SAM2 on GPU…"
        DispatchQueue.global(qos: .userInitiated).async {
            let result: String
            do {
                result = try Benchmark.run()
            } catch {
                result = "FAIL \(error)"
            }
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            try? result.write(to: docs.appendingPathComponent("result.txt"),
                              atomically: true, encoding: .utf8)
            // Emit to the device console so the result can be read headlessly on a real device.
            NSLog("SAM2BENCH\n%@", result)
            DispatchQueue.main.async {
                status = result
                running = false
            }
        }
    }
}

/// SAM2 image-path benchmark: encoder once, mask decoder per point, flat concatenated I/O
/// (identical layout to the Android module and the converted tflite).
enum Benchmark {
    static let SIZE = 1024
    static let IE = 256 * 64 * 64          // image_embeddings
    static let F0 = 32 * 256 * 256         // high-res feature s0
    static let F1 = 64 * 128 * 128         // high-res feature s1
    static let SP = 512                    // sparse prompt (2 x 256)
    static let ENC_OUT = IE + F0 + F1      // encoder flat output
    static let DEC_IN = IE + SP + F0 + F1  // decoder flat input
    static let DEC_OUT = 3 * 256 * 256     // 3 multimask logits
    static let MEAN: [Float] = [0.485, 0.456, 0.406]
    static let STD: [Float] = [0.229, 0.224, 0.225]
    static let WARMUP = 5
    static let RUNS = 20

    static func bundlePath(_ name: String, _ ext: String) throws -> String {
        guard let path = Bundle.main.path(forResource: name, ofType: ext) else {
            throw NSError(domain: "SAM2", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(name).\(ext) not in bundle"])
        }
        return path
    }

    static func run() throws -> String {
        let encoder = LiteRTModel()
        let decoder = LiteRTModel()
        let (encFully, encCompile) = try encoder.compileOnGPU(path: try bundlePath("sam2_encoder", "tflite"))
        let (decFully, decCompile) = try decoder.compileOnGPU(path: try bundlePath("sam2_decoder", "tflite"))

        // Prompt-encoder constants (Gaussian projection, point embed, not-a-point).
        let prompt = try loadFloats(try bundlePath("sam2_prompt", "bin"))
        let gaussian = Array(prompt[0..<256])
        let pointEmbed1 = Array(prompt[256..<512])
        let notAPoint = Array(prompt[512..<768])

        let pixels = circleInput()
        let encDims: [Int32] = [1, 3, Int32(SIZE), Int32(SIZE)]
        let encOutDims: [Int32] = [1, Int32(ENC_OUT)]
        let decInDims: [Int32] = [1, Int32(DEC_IN)]
        let decOutDims: [Int32] = [1, 3, 256, 256]

        func encode() throws -> [Float] {
            try encoder.run(input: pixels, inputDims: encDims, outputCount: ENC_OUT, outputDims: encOutDims)
        }
        func decode(_ flat: [Float]) throws -> [Float] {
            let ie = Array(flat[0..<IE])
            let f0 = Array(flat[IE..<(IE + F0)])
            let f1 = Array(flat[(IE + F0)..<(IE + F0 + F1)])
            let sparse = sparseEmbedding(gaussian: gaussian, pointEmbed1: pointEmbed1, notAPoint: notAPoint)
            var decIn = [Float](repeating: 0, count: DEC_IN)
            decIn.replaceSubrange(0..<IE, with: ie)
            decIn.replaceSubrange(IE..<(IE + SP), with: sparse)
            decIn.replaceSubrange((IE + SP)..<(IE + SP + F0), with: f0)
            decIn.replaceSubrange((IE + SP + F0)..<DEC_IN, with: f1)
            return try decoder.run(input: decIn, inputDims: decInDims, outputCount: DEC_OUT, outputDims: decOutDims)
        }

        var flat = try encode()
        for _ in 0..<WARMUP { flat = try encode(); _ = try decode(flat) }

        let encTimes = try (0..<RUNS).map { _ -> Double in
            let t = Date(); flat = try encode(); return -t.timeIntervalSinceNow * 1000.0
        }
        flat = try encode()
        let decTimes = try (0..<RUNS).map { _ -> Double in
            let t = Date(); _ = try decode(flat); return -t.timeIntervalSinceNow * 1000.0
        }
        let mask = try decode(flat)
        let fg = mask.prefix(256 * 256).reduce(0) { $0 + ($1 > 0 ? 1 : 0) }

        return String(
            format: """
            SAM2 hiera-tiny · LiteRT Apple GPU
            encoder: fullyGPU=%@ compile=%.1fs
            decoder: fullyGPU=%@ compile=%.1fs
            enc_median=%.1fms  dec_median=%.1fms  (runs=%d)
            mask_fg=%d/%d
            """,
            encFully ? "YES" : "NO", encCompile,
            decFully ? "YES" : "NO", decCompile,
            median(encTimes), median(decTimes), RUNS, fg, 256 * 256)
    }

    /// Sinusoidal position encoding of a positive point at the image center.
    static func sparseEmbedding(gaussian: [Float], pointEmbed1: [Float], notAPoint: [Float]) -> [Float] {
        var sparse = [Float](repeating: 0, count: SP)
        let center = Float(SIZE) / 2.0
        let cc = 2.0 * ((center + 0.5) / Float(SIZE)) - 1.0
        let twoPi = Float(2.0 * Double.pi)
        for k in 0..<128 {
            let proj = twoPi * (cc * gaussian[k] + cc * gaussian[128 + k])
            sparse[k] = sin(proj) + pointEmbed1[k]
            sparse[128 + k] = cos(proj) + pointEmbed1[128 + k]
        }
        for k in 0..<256 { sparse[256 + k] = notAPoint[k] }
        return sparse
    }

    /// A normalized white-circle-on-black image as a CHW float array [3, 1024, 1024].
    static func circleInput() -> [Float] {
        let plane = SIZE * SIZE
        var out = [Float](repeating: 0, count: 3 * plane)
        let cx = SIZE / 2, cy = SIZE / 2, r2 = 240 * 240
        for y in 0..<SIZE {
            for x in 0..<SIZE {
                let value: Float = ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2) ? 1.0 : 0.0
                let i = y * SIZE + x
                out[i] = (value - MEAN[0]) / STD[0]
                out[plane + i] = (value - MEAN[1]) / STD[1]
                out[2 * plane + i] = (value - MEAN[2]) / STD[2]
            }
        }
        return out
    }

    static func loadFloats(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }
}

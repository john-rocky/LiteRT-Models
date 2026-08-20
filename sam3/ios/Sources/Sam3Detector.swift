import CoreGraphics
import Foundation
import UIKit

/// SAM 3.1 text-prompted detection + instance segmentation (image side of the
/// Object-Multiplex checkpoint) on LiteRT CompiledModel, three graphs:
///   vision : image [1,3,1008,1008] -> [fpn288 | fpn144 | fpn72]         (Metal fp16)
///   text   : token embeddings [1,32,1024] -> text memory [32*256]       (CPU — exact;
///            the CLIP-L residual stream reaches |x|~1.2e3 and fp16 GPU execution
///            corrupts the prompt embedding for some prompts)
///   head   : [fpn x3 | text_mem | pad(32)] -> [logits(200) | boxes(200x4 cxcywh,
///            normalized) | presence(1) | mask logits (200x288x288)]     (Metal f32)
/// Host: BPE tokenize, fp16 token-embedding lookup, score threshold, mask sigmoid.
/// score = sigmoid(logit) * sigmoid(presence); keep > threshold.
final class Sam3Detector {
    static let size = 1008
    static let tok = 32
    static let tdim = 1024
    static let queries = 200
    static let maskSize = 288
    static let nVis = 256 * (288 * 288 + 144 * 144 + 72 * 72)

    struct Detection: Identifiable {
        let id = UUID()
        let score: Float
        /// cx, cy, w, h normalized to [0,1]
        let box: [Float]
        /// sigmoid mask probabilities, maskSize x maskSize, row-major
        let mask: [Float]
    }

    struct StageTiming {
        var visionMs = 0.0
        var textMs = 0.0
        var headMs = 0.0
        var visionCached = false
    }

    /// Model files live in Documents (copied over from Finder / the Files app);
    /// bundle resources are the fallback so a dev build can also embed them.
    static func locate(_ name: String) -> URL? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let inDocs = docs.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: inDocs.path) { return inDocs }
        let stem = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        return Bundle.main.url(forResource: stem, withExtension: ext)
    }

    static let requiredFiles = [
        "sam3_vision.tflite", "sam3_text.tflite", "sam3_head.tflite",
        "sam3_token_embed.bin", "vocab.json", "merges.txt",
    ]

    static func missingFiles() -> [String] {
        requiredFiles.filter { locate($0) == nil }
    }

    private let tokenizer: BpeTokenizer
    private let vision: Sam3Graph
    private let text: Sam3Graph
    private let head: Sam3Graph
    /// fp16 [49408 x 1024] row-major token-embedding table, memory-mapped (101 MB).
    private let tokenTable: Data

    private(set) var timing = StageTiming()
    var compileReport: String {
        [vision, text, head]
            .map { "\($0.name): \($0.accel.label), compile \(String(format: "%.1f", $0.compileSeconds))s, fully_accelerated=\($0.fullyAccelerated)" }
            .joined(separator: "\n")
    }

    /// Compiles all three graphs; slow on first launch (Metal shader compile).
    /// `progress` is called with a stage label before each compile.
    init(progress: @escaping (String) -> Void) throws {
        func url(_ name: String) throws -> URL {
            guard let u = Sam3Detector.locate(name) else {
                throw LiteRTError.interface("missing \(name)")
            }
            return u
        }
        // Compile with the preferred accelerator, falling back down the list so a
        // delegate failure on some device/OS degrades instead of blocking the app.
        func compile(_ name: String, prefer: [Sam3Accel]) throws -> Sam3Graph {
            let path = try url(name).path
            var lastError: Error?
            for accel in prefer {
                do {
                    return try Sam3Graph(path: path, accel: accel)
                } catch {
                    lastError = error
                    print("SAM3: \(name) on \(accel.label) failed: \(error)")
                }
            }
            throw lastError ?? LiteRTError.interface("no accelerator for \(name)")
        }
        progress("Loading tokenizer…")
        tokenizer = try BpeTokenizer(vocabURL: try url("vocab.json"), mergesURL: try url("merges.txt"))
        tokenTable = try Data(contentsOf: try url("sam3_token_embed.bin"), options: .alwaysMapped)
        progress("Compiling vision graph on the GPU…")
        vision = try compile("sam3_vision.tflite", prefer: [.gpu, .cpu])
        progress("Compiling text graph…")
        text = try compile("sam3_text.tflite", prefer: [.cpu])
        progress("Compiling detection head on the GPU…")
        head = try compile("sam3_head.tflite", prefer: [.gpuF32, .gpu, .cpu])
    }

    var visionAccel: String { vision.accel.label }
    var textAccel: String { text.accel.label }
    var headAccel: String { head.accel.label }

    private func preprocess(_ image: UIImage) -> [Float] {
        let side = Self.size
        let space = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        pixels.withUnsafeMutableBytes { raw in
            let ctx = CGContext(
                data: raw.baseAddress, width: side, height: side, bitsPerComponent: 8,
                bytesPerRow: side * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .medium
            if let cg = image.cgImage {
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: side, height: side))
            }
        }
        let plane = side * side
        var out = [Float](repeating: 0, count: 3 * plane)
        out.withUnsafeMutableBufferPointer { o in
            pixels.withUnsafeBufferPointer { p in
                for i in 0..<plane {
                    o[i] = (Float(p[i * 4]) / 255 - 0.5) / 0.5
                    o[plane + i] = (Float(p[i * 4 + 1]) / 255 - 0.5) / 0.5
                    o[2 * plane + i] = (Float(p[i * 4 + 2]) / 255 - 0.5) / 0.5
                }
            }
        }
        return out
    }

    // Vision features are per-image; cache them so re-prompting skips the big graph.
    private var cachedKey: ObjectIdentifier?
    private var visFeat: [Float] = []

    private func runVision(_ image: UIImage) throws {
        if cachedKey == ObjectIdentifier(image) {
            timing.visionMs = 0
            timing.visionCached = true
            return
        }
        let t0 = Date()
        let input = preprocess(image)
        visFeat = try vision.run(input)
        timing.visionMs = Date().timeIntervalSince(t0) * 1000
        timing.visionCached = false
        cachedKey = ObjectIdentifier(image)
    }

    private func runText(_ prompt: String) throws -> (mem: [Float], pad: [Float]) {
        let t0 = Date()
        let ids = tokenizer.encode(prompt)
        var emb = [Float](repeating: 0, count: Self.tok * Self.tdim)
        tokenTable.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let table = raw.bindMemory(to: Float16.self)
            for t in 0..<Self.tok {
                let base = ids[t] * Self.tdim
                for d in 0..<Self.tdim {
                    emb[t * Self.tdim + d] = Float(table[base + d])
                }
            }
        }
        let mem = try text.run(emb)  // [32*256]
        let pad = (0..<Self.tok).map { ids[$0] == 0 ? Float(1) : Float(0) }
        timing.textMs = Date().timeIntervalSince(t0) * 1000
        return (mem, pad)
    }

    /// Returns detections above `threshold`, unsorted (query order).
    func detect(image: UIImage, prompt: String, threshold: Float = 0.5) throws -> [Detection] {
        try runVision(image)
        let (mem, pad) = try runText(prompt)
        let t0 = Date()
        var headInput = [Float](repeating: 0, count: Self.nVis + Self.tok * 256 + Self.tok)
        headInput.replaceSubrange(0..<Self.nVis, with: visFeat)
        headInput.replaceSubrange(Self.nVis..<(Self.nVis + Self.tok * 256), with: mem)
        headInput.replaceSubrange((Self.nVis + Self.tok * 256)..., with: pad)
        let y = try head.run(headInput)
        timing.headMs = Date().timeIntervalSince(t0) * 1000

        let presence = 1 / (1 + exp(-y[1000]))
        var out: [Detection] = []
        let maskArea = Self.maskSize * Self.maskSize
        for q in 0..<Self.queries {
            let score = 1 / (1 + exp(-y[q])) * presence
            if score <= threshold { continue }
            let box = Array(y[(200 + q * 4)..<(200 + q * 4 + 4)])
            let base = 1001 + q * maskArea
            var mask = [Float](repeating: 0, count: maskArea)
            for i in 0..<maskArea { mask[i] = 1 / (1 + exp(-y[base + i])) }
            out.append(Detection(score: score, box: box, mask: mask))
        }
        return out
    }
}

import Foundation

/// SAM 2.1 VIDEO-path device probe: the four per-frame graphs (encode / memcond{7,2} /
/// decode / memorize) exported by `sam2/scripts/convert_sam2_video.py`.
///
/// For each graph: compile on the GPU, report fully-accelerated + compile time, run the
/// Mac-recorded test vector (`<name>_in.bin`, from `verify_sam2_video.py vectors`), compare
/// the output against the Mac fp16 Interpreter output (`<name>_out.bin`: corr, max|diff|,
/// and binary-mask IoU for the decoder) — the on-device numeric check that catches silent
/// GPU miscomputes — then time warm runs. Ends with the per-frame tracking budget
/// (memcond + decode + memorize, encoder listed separately).
enum VideoBench {
    static let WARMUP = 3
    static let RUNS = 10
    static let IE = 256 * 64 * 64
    static let GRAPHS: [(name: String, inDims: [Int32], outDims: [Int32])] = [
        ("sam2v_encode", [1, 3, 1024, 1024], [1, 4_194_304]),
        ("sam2v_memcond7", [1, 2_920_960], [1, 1_048_576]),
        ("sam2v_memcond2", [1, 1_589_440], [1, 1_048_576]),
        ("sam2v_decode", [1, 4_194_817], [1, 263_173]),
        ("sam2v_memorize", [1, 2_097_153], [1, 262_144]),
    ]

    static func run() throws -> String {
        var report = "SAM2.1 hiera-tiny VIDEO path · LiteRT Apple GPU (iPhone)\n"
        var perFrame: [String: Double] = [:]
        for g in GRAPHS {
            guard let modelPath = Bundle.main.path(forResource: g.name, ofType: "tflite") else {
                report += "\(g.name): (not bundled)\n"
                continue
            }
            let model = LiteRTModel()
            let (fully, compileSec): (Bool, Double)
            do {
                (fully, compileSec) = try model.compileOnGPU(path: modelPath)
            } catch {
                report += "\(g.name): COMPILE FAIL \(error)\n"
                continue
            }
            let inCount = Int(g.inDims.reduce(1, *))
            let outCount = Int(g.outDims.reduce(1, *))
            var line = String(format: "%@: fullyGPU=%@ compile=%.1fs", g.name, fully ? "YES" : "NO", compileSec)

            var input: [Float]
            var expected: [Float]? = nil
            if let ip = Bundle.main.path(forResource: g.name + "_in", ofType: "bin"),
               let v = try? Benchmark.loadFloats(ip), v.count == inCount {
                input = v
                if let op = Bundle.main.path(forResource: g.name + "_out", ofType: "bin"),
                   let e = try? Benchmark.loadFloats(op), e.count == outCount {
                    expected = e
                }
            } else {
                // No recorded vector: deterministic pseudo-random input, timing only.
                var s: UInt32 = 12345
                input = (0..<inCount).map { _ in
                    s = s &* 1_664_525 &+ 1_013_904_223
                    return Float(s >> 8) / Float(1 << 24) - 0.5
                }
                line += " (no vector: timing only)"
            }
            do {
                let out = try model.run(input: input, inputDims: g.inDims, outputCount: outCount, outputDims: g.outDims)
                if let e = expected {
                    let (c, md) = corrMaxDiff(out, e)
                    line += String(format: "  vs Mac: corr=%.5f max|d|=%.4f", c, md)
                    if g.name == "sam2v_decode" {
                        // masks(4x256x256) | iou(4) | ptr(4x256) | obj(1): IoU of the best multimask token
                        let iouOff = 4 * 65536
                        var best = 1
                        for k in 2..<4 where e[iouOff + k] > e[iouOff + best] { best = k }
                        let a = Array(out[(best * 65536)..<((best + 1) * 65536)])
                        let b = Array(e[(best * 65536)..<((best + 1) * 65536)])
                        line += String(format: " maskIoU=%.4f fg=%d/%d obj=%.2f/%.2f",
                                       maskIoU(a, b), a.filter { $0 > 0 }.count, b.filter { $0 > 0 }.count,
                                       out[outCount - 1], e[outCount - 1])
                    }
                }
                for _ in 0..<WARMUP {
                    _ = try model.run(input: input, inputDims: g.inDims, outputCount: outCount, outputDims: g.outDims)
                }
                let times = try (0..<RUNS).map { _ -> Double in
                    let t = Date()
                    _ = try model.run(input: input, inputDims: g.inDims, outputCount: outCount, outputDims: g.outDims)
                    return -t.timeIntervalSinceNow * 1000.0
                }
                let med = Benchmark.median(times)
                perFrame[g.name] = med
                line += String(format: "  median=%.1fms (min %.1f, runs=%d)", med, times.min() ?? 0, RUNS)
            } catch {
                line += " RUN FAIL \(error)"
            }
            report += line + "\n"
            NSLog("SAM2VBENCH %@", line)
        }
        if let d = perFrame["sam2v_decode"], let m = perFrame["sam2v_memorize"] {
            let e = perFrame["sam2v_encode"] ?? 0
            if let c7 = perFrame["sam2v_memcond7"] {
                report += String(format: "per-frame (7-slot): memcond+decode+memorize=%.1fms  +encode=%.1fms\n",
                                 c7 + d + m, c7 + d + m + e)
            }
            if let c2 = perFrame["sam2v_memcond2"] {
                report += String(format: "per-frame (2-slot): memcond+decode+memorize=%.1fms  +encode=%.1fms\n",
                                 c2 + d + m, c2 + d + m + e)
            }
        }
        return report
    }

    static func corrMaxDiff(_ a: [Float], _ b: [Float]) -> (Double, Double) {
        let n = Double(a.count)
        var sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0, md = 0.0
        for i in 0..<a.count {
            let x = Double(a[i]), y = Double(b[i])
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y
            md = max(md, abs(x - y))
        }
        let cov = sab / n - (sa / n) * (sb / n)
        let va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n)
        let c = (va > 0 && vb > 0) ? cov / (va * vb).squareRoot() : .nan
        return (c, md)
    }

    static func maskIoU(_ a: [Float], _ b: [Float]) -> Double {
        var inter = 0, uni = 0
        for i in 0..<a.count {
            let x = a[i] > 0, y = b[i] > 0
            if x && y { inter += 1 }
            if x || y { uni += 1 }
        }
        return uni == 0 ? 1.0 : Double(inter) / Double(uni)
    }
}

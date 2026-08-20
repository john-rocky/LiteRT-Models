import Foundation

/// Headless tracker verification: when <modelsRoot>/tracker/expected/manifest.json
/// exists, run the full tracker on the bundled clip and compare per frame against
/// the Mac host-loop fixtures (ids equal, |dprob|, video-res mask IoU). Results go
/// to the progress callback and <modelsRoot>/tracker_result.txt.
enum TrackerAutotest {
    static func shouldRun(modelsRoot: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: modelsRoot.appendingPathComponent("tracker/expected/manifest.json").path)
    }

    private static func readInts(_ f: URL) throws -> [Int32] {
        let data = try Data(contentsOf: f)
        var out = [Int32](repeating: 0, count: data.count / 4)
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }

    private static func readFloats(_ f: URL) throws -> [Float] {
        let data = try Data(contentsOf: f)
        var out = [Float](repeating: 0, count: data.count / 4)
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }

    /// packed 1 bit/px LSB-first (np.packbits bitorder=little) -> per-object words.
    private static func readMasks(_ f: URL, _ nObj: Int, _ px: Int) throws -> [[UInt64]] {
        let bytes = [UInt8](try Data(contentsOf: f))
        let bytesPerObj = (px + 7) / 8
        return (0..<nObj).map { o in
            var w = [UInt64](repeating: 0, count: TM.packedWords(px))
            for b in 0..<bytesPerObj {
                let v = UInt64(bytes[o * bytesPerObj + b])
                if v != 0 {
                    let bit = b * 8
                    w[bit >> 6] |= v << UInt64(bit & 63)
                }
            }
            return w
        }
    }

    /// Runs the autotest and returns the verdict line. `gpuAccel` is .gpu on the
    /// iPhone (ML Drift Metal) and .cpu on the macOS harness (the mac prebuilt's
    /// GPU accelerator is WebGPU and mis-executes big graphs).
    static func run(modelsRoot: URL, gpuAccel: Sam3Accel,
                    progress: @escaping (String) -> Void) throws -> String {
        let trackerDir = modelsRoot.appendingPathComponent("tracker")
        let manifestData = try Data(contentsOf: trackerDir.appendingPathComponent("expected/manifest.json"))
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              let frames = manifest["frames"] as? Int,
              let h = manifest["height"] as? Int,
              let w = manifest["width"] as? Int,
              let prompt = manifest["prompt"] as? String
        else { throw LiteRTError.interface("bad expected/manifest.json") }
        let px = h * w
        var sb = ""

        let t0 = Date()
        let tracker = try Sam3Tracker(modelsRoot: modelsRoot, prompt: prompt, gpuAccel: gpuAccel,
                                      log: { progress($0) })
        let compileMs = Int(Date().timeIntervalSince(t0) * 1000)
        progress("graphs compiled in \(compileMs)ms")
        sb += "compile \(compileMs)ms\n"

        let t1 = Date()
        let results = try tracker.track(clipDir: trackerDir.appendingPathComponent("clip")) {
            progress($0)
        }
        let trackMs = Int(Date().timeIntervalSince(t1) * 1000)
        guard tracker.vH == h && tracker.vW == w else {
            throw LiteRTError.interface("clip resolution \(tracker.vW)x\(tracker.vH) != manifest \(w)x\(h)")
        }

        var allIdsAgree = true
        var minIoU = 1.0
        var maxDp = 0.0
        for fi in 0..<frames {
            let expected = trackerDir.appendingPathComponent("expected")
            let refIds = try readInts(expected.appendingPathComponent("f\(fi)_ids.bin")).map(Int.init)
            let refProbs = try readFloats(expected.appendingPathComponent("f\(fi)_probs.bin"))
            let refMasks = try readMasks(expected.appendingPathComponent("f\(fi)_masks.bin"),
                                         refIds.count, px)
            let got = results[fi]
            let gotIds = got?.ids ?? []
            let same = refIds == gotIds
            allIdsAgree = allIdsAgree && same
            var ious: [Double] = []
            for j in 0..<refIds.count {
                guard let k = gotIds.firstIndex(of: refIds[j]) else { continue }
                let inter = Double(TM.popcountAnd(refMasks[j], got!.masks[k]))
                let union = Double(TM.popcountOr(refMasks[j], got!.masks[k]))
                let iou = inter / max(union, 1)
                ious.append(iou)
                minIoU = min(minIoU, iou)
            }
            var dp = 0.0
            if same {
                for j in 0..<refIds.count {
                    dp = max(dp, Double(abs(refProbs[j] - got!.probs[j])))
                }
                maxDp = max(maxDp, dp)
            }
            let iouStr = ious.map { String(format: "%.3f", $0) }.joined(separator: "/")
            let line = "f\(fi): ids ref=\(refIds) got=\(gotIds) same=\(same) "
                + String(format: "|dprob|=%.4f ", dp) + "IoU=\(iouStr)"
            progress(line)
            sb += line + "\n"
        }
        let stats = tracker.graphStats()
        progress(stats)
        let verdict = "TRACKER ids-agree=\(allIdsAgree) "
            + String(format: "minIoU=%.4f max|dprob|=%.4f ", minIoU, maxDp)
            + "total=\(trackMs)ms/\(frames) frames"
        progress(verdict)
        sb += stats + "\n" + verdict + "\n"
        try? (sb as NSString).write(
            to: modelsRoot.appendingPathComponent("tracker_result.txt"),
            atomically: true, encoding: String.Encoding.utf8.rawValue)
        return verdict
    }
}

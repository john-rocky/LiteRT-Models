import CoreGraphics
import Foundation
import ImageIO

/// SAM 3.1 Object-Multiplex video tracker: the host state machine ported 1:1 from
/// the Kotlin port (app/src/main/java/com/sam3/Sam3Tracker.kt), whose ground truth
/// is scripts/tracker_host_loop.py. Detection + quirk-exact NMS, association,
/// hotstart, occlusion suppression, recondition every 16 frames, memory bank with
/// temporal pos-enc, object pointers, mask-as-output init, masklet confirmation.
///
/// No UIKit here — CoreGraphics/ImageIO only, so the same file also compiles into
/// the macOS verification harness (where every graph runs on CPU; the mac prebuilt's
/// GPU accelerator is WebGPU and mis-dispatches big graphs).
///
/// v1 always uses the fixed N=7 memory-attention graph; the bank zero-pads to
/// 7 slots + keep mask, so swapping in trk_memattn_n{1..6} for small banks later
/// only needs a per-N graph map at the single memattn call site in
/// memoryConditionedFeatures().
final class Sam3Tracker {
    static let C = 256
    static let L = 5184  // 72*72
    static let MASK = 288
    static let IMG = 1008
    static let INMASK = 1152
    static let NO_OBJ_SCORE: Float = -1024
    static let PAD = -1
    static let REMOVED = -1116

    private let C = Sam3Tracker.C
    private let L = Sam3Tracker.L
    private let MASK = Sam3Tracker.MASK
    private let IMG = Sam3Tracker.IMG
    private let INMASK = Sam3Tracker.INMASK
    private let NO_OBJ_SCORE = Sam3Tracker.NO_OBJ_SCORE

    static let visLayout: [(String, Int, Int)] = [
        ("sam3_fpn288", 256, 288), ("sam3_fpn144", 256, 144), ("sam3_fpn72", 256, 72),
        ("inter_h0", 32, 288), ("inter_h1", 64, 144), ("inter_f2", 256, 72),
        ("prop_h0", 32, 288), ("prop_h1", 64, 144), ("prop_f2", 256, 72),
    ]

    /// "rss=... MB" of this process, for load-time memory tracing.
    static func procMem() -> String {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? "rss=\(Int(info.resident_size) / (1024 * 1024)) MB" : "rss=? MB"
    }

    private static func cpy(_ src: [Float], _ so: Int, _ dst: inout [Float], _ dOff: Int, _ n: Int) {
        src.withUnsafeBufferPointer { sp in
            dst.withUnsafeMutableBufferPointer { dp in
                (dp.baseAddress! + dOff).update(from: sp.baseAddress! + so, count: n)
            }
        }
    }

    private static func fill(_ dst: inout [Float], _ from: Int, _ to: Int, _ v: Float) {
        dst.withUnsafeMutableBufferPointer { dp in
            for i in from..<to { dp[i] = v }
        }
    }

    /// Timed wrapper over one compiled graph.
    private final class GraphT {
        let name: String
        let g: Sam3Graph
        var calls = 0
        var ms = 0.0

        init(_ url: URL, _ accel: Sam3Accel, perRunBuffers: Bool = false,
             log: (String) -> Void) throws {
            name = url.lastPathComponent
            g = try Sam3Graph(path: url.path, accel: accel, perRunBuffers: perRunBuffers)
            log("loaded \(name)  \(Sam3Tracker.procMem())")
        }

        func run(_ input: [Float]) throws -> [Float] {
            let t0 = Date()
            var x = input
            if x.count < g.inputCount {
                // trk_maskdec's export dummy was oversized (extra 16*16*256 instead of
                // 16*256); the tail is sliced off inside the graph. The python loop
                // leaves it zero via short writes into a zero-init buffer — match that.
                x.append(contentsOf: [Float](repeating: 0, count: g.inputCount - x.count))
            }
            let y = try g.run(x)
            ms += Date().timeIntervalSince(t0) * 1000
            calls += 1
            return y
        }
    }

    private let modelsRoot: URL
    private let trackerDir: URL
    private let log: (String) -> Void

    private func rootFile(_ name: String) throws -> URL {
        let f = modelsRoot.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: f.path) else {
            throw LiteRTError.interface("missing \(name) — stage the tracker models first")
        }
        return f
    }

    private func graphFile(_ name: String) throws -> URL {
        let g = trackerDir.appendingPathComponent("graphs/\(name)")
        return FileManager.default.fileExists(atPath: g.path) ? g : try rootFile(name)
    }

    let consts: TrackerConsts
    private let tokenTable: Data
    private let textMemPad: ([Float], [Float])
    private let vis: GraphT
    private let head: GraphT
    private let memattn: GraphT
    private let maskdec: GraphT
    private let memenc: GraphT
    private let initdec: GraphT
    private var graphs: [GraphT] { [vis, head, memattn, maskdec, memenc, initdec] }
    private var textMs = 0.0

    func graphStats() -> String {
        graphs.map { "\($0.name): \($0.calls)x/\(Int($0.ms))ms" }.joined(separator: "  ")
    }

    func graphSnapshot() -> [Double] { graphs.map { $0.ms } }

    func graphDelta(_ prev: [Double]) -> String {
        var parts: [String] = []
        let gs = graphs
        for i in 0..<gs.count where gs[i].ms > prev[i] + 0.5 {
            let stem = gs[i].name.replacingOccurrences(of: ".tflite", with: "")
            parts.append("\(stem)=\(Int(gs[i].ms - prev[i]))ms")
        }
        return parts.joined(separator: " ")
    }

    // ---------------- vision layout
    private var visOff: [String: Int] = [:]
    private var visLen: [String: Int] = [:]
    private let nVisHead = 256 * (288 * 288 + 144 * 144 + 72 * 72)

    // ---------------- flags
    private let scoreThresh: Float
    private let nmsThresh: Float
    private let assocIou: Float
    private let trkAssocIou: Float
    private let newDetThresh: Float
    private let iomRecond: Float
    private let oslThresh: Float
    private let hotstartDelay: Int
    private let hotstartUnmatch: Int
    private let hotstartDup: Int
    private let minKeepAlive: Int
    private let maxKeepAlive: Int
    private let initKeepAlive: Int
    private let occlThresh: Float
    private let recondEvery: Int
    private let maxObjects: Int
    private let maxCondFrames: Int
    private let maxObjPtrsFlag: Int
    private let muxCount: Int
    private let confThresh: Int
    private let sigScale: Float
    private let sigBias: Float
    private let condFg: Float
    private let condBg: Float
    private let suppressBoundary: Bool
    private let nonOverlapOut: Bool

    // ---------------- host constants, pre-shaped
    private var pos72Flat = [Float](repeating: 0, count: 5184 * 256)  // (5184, 256)
    private var tposEnc = [[Float]](repeating: [Float](repeating: 0, count: 256), count: 7)

    // ---------------- run state
    private var numFrames = 0
    private(set) var vH = 0
    private(set) var vW = 0
    private var states: [TrackState] = []
    private var objIdsAll: [Int] = []
    private var maxObjId = -1
    private var objIdToScore: [Int: Float] = [:]
    private var sam2ScoreFrame: [Int: [Int: Float]] = [:]
    private var removedObjIds = Set<Int>()
    private var confStatus: [Int] = []
    private var confCnt: [Int] = []
    private let hot = Hotstart()
    private var visY: [Float] = []
    private var curImageFeatures: [Float] = []  // (5184, 256)

    private final class Hotstart {
        var n = 0
        var firstFrame: [Int] = []
        var unmatchCnt: [Int] = []
        var keepAlive: [Int] = []
        var removed: [Bool] = []
        var lastOccl: [Int] = []
        var overlap: [[Int]] = []
    }

    private final class FrameEntry {
        var nRows = 0
        var predMasks: [Float] = []           // (nRows, 288*288)
        var osl: [Float] = []                 // (nRows)
        var objPtr: [Float] = []              // mux'd (nb, 16, 256)
        var conditioning = Set<Int>()
        var maskmem: [Float]?                 // (nb, 256, 72, 72) bf16-rounded
        var imageFeatures: [Float]?           // (5184, 256)
        var predMasksVideoRes: [Float]?       // (nRows, H*W)
    }

    private final class TrackState {
        var mux: MultiplexState?
        var objIdOrder: [Int] = []            // insertion order (LinkedHashMap keys)
        var objIdToIdx: [Int: Int] = [:]
        var objIds: [Int] { objIdOrder }
        var outputCond: [Int: FrameEntry] = [:]
        var outputNonCond: [Int: FrameEntry] = [:]
        var tempCond: [Int: [Int: [Float]]] = [:]     // objIdx -> frame -> (H*W)
        var tempNonCond: [Int: [Int: [Float]]] = [:]
        var framesTracked = Set<Int>()
        var consolidatedCond = Set<Int>()
        var consolidatedNonCond = Set<Int>()
        var curPropF2: [Float]?               // (256*5184)
    }

    private final class MultiplexState {
        var assignments: [[Int]]
        let capacity: Int
        var objectIds: [Int]
        var numBuckets = 0
        var muxCount = 0
        var totalValid = 0
        var totalNonPadding = 0
        var slotOf: [Int: (Int, Int)] = [:]

        init(assignments: [[Int]], capacity: Int, objectIds: [Int]) {
            self.assignments = assignments
            self.capacity = capacity
            self.objectIds = objectIds
            reinit()
        }

        func reinit() {
            numBuckets = assignments.count
            muxCount = assignments[0].count
            totalValid = assignments.reduce(0) { $0 + $1.filter { $0 >= 0 }.count }
            totalNonPadding = assignments.reduce(0) { $0 + $1.filter { $0 != Sam3Tracker.PAD }.count }
            slotOf.removeAll()
            for bi in 0..<assignments.count {
                for si in 0..<assignments[bi].count {
                    let o = assignments[bi][si]
                    if o >= 0 { slotOf[o] = (bi, si) }
                }
            }
        }

        var availableSlots: Int { numBuckets * capacity - totalNonPadding }

        func mux(_ x: [Float], _ item: Int) -> [Float] {
            var out = [Float](repeating: 0, count: numBuckets * muxCount * item)
            for (o, bs) in slotOf {
                Sam3Tracker.cpy(x, o * item, &out, (bs.0 * muxCount + bs.1) * item, item)
            }
            return out
        }

        func demux(_ x: [Float], _ item: Int) -> [Float] {
            var out = [Float](repeating: 0, count: totalValid * item)
            for (o, bs) in slotOf {
                Sam3Tracker.cpy(x, (bs.0 * muxCount + bs.1) * item, &out, o * item, item)
            }
            return out
        }

        func validMask() -> [Float] {
            var m = [Float](repeating: 0, count: numBuckets * muxCount)
            for (_, bs) in slotOf { m[bs.0 * muxCount + bs.1] = 1 }
            return m
        }

        func addObjects(_ objectIndices: [Int], _ ids: [Int]) {
            var remIdx = objectIndices
            var remIds = ids
            for b in 0..<assignments.count {
                for i in 0..<capacity {
                    if remIdx.isEmpty { break }
                    if assignments[b][i] == Sam3Tracker.PAD {
                        assignments[b][i] = remIdx.removeFirst()
                        objectIds.append(remIds.removeFirst())
                    }
                }
                if remIdx.isEmpty { break }
            }
            while !remIdx.isEmpty {
                var nb = [Int](repeating: Sam3Tracker.PAD, count: muxCount)
                for i in 0..<capacity {
                    if remIdx.isEmpty { break }
                    nb[i] = remIdx.removeFirst()
                    objectIds.append(remIds.removeFirst())
                }
                assignments.append(nb)
            }
            reinit()
        }

        func removeObjects(_ objectIndices: [Int]) {
            var rem = objectIndices
            for b in 0..<assignments.count {
                for si in 0..<assignments[b].count {
                    if let k = rem.firstIndex(of: assignments[b][si]) {
                        rem.remove(at: k)
                        assignments[b][si] = Sam3Tracker.REMOVED
                    }
                }
            }
            assignments = assignments.filter { b in
                !b.allSatisfy { $0 == Sam3Tracker.PAD || $0 == Sam3Tracker.REMOVED }
            }
            if assignments.isEmpty {
                objectIds = []
                return
            }
            let pos = Array(Set(assignments.flatMap { $0 }.filter { $0 >= 0 })).sorted()
            var remap: [Int: Int] = [:]
            for (new, old) in pos.enumerated() { remap[old] = new }
            for b in 0..<assignments.count {
                for i in 0..<assignments[b].count where assignments[b][i] >= 0 {
                    assignments[b][i] = remap[assignments[b][i]]!
                }
            }
            var newIds = [Int](repeating: 0, count: pos.count)
            for (old, new) in remap { newIds[new] = objectIds[old] }
            objectIds = newIds
            reinit()
        }
    }

    // ================================================================ init
    /// Compiles the graphs. The text encoder is XNNPACK-resident (~1.7 GB unpacked
    /// from the 606 MB fp16 file) and the tracker needs exactly ONE prompt encoding
    /// — run it before the GPU graphs are up and release it, or the process tops
    /// out the device's memory and gets killed (lesson from the Pixel port).
    init(modelsRoot: URL, prompt: String, gpuAccel: Sam3Accel,
         log: @escaping (String) -> Void) throws {
        self.modelsRoot = modelsRoot
        self.trackerDir = modelsRoot.appendingPathComponent("tracker")
        self.log = log
        consts = try TrackerConsts(trackerDir: trackerDir)
        tokenTable = try Data(
            contentsOf: modelsRoot.appendingPathComponent("sam3_token_embed.bin"),
            options: .alwaysMapped)

        scoreThresh = consts.flagFloat("score_threshold_detection")
        nmsThresh = consts.flagFloat("det_nms_thresh")
        assocIou = consts.flagFloat("assoc_iou_thresh")
        trkAssocIou = consts.flagFloat("trk_assoc_iou_thresh")
        newDetThresh = consts.flagFloat("new_det_thresh")
        iomRecond = consts.flagFloat("iom_thresh_recondition")
        oslThresh = consts.flagFloat("object_score_logit_threshold")
        hotstartDelay = consts.flagInt("hotstart_delay")
        hotstartUnmatch = consts.flagInt("hotstart_unmatch_thresh")
        hotstartDup = consts.flagInt("hotstart_dup_thresh")
        minKeepAlive = consts.flagInt("min_trk_keep_alive")
        maxKeepAlive = consts.flagInt("max_trk_keep_alive")
        initKeepAlive = consts.flagInt("init_trk_keep_alive")
        occlThresh = consts.flagFloat("suppress_overlap_recent_occl_thresh")
        recondEvery = consts.flagInt("recondition_every_nth_frame")
        maxObjects = consts.flagInt("max_num_objects")
        maxCondFrames = consts.flagInt("max_cond_frames_in_attn")
        maxObjPtrsFlag = consts.flagInt("max_obj_ptrs_in_encoder")
        muxCount = consts.flagInt("multiplex_count")
        confThresh = consts.flagInt("masklet_confirmation_consecutive_det_thresh")
        sigScale = consts.flagFloat("sigmoid_scale_for_mem_enc")
        sigBias = consts.flagFloat("sigmoid_bias_for_mem_enc")
        condFg = consts.flagFloat("condition_fg")
        condBg = consts.flagFloat("condition_bg")
        suppressBoundary = consts.flagBool("suppress_det_close_to_boundary")
        nonOverlapOut = consts.flagBool("non_overlap_masks_for_output")

        var o = 0
        for (name, c, hw) in Sam3Tracker.visLayout {
            visOff[name] = o
            visLen[name] = c * hw * hw
            o += c * hw * hw
        }
        let p = consts["pos_72"]
        for t in 0..<L { for c in 0..<C { pos72Flat[t * C + c] = p[c * L + t] } }
        let tp = consts["maskmem_tpos_enc"]
        for k in 0..<7 { for c in 0..<C { tposEnc[k][c] = tp[k * C + c] } }

        // text: encode once, release before the big graphs come up
        let tokenizer = try BpeTokenizer(
            vocabURL: modelsRoot.appendingPathComponent("vocab.json"),
            mergesURL: modelsRoot.appendingPathComponent("merges.txt"))
        let ids = tokenizer.encode(prompt)
        var emb = [Float](repeating: 0, count: 32 * 1024)
        tokenTable.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let table = raw.bindMemory(to: Float16.self)
            for t in 0..<32 {
                let base = ids[t] * 1024
                for d in 0..<1024 { emb[t * 1024 + d] = Float(table[base + d]) }
            }
        }
        var mem: [Float] = []
        do {
            let textPath = modelsRoot.appendingPathComponent("sam3_text.tflite")
            guard FileManager.default.fileExists(atPath: textPath.path) else {
                throw LiteRTError.interface("missing sam3_text.tflite")
            }
            let text = try GraphT(textPath, .cpu, log: log)
            mem = try text.run(emb)
        }
        log("text encoded + released  \(Sam3Tracker.procMem())")
        textMemPad = (mem, (0..<32).map { ids[$0] == 0 ? Float(1) : Float(0) })

        vis = try GraphT(try graphFileStatic(modelsRoot, trackerDir, "sam3_vision_tri.tflite"), gpuAccel, log: log)
        // Head on CPU: its GPU intermediates are the largest wired-memory consumer
        // after the trunk, and the device teeters at the system memory ceiling with
        // every graph resident. XNNPACK is fp32-exact and ~1-2 s on this device.
        head = try GraphT(try rootFileStatic(modelsRoot, "sam3_head.tflite"), .cpu, log: log)
        memattn = try GraphT(try graphFileStatic(modelsRoot, trackerDir, "trk_memattn_n7.tflite"), gpuAccel, log: log)
        maskdec = try GraphT(try graphFileStatic(modelsRoot, trackerDir, "trk_maskdec.tflite"), gpuAccel, log: log)
        memenc = try GraphT(try graphFileStatic(modelsRoot, trackerDir, "trk_memenc.tflite"), gpuAccel, log: log)
        initdec = try GraphT(try graphFileStatic(modelsRoot, trackerDir, "trk_initdec.tflite"), gpuAccel, perRunBuffers: true, log: log)
    }

    // ================================================================ frame loading
    /// JPEG -> RGB float 0..255 -> triangle-filter resize to 1008 (PIL BILINEAR
    /// shape; rounded back to uint8 like PIL) -> /255 -> fp16 -> normalize in fp16.
    func loadFrame(_ url: URL) throws -> [UInt16] {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw LiteRTError.interface("cannot decode \(url.lastPathComponent)") }
        let w = cg.width
        let h = cg.height
        vH = h
        vW = w
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        pixels.withUnsafeMutableBytes { raw in
            let ctx = CGContext(
                data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .none
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var chan = [Float](repeating: 0, count: 3 * h * w)
        pixels.withUnsafeBufferPointer { pp in
            chan.withUnsafeMutableBufferPointer { cp in
                for i in 0..<(w * h) {
                    cp[i] = Float(pp[i * 4])
                    cp[h * w + i] = Float(pp[i * 4 + 1])
                    cp[2 * h * w + i] = Float(pp[i * 4 + 2])
                }
            }
        }
        let rs = TM.interpBilinearAA(chan, 3, h, w, IMG, IMG)
        var out = [UInt16](repeating: 0, count: 3 * IMG * IMG)
        for i in 0..<rs.count {
            var v = Int((rs[i] + 0.5).rounded(.down))
            if v < 0 { v = 0 }
            if v > 255 { v = 255 }
            let h16 = TM.halfBitsToFloat(TM.toHalfBits(Float(v) / 255))
            let sub = TM.halfBitsToFloat(TM.toHalfBits(h16 - 0.5))
            out[i] = TM.toHalfBits(sub * 2)  // /0.5 is exact in fp16
        }
        return out
    }

    // ================================================================ graph fronts
    private func runVision(_ frame: [UInt16]) throws {
        var input = [Float](repeating: 0, count: frame.count)
        for i in 0..<frame.count { input[i] = TM.halfBitsToFloat(frame[i]) }
        visY = try vis.run(input)
        let f2 = visOff["prop_f2"]!
        curImageFeatures = [Float](repeating: 0, count: L * C)
        visY.withUnsafeBufferPointer { vp in
            curImageFeatures.withUnsafeMutableBufferPointer { cp in
                for t in 0..<L { for c in 0..<C { cp[t * C + c] = vp[f2 + c * L + t] } }
            }
        }
    }

    private func visSlice(_ name: String) -> [Float] {
        let o = visOff[name]!
        return Array(visY[o..<(o + visLen[name]!)])
    }

    private struct DetOut {
        let scores: [Float]      // (200), sorted per the deterministic perm
        let bboxXyxy: [Float]    // (200*4)
        let keep: [Bool]         // (200)
        let headY: [Float]       // full head output (mask rows live here)
        let perm: [Int]
        func maskOffset(_ row: Int) -> Int { 1001 + perm[row] * (288 * 288) }
    }

    private func runDetection(_ textMem: [Float], _ pad: [Float]) throws -> DetOut {
        var headIn = [Float](repeating: 0, count: nVisHead + 32 * 256 + 32)
        Sam3Tracker.cpy(visY, 0, &headIn, 0, nVisHead)
        Sam3Tracker.cpy(textMem, 0, &headIn, nVisHead, 32 * 256)
        Sam3Tracker.cpy(pad, 0, &headIn, nVisHead + 32 * 256, 32)
        let y = try head.run(headIn)

        let probs0 = (0..<200).map { TM.sigmoid(y[$0]) }
        let isValid = (0..<200).map { probs0[$0] > scoreThresh }
        let packed = (0..<200).map { q in TM.pack(y, 1001 + q * MASK * MASK, MASK * MASK, 0) }
        let area = (0..<200).map { Float(TM.popcount(packed[$0])) }

        // NMS with the perflib row-area-IoM quirk; suppressed rows still suppress.
        // Stable descending sort (torch stable argsort semantics).
        let order = (0..<200).sorted { a, b in
            probs0[a] != probs0[b] ? probs0[a] > probs0[b] : a < b
        }
        var keepS = (0..<200).map { isValid[order[$0]] }
        for i in 0..<200 {
            let qi = order[i]
            for j in (i + 1)..<200 {
                if !keepS[j] { continue }
                let qj = order[j]
                if Float(TM.popcountAnd(packed[qi], packed[qj])) / (area[qi] + 1e-8) > nmsThresh {
                    keepS[j] = false
                }
            }
        }
        var keepQ = [Bool](repeating: false, count: 200)
        for i in 0..<200 { keepQ[order[i]] = keepS[i] }

        let probs = (0..<200).map { TM.sigmoid(y[$0] - (keepQ[$0] ? 0 : 1e4)) }
        var pos = [Bool](repeating: false, count: 200)
        for q in 0..<200 {
            var p = probs[q] > scoreThresh
            if p && suppressBoundary {
                let cx = y[200 + q * 4]
                let cy = y[200 + q * 4 + 1]
                let bw = y[200 + q * 4 + 2]
                let bh = y[200 + q * 4 + 3]
                let xc = ((cx - bw / 2) + (cx + bw / 2)) / 2
                let yc = ((cy - bh / 2) + (cy + bh / 2)) / 2
                p = xc > 0.025 && xc < 0.975 && yc > 0.025 && yc < 0.975
            }
            pos[q] = p
        }
        // deterministic stand-in for torch's unstable bool argsort (see the spec doc)
        let t = (0..<200).filter { pos[$0] }
        var perm: [Int] = []
        perm.reserveCapacity(200)
        if !t.isEmpty {
            perm.append(t[0])
            perm.append(contentsOf: t.dropFirst().reversed())
        }
        perm.append(contentsOf: (0..<200).filter { !pos[$0] })

        let scores = (0..<200).map { probs[perm[$0]] }
        var bbox = [Float](repeating: 0, count: 800)
        for r in 0..<200 {
            let q = perm[r]
            let cx = y[200 + q * 4]
            let cy = y[200 + q * 4 + 1]
            let bw = y[200 + q * 4 + 2]
            let bh = y[200 + q * 4 + 3]
            bbox[r * 4] = cx - bw / 2
            bbox[r * 4 + 1] = cy - bh / 2
            bbox[r * 4 + 2] = cx + bw / 2
            bbox[r * 4 + 3] = cy + bh / 2
        }
        let keep = (0..<200).map { pos[perm[$0]] }
        return DetOut(scores: scores, bboxXyxy: bbox, keep: keep, headY: y, perm: perm)
    }

    /// det mask rows (float logits) -> packed binary at the given size.
    private func detMasksAt(_ det: DetOut, _ rows: [Int], _ size: Int) -> [[UInt64]] {
        rows.map { row in
            let o = det.maskOffset(row)
            if size == MASK {
                return TM.pack(det.headY, o, MASK * MASK, 0)
            }
            let up = TM.interpBilinear(Array(det.headY[o..<(o + MASK * MASK)]), 1, MASK, MASK, size, size)
            return TM.pack(up, 0, size * size, 0)
        }
    }

    // ================================================================ interactive init
    private func denseEmbed(_ maskF: [Float], _ n: Int) -> [Float] {
        var x = TM.convBlock(maskF, n, 1, INMASK, INMASK,
                             consts["interactive_mask_downsample.w"], 1, 4,
                             consts["interactive_mask_downsample.b"])
        x = TM.convBlock(x, n, 1, MASK, MASK,
                         consts["mask_downscaling.0.w"], 4, 2, consts["mask_downscaling.0.b"])
        TM.layerNorm2dInPlace(&x, n, 4, 144, 144, consts["mask_downscaling.1.w"], consts["mask_downscaling.1.b"])
        TM.geluInPlace(&x)
        x = TM.convBlock(x, n, 4, 144, 144,
                         consts["mask_downscaling.3.w"], 16, 2, consts["mask_downscaling.3.b"])
        TM.layerNorm2dInPlace(&x, n, 16, 72, 72, consts["mask_downscaling.4.w"], consts["mask_downscaling.4.b"])
        TM.geluInPlace(&x)
        let w6 = consts["mask_downscaling.6.w"]
        let b6 = consts["mask_downscaling.6.b"]
        var out = [Float](repeating: 0, count: n * C * L)
        x.withUnsafeBufferPointer { xp in
            out.withUnsafeMutableBufferPointer { op in
                for b in 0..<n {
                    for o in 0..<C {
                        let ob = (b * C + o) * L
                        for p in 0..<L {
                            var acc: Float = 0
                            for c in 0..<16 { acc += xp[(b * 16 + c) * L + p] * w6[o * 16 + c] }
                            op[ob + p] = acc + b6[o]
                        }
                    }
                }
            }
        }
        return out
    }

    private struct MaskAsOutput {
        let lowRes: [Float]
        let osl: [Float]
        let objPtr: [Float]
    }

    private func useMaskAsOutput(_ masks1152: [[UInt64]]) throws -> MaskAsOutput {
        let n = masks1152.count
        let px = INMASK * INMASK
        var maskF = [Float](repeating: 0, count: n * px)
        for i in 0..<n {
            for p in 0..<px where TM.testBit(masks1152[i], p) { maskF[i * px + p] = 1 }
        }
        let highRes = maskF.map { $0 * 20 - 10 }
        let lowRes = TM.interpBilinearAA(highRes, n, INMASK, INMASK, MASK, MASK)
        let interF2 = visSlice("inter_f2")
        let noMem = consts["interactivity_no_mem_embed"]
        let dense = denseEmbed(maskF, n)
        let h0 = visSlice("inter_h0")
        let h1 = visSlice("inter_h1")
        var input = [Float](repeating: 0, count: C * L + h0.count + h1.count + 512 + C * L)
        for c in 0..<C {
            let add = noMem[c]
            for p in 0..<L { input[c * L + p] = interF2[c * L + p] + add }
        }
        Sam3Tracker.cpy(h0, 0, &input, C * L, h0.count)
        Sam3Tracker.cpy(h1, 0, &input, C * L + h0.count, h1.count)
        Sam3Tracker.cpy(consts["sparse_const"], 0, &input, C * L + h0.count + h1.count, 512)
        let denseOff = C * L + h0.count + h1.count + 512
        var tokens = [Float](repeating: 0, count: n * 256)
        var oslG = [Float](repeating: 0, count: n)
        for i in 0..<n {
            Sam3Tracker.cpy(dense, i * C * L, &input, denseOff, C * L)
            let y = try initdec.run(input)
            let o = MASK * MASK
            Sam3Tracker.cpy(y, o + 1, &tokens, i * 256, 256)
            oslG[i] = y[o + 257]
        }
        var ptr = consts.mlp3(tokens, n, "interactive_obj_ptr_proj")
        ptr = consts.noObjPtrBlend(ptr, n, (0..<n).map { oslG[$0] > oslThresh ? Float(1) : Float(0) })
        let lamM = (0..<n).map { TM.popcount(masks1152[$0]) > 0 ? Float(1) : Float(0) }
        let osl = (0..<n).map { 20 * lamM[$0] - 10 }
        ptr = consts.noObjPtrBlend(ptr, n, lamM)
        return MaskAsOutput(lowRes: lowRes, osl: osl, objPtr: ptr)
    }

    // ================================================================ propagation
    private func selectCond(_ cond: [Int: FrameEntry], _ frameIdx: Int)
        -> (selected: [(Int, FrameEntry)], unselected: [Int: FrameEntry]) {
        var selected: [(Int, FrameEntry)] = []
        if maxCondFrames == -1 || cond.count <= maxCondFrames {
            for t in cond.keys.sorted() { selected.append((t, cond[t]!)) }
            return (selected, [:])
        }
        var selKeys = Set<Int>()
        if let before = cond.keys.filter({ $0 < frameIdx }).max() {
            selected.append((before, cond[before]!))
            selKeys.insert(before)
        }
        if let after = cond.keys.filter({ $0 >= frameIdx }).min(), !selKeys.contains(after) {
            selected.append((after, cond[after]!))
            selKeys.insert(after)
        }
        let rest = cond.keys.filter { !selKeys.contains($0) }
            .sorted { abs($0 - frameIdx) != abs($1 - frameIdx)
                ? abs($0 - frameIdx) < abs($1 - frameIdx) : $0 < $1 }
            .prefix(maxCondFrames - selected.count)
        for t in rest {
            selected.append((t, cond[t]!))
            selKeys.insert(t)
        }
        var unselected: [Int: FrameEntry] = [:]
        for (t, v) in cond where !selKeys.contains(t) { unselected[t] = v }
        return (selected, unselected)
    }

    /// Conditioned pix features per bucket: (nb, 256*5184).
    private func memoryConditionedFeatures(_ frameIdx: Int, _ st: TrackState) throws -> [Float] {
        let (selected, unselected) = selectCond(st.outputCond, frameIdx)
        var slotTpos: [Int] = []
        var slotEntry: [FrameEntry] = []
        for (t, e) in selected {
            slotTpos.append(frameIdx - t)
            slotEntry.append(e)
        }
        for tPos in 1..<7 {
            if let e = st.outputNonCond[frameIdx - (7 - tPos)] ?? unselected[frameIdx - (7 - tPos)] {
                slotTpos.append(tPos)
                slotEntry.append(e)
            }
        }
        let validIdx = slotEntry.indices.filter { slotEntry[$0].maskmem != nil }
        let mux = st.mux!
        let nb = mux.numBuckets
        if validIdx.isEmpty {
            let f2 = st.curPropF2!
            var out = [Float](repeating: 0, count: nb * C * L)
            for b in 0..<nb { Sam3Tracker.cpy(f2, 0, &out, b * C * L, C * L) }
            return out
        }

        let maxPtr = min(numFrames, maxObjPtrsFlag)
        var ptrDist: [Int] = []
        var ptrEntry: [FrameEntry] = []
        for (t, e) in selected {
            ptrDist.append(frameIdx - t)
            ptrEntry.append(e)
        }
        for tDiff in 1..<maxPtr {
            let t = frameIdx - tDiff
            if t < 0 { break }
            if let e = st.outputNonCond[t] ?? unselected[t] {
                ptrDist.append(tDiff)
                ptrEntry.append(e)
            }
        }
        let p = ptrDist.count
        let p16 = p * 16

        let nTok = validIdx.count * L
        // transient: ~117 MB, freed after the frame
        var memattnIn = [Float](repeating: 0, count: L * C + 3 * 7 * L * C + 2 * 256 * C + (7 * L + 256))
        let miOff = L * C
        let mipOff = miOff + 7 * L * C
        let mmOff = mipOff + 7 * L * C
        let ptrOff = mmOff + 7 * L * C
        let ptrPosOff = ptrOff + 256 * C
        let keepOff = ptrPosOff + 256 * C

        Sam3Tracker.cpy(curImageFeatures, 0, &memattnIn, 0, L * C)  // pix = cur_prop_f2 (5184,256)
        for (si, vi) in validIdx.enumerated() {
            let tPos = slotTpos[vi]
            let te = (tPos <= 0 || tPos >= 7) ? tposEnc[6] : tposEnc[7 - tPos - 1]
            Sam3Tracker.cpy(slotEntry[vi].imageFeatures!, 0, &memattnIn, miOff + si * L * C, L * C)
            let b1 = mipOff + si * L * C
            pos72Flat.withUnsafeBufferPointer { pp in
                memattnIn.withUnsafeMutableBufferPointer { mp in
                    for t in 0..<L { for c in 0..<C { mp[b1 + t * C + c] = pp[t * C + c] + te[c] } }
                }
            }
        }
        Sam3Tracker.fill(&memattnIn, keepOff, keepOff + nTok, 1)
        Sam3Tracker.fill(&memattnIn, keepOff + 7 * L, keepOff + 7 * L + p16, 1)
        if p > 0 {
            var objPos = TM.sine1dPe((0..<p).map { Float(ptrDist[$0]) / Float(maxPtr - 1) }, C)
            objPos = TM.linear(objPos, p, C, consts["obj_ptr_tpos_proj.w"], C, consts["obj_ptr_tpos_proj.b"])
            for i in 0..<p {
                for rep in 0..<16 {
                    Sam3Tracker.cpy(objPos, i * C, &memattnIn, ptrPosOff + (i * 16 + rep) * C, C)
                }
            }
        }

        var out = [Float](repeating: 0, count: nb * C * L)
        for b in 0..<nb {
            for (si, vi) in validIdx.enumerated() {
                // maskmem features (nb,256,72,72) -> tokens (5184,256) of bucket b
                let mm = slotEntry[vi].maskmem!
                let base = mmOff + si * L * C
                let mb = b * C * L
                mm.withUnsafeBufferPointer { mp in
                    memattnIn.withUnsafeMutableBufferPointer { ip in
                        for t in 0..<L { for c in 0..<C { ip[base + t * C + c] = mp[mb + c * L + t] } }
                    }
                }
            }
            for (pi, e) in ptrEntry.enumerated() {
                // obj_ptr mux'd (nb,16,256): bucket b's 16 slots
                Sam3Tracker.cpy(e.objPtr, b * 16 * C, &memattnIn, ptrOff + pi * 16 * C, 16 * C)
            }
            let y = try memattn.run(memattnIn)  // (5184, 256)
            y.withUnsafeBufferPointer { yp in
                out.withUnsafeMutableBufferPointer { op in
                    for t in 0..<L { for c in 0..<C { op[b * C * L + c * L + t] = yp[t * C + c] } }
                }
            }
        }
        return out
    }

    private struct SamHeadsOut {
        let low: [Float]
        let osl: [Float]
        let objPtr: [Float]
    }

    /// pixWithMem (nb, 256*5184) -> per-object best masks (nV,288*288), osl, obj_ptr.
    private func forwardSamHeadsProp(_ pixWithMem: [Float], _ st: TrackState) throws -> SamHeadsOut {
        let mux = st.mux!
        let valid = mux.validMask()
        let validE = consts["output_valid_embed"]
        let invalidE = consts["output_invalid_embed"]
        let h0 = visSlice("prop_h0")
        let h1 = visSlice("prop_h1")
        var input = [Float](repeating: 0, count: C * L + h0.count + h1.count + 16 * 256)
        Sam3Tracker.cpy(h0, 0, &input, C * L, h0.count)
        Sam3Tracker.cpy(h1, 0, &input, C * L + h0.count, h1.count)
        let mergedOff = C * L + h0.count + h1.count
        let nb = mux.numBuckets
        var masksB = [Float](repeating: 0, count: nb * 16 * 3 * MASK * MASK)
        var iousB = [Float](repeating: 0, count: nb * 16 * 3)
        var oslB = [Float](repeating: 0, count: nb * 16)
        var tokB = [Float](repeating: 0, count: nb * 16 * 3 * 256)
        for b in 0..<nb {
            Sam3Tracker.cpy(pixWithMem, b * C * L, &input, 0, C * L)
            for s in 0..<16 {
                let v = valid[b * 16 + s]
                for c in 0..<256 {
                    input[mergedOff + s * 256 + c] =
                        v * validE[s * 256 + c] + (1 - v) * invalidE[s * 256 + c]
                }
            }
            let y = try maskdec.run(input)
            let o = 16 * 3 * MASK * MASK
            Sam3Tracker.cpy(y, 0, &masksB, b * o, o)
            Sam3Tracker.cpy(y, o, &iousB, b * 48, 48)
            Sam3Tracker.cpy(y, o + 48, &oslB, b * 16, 16)
            Sam3Tracker.cpy(y, o + 64, &tokB, b * 16 * 3 * 256, 16 * 3 * 256)
        }
        let nV = mux.totalValid
        let lowMulti = mux.demux(masksB, 3 * MASK * MASK)
        let ious = mux.demux(iousB, 3)
        let osl = mux.demux(oslB, 1)
        let tokens = mux.demux(tokB, 3 * 256)
        var low = [Float](repeating: 0, count: nV * MASK * MASK)
        var token = [Float](repeating: 0, count: nV * 256)
        for r in 0..<nV {
            let isObj = osl[r] > oslThresh
            var best = 0
            for k in 1..<3 where ious[r * 3 + k] > ious[r * 3 + best] { best = k }
            let src = (r * 3 + best) * MASK * MASK
            if isObj {
                Sam3Tracker.cpy(lowMulti, src, &low, r * MASK * MASK, MASK * MASK)
            } else {
                Sam3Tracker.fill(&low, r * MASK * MASK, (r + 1) * MASK * MASK, NO_OBJ_SCORE)
            }
            Sam3Tracker.cpy(tokens, (r * 3 + best) * 256, &token, r * 256, 256)
        }
        var ptr = consts.mlp3(token, nV, "obj_ptr_proj")
        ptr = consts.noObjPtrBlend(ptr, nV, (0..<nV).map { osl[$0] > oslThresh ? Float(1) : Float(0) })
        return SamHeadsOut(low: low, osl: osl, objPtr: ptr)
    }

    /// masksHigh (n, hs*hs) float logits at hs=1008 or 1152; returns maskmem
    /// (nb, 256*5184). Resizes each mux slot channel directly into the graph input.
    private func encodeNewMemory(_ propF2: [Float], _ masksHigh: [Float], _ hs: Int,
                                 _ osl: [Float], _ condObjs: Set<Int>,
                                 _ mux: MultiplexState) throws -> [Float] {
        let n = osl.count
        let px = hs * hs
        var maskForMem = [Float](repeating: 0, count: n * px)
        for i in 0..<(n * px) { maskForMem[i] = TM.sigmoid(masksHigh[i]) * sigScale + sigBias }
        var condVals = [Float](repeating: condBg, count: n)
        for o in condObjs where o < n { condVals[o] = condFg }
        let nb = mux.numBuckets
        var out = [Float](repeating: 0, count: nb * C * L)
        var memencIn = [Float](repeating: 0, count: C * L + 32 * IMG * IMG)  // transient, ~135 MB
        for b in 0..<nb {
            Sam3Tracker.fill(&memencIn, 0, memencIn.count, 0)
            Sam3Tracker.cpy(propF2, 0, &memencIn, 0, C * L)
            for s in 0..<16 {
                let o = mux.assignments[b][s]
                if o < 0 { continue }
                let chOff = C * L + s * IMG * IMG
                if hs == IMG {
                    Sam3Tracker.cpy(maskForMem, o * px, &memencIn, chOff, IMG * IMG)
                } else {
                    let rs = TM.interpBilinear(
                        Array(maskForMem[(o * px)..<((o + 1) * px)]), 1, hs, hs, IMG, IMG)
                    Sam3Tracker.cpy(rs, 0, &memencIn, chOff, IMG * IMG)
                }
                // condition channel: constant map (bilinear of a constant is the constant)
                Sam3Tracker.fill(&memencIn, C * L + (16 + s) * IMG * IMG,
                                 C * L + (17 + s) * IMG * IMG, condVals[o])
            }
            let y = try memenc.run(memencIn)  // (256, 72, 72)
            Sam3Tracker.cpy(y, 0, &out, b * C * L, C * L)
        }
        // += sum over empty slots of no_obj_embed_spatial
        let noObj = consts["no_obj_embed_spatial"]  // (16, 256)
        var oslMuxFull = [Float](repeating: 0, count: mux.totalValid)
        for i in 0..<min(n, mux.totalValid) { oslMuxFull[i] = osl[i] }
        let oslMux = mux.mux(oslMuxFull, 1)
        out.withUnsafeMutableBufferPointer { op in
            for b in 0..<nb {
                for s in 0..<16 {
                    if oslMux[b * 16 + s] > oslThresh { continue }
                    for c in 0..<C {
                        let add = noObj[s * 256 + c]
                        let base = b * C * L + c * L
                        for t in 0..<L { op[base + t] += add }
                    }
                }
            }
        }
        return out
    }

    // ================================================================ mask utilities
    /// masks (n, hw) float, per-pixel argmax keeps its object, losers min(x,-10).
    private func applyNonOverlapping(_ masks: inout [Float], _ n: Int, _ hw: Int) {
        if n <= 1 { return }
        masks.withUnsafeMutableBufferPointer { mp in
            for p in 0..<hw {
                var arg = 0
                for i in 1..<n where mp[i * hw + p] > mp[arg * hw + p] { arg = i }
                for i in 0..<n where i != arg && mp[i * hw + p] > -10 { mp[i * hw + p] = -10 }
            }
        }
    }

    /// Kill (min -10) objects whose pixel-argmax non-overlapped area shrinks below
    /// 0.3x; the non-overlap pass is used ONLY for the ratio — surviving objects
    /// keep their ORIGINAL mask (reference quirk).
    private func suppressPwAreaShrinkage(_ masks: inout [Float], _ n: Int, _ hw: Int) {
        if n <= 1 { return }
        var areaBefore = [Int](repeating: 0, count: n)
        for i in 0..<n {
            for p in 0..<hw where masks[i * hw + p] > 0 { areaBefore[i] += 1 }
        }
        var pw = masks
        applyNonOverlapping(&pw, n, hw)
        for i in 0..<n {
            var after = 0
            for p in 0..<hw where pw[i * hw + p] > 0 { after += 1 }
            if Float(after) / Float(max(areaBefore[i], 1)) < 0.3 {
                for p in 0..<hw where masks[i * hw + p] > -10 { masks[i * hw + p] = -10 }
            }
        }
    }

    /// pred (n, 288*288) -> video res (n, H*W) with the output non-overlap rule.
    private func videoResOutput(_ pred: [Float], _ n: Int) -> [Float] {
        var v = TM.interpBilinear(pred, n, MASK, MASK, vH, vW)
        if nonOverlapOut { applyNonOverlapping(&v, n, vH * vW) }
        return v
    }

    // ================================================================ SAM2-state ops
    private func addNewMasks(_ st: TrackState, _ frameIdx: Int, _ objIds: [Int],
                             _ masks1152: [[UInt64]], reconditioning: Bool) throws {
        let n = masks1152.count
        var objIdxs: [Int] = []
        for oid in objIds {
            if let existing = st.objIdToIdx[oid] {
                objIdxs.append(existing)
            } else {
                precondition(!reconditioning)
                let idx = st.objIdToIdx.count
                st.objIdToIdx[oid] = idx
                st.objIdOrder.append(oid)
                objIdxs.append(idx)
            }
        }
        // video-res binary via antialiased resize of the 1152 binary mask
        let px = INMASK * INMASK
        var maskF = [Float](repeating: 0, count: n * px)
        for i in 0..<n {
            for p in 0..<px where TM.testBit(masks1152[i], p) { maskF[i * px + p] = 1 }
        }
        let videoF = TM.interpBilinearAA(maskF, n, INMASK, INMASK, vH, vW)
        let video: [[Bool]] = (0..<n).map { i in
            (0..<(vH * vW)).map { p in videoF[i * vH * vW + p] > 0.5 }
        }
        let isNewState = st.mux == nil
        if !reconditioning && isNewState {
            let cap = muxCount
            let nb = (n + cap - 1) / cap
            var assignments: [[Int]] = []
            for b in 0..<nb {
                assignments.append((0..<cap).map { i in
                    let v = b * cap + i
                    return v < n ? v : Sam3Tracker.PAD
                })
            }
            st.mux = MultiplexState(assignments: assignments, capacity: cap, objectIds: objIds)
        }
        let isCond = !st.framesTracked.contains(frameIdx)

        let maskOut = try useMaskAsOutput(masks1152)
        let current: FrameEntry
        if reconditioning || !isNewState {
            let existing = st.outputCond[frameIdx] ?? st.outputNonCond[frameIdx]!
            let low = maskOut.lowRes  // (n, 288*288), same size as stored
            if reconditioning {
                for j in 0..<objIdxs.count {
                    let oi = objIdxs[j]
                    Sam3Tracker.cpy(low, j * MASK * MASK, &existing.predMasks, oi * MASK * MASK, MASK * MASK)
                    existing.osl[oi] = maskOut.osl[j]
                }
                var ptr = st.mux!.demux(existing.objPtr, 256)
                for j in 0..<objIdxs.count {
                    Sam3Tracker.cpy(maskOut.objPtr, j * 256, &ptr, objIdxs[j] * 256, 256)
                }
                existing.objPtr = st.mux!.mux(ptr, 256)
                existing.conditioning.formUnion(objIdxs)
            } else {
                let mux = st.mux!
                let oldPtr = mux.demux(existing.objPtr, 256)
                let start = mux.totalValid
                mux.addObjects(Array(start..<(start + n)), objIds)
                var pm = existing.predMasks
                pm.append(contentsOf: [Float](repeating: 0, count: (start + n) * MASK * MASK - pm.count))
                Sam3Tracker.cpy(low, 0, &pm, start * MASK * MASK, n * MASK * MASK)
                existing.predMasks = pm
                var os = existing.osl
                os.append(contentsOf: [Float](repeating: 0, count: start + n - os.count))
                for i in 0..<n { os[start + i] = maskOut.osl[i] }
                existing.osl = os
                var allPtr = oldPtr
                allPtr.append(contentsOf: [Float](repeating: 0, count: (start + n) * 256 - allPtr.count))
                Sam3Tracker.cpy(maskOut.objPtr, 0, &allPtr, start * 256, n * 256)
                existing.objPtr = mux.mux(allPtr, 256)
                existing.nRows = start + n
                existing.conditioning.formUnion(start..<(start + n))
            }
            current = existing
            var vres = videoResOutput(existing.predMasks, existing.nRows)
            for j in 0..<objIdxs.count {
                let oi = objIdxs[j]
                for p in 0..<(vH * vW) {
                    vres[oi * vH * vW + p] = video[j][p] ? -NO_OBJ_SCORE : NO_OBJ_SCORE
                }
            }
            current.predMasksVideoRes = vres
        } else {
            current = FrameEntry()
            current.nRows = n
            current.predMasks = maskOut.lowRes
            current.osl = maskOut.osl
            current.objPtr = st.mux!.mux(maskOut.objPtr, 256)
            current.conditioning.formUnion(objIdxs)
            current.imageFeatures = curImageFeatures
            var vres = videoResOutput(current.predMasks, n)
            for j in 0..<objIdxs.count {
                let oi = objIdxs[j]
                for p in 0..<(vH * vW) {
                    vres[oi * vH * vW + p] = video[j][p] ? -NO_OBJ_SCORE : NO_OBJ_SCORE
                }
            }
            current.predMasksVideoRes = vres
        }

        if isCond && st.outputNonCond[frameIdx] != nil {
            st.outputNonCond.removeValue(forKey: frameIdx)
            st.consolidatedNonCond.remove(frameIdx)
        }
        if isCond {
            st.outputCond[frameIdx] = current
            st.consolidatedCond.insert(frameIdx)
        } else {
            st.outputNonCond[frameIdx] = current
            st.consolidatedNonCond.insert(frameIdx)
        }

        // per-object temp entries (video res) with cross-suppression among the new masks
        var combined = [Bool](repeating: false, count: vH * vW)
        for j in 0..<n {
            for p in 0..<(vH * vW) where video[j][p] { combined[p] = true }
        }
        for j in 0..<objIdxs.count {
            var m = [Float](repeating: 0, count: vH * vW)
            for p in 0..<(vH * vW) {
                m[p] = video[j][p] ? -NO_OBJ_SCORE : NO_OBJ_SCORE
                if n > 1 {
                    var others = false
                    for k in 0..<n where k != j && video[k][p] {
                        others = true
                        break
                    }
                    if others { m[p] = NO_OBJ_SCORE }
                }
            }
            if isCond {
                st.tempCond[objIdxs[j], default: [:]][frameIdx] = m
            } else {
                st.tempNonCond[objIdxs[j], default: [:]][frameIdx] = m
            }
        }
        let tstoreKeys = isCond ? Array(st.tempCond.keys) : Array(st.tempNonCond.keys)
        for oi2 in tstoreKeys {
            if objIdxs.contains(oi2) { continue }
            if isCond {
                guard var m = st.tempCond[oi2]?[frameIdx] else { continue }
                for p in 0..<(vH * vW) where combined[p] { m[p] = NO_OBJ_SCORE }
                st.tempCond[oi2]![frameIdx] = m
            } else {
                guard var m = st.tempNonCond[oi2]?[frameIdx] else { continue }
                for p in 0..<(vH * vW) where combined[p] { m[p] = NO_OBJ_SCORE }
                st.tempNonCond[oi2]![frameIdx] = m
            }
        }
    }

    private func preflight(_ st: TrackState) throws {
        let nobj = st.mux!.totalValid
        for isCond in [false, true] {
            let tstore = isCond ? st.tempCond : st.tempNonCond
            var frames = Set<Int>()
            for d in tstore.values { frames.formUnion(d.keys) }
            if isCond {
                st.consolidatedCond.formUnion(frames)
            } else {
                st.consolidatedNonCond.formUnion(frames)
            }
            for f in frames.sorted() {
                let allOut = st.outputCond[f] ?? st.outputNonCond[f]!
                // cons rows: temp entries (aa-resized) where present, stored 288 rows
                // otherwise — the python base-resize is dead (all rows overwritten).
                var cons = allOut.predMasks
                if cons.count < nobj * MASK * MASK {
                    cons.append(contentsOf: [Float](repeating: 0, count: nobj * MASK * MASK - cons.count))
                } else if cons.count > nobj * MASK * MASK {
                    cons = Array(cons[0..<(nobj * MASK * MASK)])
                }
                for oi in 0..<nobj {
                    guard let src = tstore[oi]?[f] else { continue }
                    let rs = TM.interpBilinearAA(src, 1, vH, vW, MASK, MASK)
                    Sam3Tracker.cpy(rs, 0, &cons, oi * MASK * MASK, MASK * MASK)
                }
                var high = TM.interpBilinear(cons, nobj, MASK, MASK, IMG, IMG)
                applyNonOverlapping(&high, nobj, IMG * IMG)
                var featsMem = try encodeNewMemory(st.curPropF2!, high, IMG, allOut.osl,
                                                   allOut.conditioning, st.mux!)
                TM.bf16InPlace(&featsMem)
                let e = FrameEntry()
                e.nRows = nobj
                e.predMasks = cons
                e.osl = allOut.osl
                e.objPtr = allOut.objPtr
                e.conditioning = allOut.conditioning
                e.maskmem = featsMem
                e.imageFeatures = curImageFeatures
                if isCond {
                    st.outputCond[f] = e
                } else {
                    st.outputNonCond[f] = e
                }
            }
            if isCond {
                for k in st.tempCond.keys { st.tempCond[k] = [:] }
            } else {
                for k in st.tempNonCond.keys { st.tempNonCond[k] = [:] }
            }
        }
        for f in st.outputCond.keys {
            st.outputNonCond.removeValue(forKey: f)
            st.consolidatedNonCond.remove(f)
        }
    }

    private struct PropOut {
        let ids: [Int]
        let masks: [Float]
        let scores: [Float]
    }

    private func propagateStateOneFrame(_ st: TrackState, _ frameIdx: Int) throws -> PropOut {
        let cur: FrameEntry
        if st.consolidatedCond.contains(frameIdx) {
            cur = st.outputCond[frameIdx]!
        } else if st.consolidatedNonCond.contains(frameIdx) {
            cur = st.outputNonCond[frameIdx]!
        } else {
            let pix = try memoryConditionedFeatures(frameIdx, st)
            let out = try forwardSamHeadsProp(pix, st)
            cur = FrameEntry()
            cur.nRows = out.osl.count
            cur.predMasks = out.low
            cur.osl = out.osl
            cur.objPtr = st.mux!.mux(out.objPtr, 256)
            cur.imageFeatures = curImageFeatures
            st.outputNonCond[frameIdx] = cur
        }
        st.framesTracked.insert(frameIdx)
        return PropOut(ids: st.objIds,
                       masks: Array(cur.predMasks[0..<(cur.nRows * MASK * MASK)]),
                       scores: cur.osl)
    }

    // ================================================================ planning
    private struct Adt {
        let trkIsUnmatched: [Bool]
        let isNewDet: [Bool]
        let imMask: [[Bool]]                 // (200, N)
        let hiConfKeys: [Int]                // trk obj id, insertion order
        let hiConf: [Int: Int]               // trk obj id -> det row
        let detMatched: [Int: [Int]]         // det row -> matched trk ids
    }

    private func associate(_ det: DetOut, _ trkMasks: [Float], _ trkObjIds: [Int]) -> Adt {
        let nTrk = trkObjIds.count
        if nTrk == 0 {
            // NOTE: the reference does NOT gate on `keep` in the empty-track branch.
            let isNew = (0..<200).map { det.scores[$0] >= newDetThresh }
            return Adt(trkIsUnmatched: [], isNewDet: isNew,
                       imMask: [[Bool]](repeating: [], count: 200),
                       hiConfKeys: [], hiConf: [:], detMatched: [:])
        }
        let emptyWords = [UInt64](repeating: 0, count: TM.packedWords(MASK * MASK))
        let detPacked = (0..<200).map { r in
            det.keep[r] ? TM.pack(det.headY, det.maskOffset(r), MASK * MASK, 0) : emptyWords
        }
        let detArea = (0..<200).map { Float(TM.popcount(detPacked[$0])) }
        let trkPacked = (0..<nTrk).map { TM.pack(trkMasks, $0 * MASK * MASK, MASK * MASK, 0) }
        let trkArea = (0..<nTrk).map { Float(TM.popcount(trkPacked[$0])) }
        let metric: [[Float]] = (0..<200).map { d in
            (0..<nTrk).map { t in
                Float(TM.popcountAnd(detPacked[d], trkPacked[t])) / (min(detArea[d], trkArea[t]) + 1e-8)
            }
        }
        var trkIsMatched = [Bool](repeating: false, count: nTrk)
        for d in 0..<200 {
            for t in 0..<nTrk where metric[d][t] >= trkAssocIou { trkIsMatched[t] = true }
        }
        let trkIsUnmatched = (0..<nTrk).map { trkArea[$0] > 0 && !trkIsMatched[$0] }
        let isNew = (0..<200).map { d in
            det.scores[d] >= newDetThresh && det.keep[d]
                && !(0..<nTrk).contains { metric[d][$0] >= assocIou }
        }
        let detMany = (0..<200).map { d in (0..<nTrk).filter { metric[d][$0] >= iomRecond }.count > 1 }
        let trkMany = (0..<nTrk).map { t in (0..<200).filter { metric[$0][t] >= iomRecond }.count > 1 }
        let metricZ: [[Float]] = (0..<200).map { d in
            (0..<nTrk).map { t in (trkMany[t] || detMany[d]) ? 0 : metric[d][t] }
        }
        let imMask = (0..<200).map { d in (0..<nTrk).map { metricZ[d][$0] >= assocIou } }
        var hiConfKeys: [Int] = []
        var hiConf: [Int: Int] = [:]
        var detMatched: [Int: [Int]] = [:]
        for d in 0..<200 {
            if !det.keep[d] { continue }
            detMatched[d] = (0..<nTrk).filter { imMask[d][$0] }.map { trkObjIds[$0] }
            var arg = 0
            var mx = metricZ[d][0]
            for t in 1..<nTrk where metricZ[d][t] > mx {
                mx = metricZ[d][t]
                arg = t
            }
            if det.scores[d] >= 0.8 && !isNew[d] && mx >= iomRecond {
                let key = trkObjIds[arg]
                if hiConf[key] == nil { hiConfKeys.append(key) }  // LinkedHashMap: keep position
                hiConf[key] = d
            }
        }
        return Adt(trkIsUnmatched: trkIsUnmatched, isNewDet: isNew, imMask: imMask,
                   hiConfKeys: hiConfKeys, hiConf: hiConf, detMatched: detMatched)
    }

    private func processHotstart(_ frameIdx: Int, _ adt: Adt) -> [Bool] {
        let n = adt.imMask.isEmpty ? 0 : adt.imMask[0].count
        if n == 0 { return [] }
        precondition(hot.n == n, "hotstart N mismatch: \(hot.n) vs \(n)")
        var matched = [Bool](repeating: false, count: n)
        for d in 0..<200 {
            for t in 0..<n where adt.imMask[d][t] { matched[t] = true }
        }
        for t in 0..<n {
            hot.keepAlive[t] = min(max(hot.keepAlive[t] + (matched[t] ? 1 : -1), minKeepAlive), maxKeepAlive)
            if adt.trkIsUnmatched[t] { hot.unmatchCnt[t] += 1 }
        }
        for d in 0..<200 {
            var cnt = 0
            for t in 0..<n where adt.imMask[d][t] { cnt += 1 }
            if cnt <= 1 { continue }
            for i in 0..<n where adt.imMask[d][i] {
                for j in (i + 1)..<n where adt.imMask[d][j] { hot.overlap[i][j] += 1 }
            }
        }
        var toRemove = [Bool](repeating: false, count: n)
        for t in 0..<n {
            let within = hot.firstFrame[t] > frameIdx - hotstartDelay
            if !within || hot.removed[t] { continue }
            var maxOv = 0
            for e in 0..<n {
                // overlap is upper-triangular (zeros below), exactly like the reference
                if hot.firstFrame[e] < hot.firstFrame[t] { maxOv = max(maxOv, hot.overlap[e][t]) }
            }
            if hot.unmatchCnt[t] >= hotstartUnmatch || maxOv >= hotstartDup { toRemove[t] = true }
        }
        for t in 0..<n where toRemove[t] { hot.removed[t] = true }
        return toRemove
    }

    private func suppressOverlappingOccl(_ frameIdx: Int, _ trkMasks: inout [Float], _ n: Int,
                                         _ toRemove: [Bool]) {
        let packed = (0..<n).map { TM.pack(trkMasks, $0 * MASK * MASK, MASK * MASK, 0) }
        let last = (0..<n).map { toRemove.count > $0 && toRemove[$0] ? 100000 : hot.lastOccl[$0] }
        var sup = [Bool](repeating: false, count: n)
        if n > 1 {
            for i in 0..<n {
                for j in (i + 1)..<n {
                    let inter = Float(TM.popcountAnd(packed[i], packed[j]))
                    let union = max(Float(TM.popcountOr(packed[i], packed[j])), 1)
                    if inter / union >= occlThresh {
                        if last[i] > last[j] && last[j] > -1 { sup[i] = true }
                        if last[j] > last[i] && last[i] > -1 { sup[j] = true }
                    }
                }
            }
        }
        for i in 0..<n {
            let occluded = TM.popcount(packed[i]) == 0
            hot.lastOccl[i] = (occluded || sup[i]) ? frameIdx : last[i]
            if sup[i] {
                Sam3Tracker.fill(&trkMasks, i * MASK * MASK, (i + 1) * MASK * MASK, -10)
            }
        }
    }

    private func updateMemories(_ frameIdx: Int, _ trkMasks: [Float], _ nTrk: Int) throws {
        var high = TM.interpBilinear(trkMasks, nTrk, MASK, MASK, INMASK, INMASK)
        suppressPwAreaShrinkage(&high, nTrk, INMASK * INMASK)
        let osl: [Float] = (0..<nTrk).map { i in
            var any = false
            for p in 0..<(INMASK * INMASK) where high[i * INMASK * INMASK + p] > 0 {
                any = true
                break
            }
            return any ? 10 : -10
        }
        // global sorted-by-id positions per state
        var owners: [(stateIdx: Int, objId: Int)] = []
        for (si, st) in states.enumerated() { for oid in st.objIds { owners.append((si, oid)) } }
        let order = owners.indices.sorted { owners[$0].objId < owners[$1].objId }
        var assign: [Int: [Int]] = [:]
        for (gpos, li) in order.enumerated() {
            assign[owners[li].stateIdx, default: []].append(gpos)
        }
        for (si, st) in states.enumerated() {
            if st.objIds.isEmpty { continue }
            let idxs = assign[si]!
            let entry = st.outputCond[frameIdx] ?? st.outputNonCond[frameIdx]
            let condObjs: Set<Int> = entry?.conditioning ?? []
            var subHigh = [Float](repeating: 0, count: idxs.count * INMASK * INMASK)
            var subOsl = [Float](repeating: 0, count: idxs.count)
            for (k, g) in idxs.enumerated() {
                Sam3Tracker.cpy(high, g * INMASK * INMASK, &subHigh, k * INMASK * INMASK, INMASK * INMASK)
                subOsl[k] = osl[g]
            }
            var featsMem = try encodeNewMemory(st.curPropF2!, subHigh, INMASK, subOsl, condObjs, st.mux!)
            if let entry = entry {
                TM.bf16InPlace(&featsMem)
                entry.maskmem = featsMem
                entry.imageFeatures = curImageFeatures
            }
        }
    }

    private func recondition(_ frameIdx: Int, _ det: DetOut, _ adt: Adt,
                             _ trkMasks: inout [Float], _ trkScores: [Float]) throws {
        struct Cand {
            let trkId: Int
            let detRow: Int
            let objPos: Int
        }
        let cands = adt.hiConfKeys
            .map { Cand(trkId: $0, detRow: adt.hiConf[$0]!, objPos: objIdsAll.firstIndex(of: $0) ?? -1) }
            .filter { TM.sigmoid(trkScores[$0.objPos]) > 0.8 }
        if cands.isEmpty { return }
        let newBin1152 = detMasksAt(det, cands.map { $0.detRow }, INMASK)
        for cd in cands {
            let o = det.maskOffset(cd.detRow)
            let tb = cd.objPos * MASK * MASK
            for p in 0..<(MASK * MASK) {
                let newV = det.headY[o + p]
                if (newV > 0) != (trkMasks[tb + p] > 0) { trkMasks[tb + p] = newV }
            }
        }
        for st in states {
            let pairIdx = cands.indices.filter { st.objIdToIdx[cands[$0].trkId] != nil }
            if pairIdx.isEmpty { continue }
            try addNewMasks(st, frameIdx, pairIdx.map { cands[$0].trkId },
                            pairIdx.map { newBin1152[$0] }, reconditioning: true)
            try preflight(st)
        }
    }

    private func updateConfirmation(_ prevIds: [Int], _ newIdsAll: [Int], _ adt: Adt,
                                    _ newDetIds: [Int]) {
        var status = [Int](repeating: 1, count: newIdsAll.count)
        var cnt = [Int](repeating: 0, count: newIdsAll.count)
        var pos: [Int: Int] = [:]
        for (i, o) in newIdsAll.enumerated() { pos[o] = i }
        for (i, o) in prevIds.enumerated() {
            guard let j = pos[o] else { continue }
            status[j] = confStatus[i]
            cnt[j] = confCnt[i]
        }
        var matched = Set(newDetIds)
        for ids in adt.detMatched.values { matched.formUnion(ids) }
        for j in 0..<newIdsAll.count {
            cnt[j] = matched.contains(newIdsAll[j]) ? cnt[j] + 1 : 0
            if cnt[j] >= confThresh { status[j] = 2 }
        }
        confStatus = status
        confCnt = cnt
    }

    // ================================================================ execution
    private func addObjectsExecution(_ frameIdx: Int, _ det: DetOut, _ newFa: [Int],
                                     _ newIds: [Int]) throws {
        let masks1152 = detMasksAt(det, newFa, INMASK)
        var best: TrackState?
        for st in states {
            guard let mux = st.mux else { continue }
            let av = mux.availableSlots
            if av >= newFa.count && (best == nil || av < best!.mux!.availableSlots) { best = st }
        }
        if best == nil {
            best = TrackState()
            states.append(best!)
        }
        best!.curPropF2 = visSlice("prop_f2")
        try addNewMasks(best!, frameIdx, newIds, masks1152, reconditioning: false)
        try preflight(best!)
    }

    private func removeObjectsExecution(_ objIds: Set<Int>) {
        // NOT exercised by the verification clip; simplified port of remove_objects.
        var keep: [TrackState] = []
        for st in states {
            let idxs = objIds.compactMap { st.objIdToIdx[$0] }.sorted()
            if !idxs.isEmpty {
                st.mux!.removeObjects(idxs)
                let removeSet = Set(idxs)
                var old2new: [Int: Int] = [:]
                var neu = 0
                for old in 0..<st.objIdToIdx.count where !removeSet.contains(old) {
                    old2new[old] = neu
                    neu += 1
                }
                var newOrder: [Int] = []
                var newMap: [Int: Int] = [:]
                for oid in st.objIdOrder {
                    let i = st.objIdToIdx[oid]!
                    if let ni = old2new[i] {
                        newOrder.append(oid)
                        newMap[oid] = ni
                    }
                }
                st.objIdOrder = newOrder
                st.objIdToIdx = newMap
                let keepRows = old2new.keys.sorted()
                for storage in [st.outputCond, st.outputNonCond] {
                    for e in storage.values {
                        var pm = [Float](repeating: 0, count: keepRows.count * MASK * MASK)
                        var os = [Float](repeating: 0, count: keepRows.count)
                        for (k, r) in keepRows.enumerated() where r < e.nRows {
                            Sam3Tracker.cpy(e.predMasks, r * MASK * MASK, &pm, k * MASK * MASK, MASK * MASK)
                            os[k] = e.osl[r]
                        }
                        e.predMasks = pm
                        e.osl = os
                        e.nRows = keepRows.count
                        var newCond = Set<Int>()
                        for o in e.conditioning { if let v = old2new[o] { newCond.insert(v) } }
                        e.conditioning = newCond
                    }
                }
            }
            if !st.objIds.isEmpty { keep.append(st) }
        }
        states = keep
    }

    // ================================================================ per-frame step
    struct FrameOut {
        var objIdToMask: [(Int, [Bool])] = []   // insertion order
        var removedNow = Set<Int>()
        var unconfirmed: [Int] = []
        var sam2Scores: [Int: Float] = [:]
    }

    private func detTrackOneFrame(_ frameIdx: Int, _ det: DetOut) throws -> FrameOut {
        // Step 2: propagation
        var objIdsLocal: [Int] = []
        var lowList: [[Float]] = []
        var scoreList: [[Float]] = []
        for st in states {
            if st.objIds.isEmpty { continue }
            st.curPropF2 = visSlice("prop_f2")
            let out = try propagateStateOneFrame(st, frameIdx)
            objIdsLocal.append(contentsOf: out.ids)
            lowList.append(out.masks)
            scoreList.append(out.scores)
        }
        let nTrk = objIdsLocal.count
        var trkMasks = [Float](repeating: 0, count: nTrk * MASK * MASK)
        var trkScores = [Float](repeating: 0, count: nTrk)
        var r = 0
        for k in 0..<lowList.count {
            Sam3Tracker.cpy(lowList[k], 0, &trkMasks, r * MASK * MASK, scoreList[k].count * MASK * MASK)
            for i in 0..<scoreList[k].count { trkScores[r + i] = scoreList[k][i] }
            r += scoreList[k].count
        }
        if objIdsLocal != objIdsLocal.sorted() {
            let order = objIdsLocal.indices.sorted { objIdsLocal[$0] < objIdsLocal[$1] }
            var m2 = [Float](repeating: 0, count: nTrk * MASK * MASK)
            var s2 = [Float](repeating: 0, count: nTrk)
            for (k, o) in order.enumerated() {
                Sam3Tracker.cpy(trkMasks, o * MASK * MASK, &m2, k * MASK * MASK, MASK * MASK)
                s2[k] = trkScores[o]
            }
            trkMasks = m2
            trkScores = s2
            objIdsLocal.sort()
        }
        precondition(objIdsAll == objIdsLocal, "obj id bookkeeping diverged")

        // Step 3: planning
        let adt = associate(det, trkMasks, objIdsAll)
        let toRemove = processHotstart(frameIdx, adt)
        if recondEvery > 0 && frameIdx % recondEvery == 0 && !adt.hiConf.isEmpty {
            try recondition(frameIdx, det, adt, &trkMasks, trkScores)
        }
        if nTrk > 0 {
            suppressOverlappingOccl(frameIdx, &trkMasks, nTrk, toRemove)
            try updateMemories(frameIdx, trkMasks, nTrk)
        }

        var newFa = (0..<200).filter { adt.isNewDet[$0] }
        let prevN = objIdsAll.count
        if prevN + newFa.count > maxObjects {
            let keepN = max(maxObjects - prevN, 0)
            // stable ascending then reversed, like np.argsort(kind="stable")[::-1]
            let order = newFa.indices.sorted { a, b in
                det.scores[newFa[a]] != det.scores[newFa[b]]
                    ? det.scores[newFa[a]] < det.scores[newFa[b]] : a < b
            }.reversed()
            newFa = Array(order.prefix(keepN)).map { newFa[$0] }
        }
        let newIds = (0..<newFa.count).map { maxObjId + 1 + $0 }
        var removedNow = Set<Int>()
        for t in 0..<toRemove.count where toRemove[t] { removedNow.insert(objIdsAll[t]) }

        let prevIds = objIdsAll
        objIdsAll = prevIds.filter { !removedNow.contains($0) } + newIds
        var frameScores = sam2ScoreFrame[frameIdx] ?? [:]
        for k in 0..<newIds.count {
            objIdToScore[newIds[k]] = det.scores[newFa[k]]
            frameScores[newIds[k]] = det.scores[newFa[k]]
        }
        if let mx = newIds.max() { maxObjId = max(maxObjId, mx) }
        for oid in removedNow {
            objIdToScore[oid] = -1e4
            frameScores[oid] = -1e4
        }
        updateConfirmation(prevIds, objIdsAll, adt, newIds)

        // hotstart array bookkeeping
        if hot.n > 0 {
            let keepIdx = (0..<hot.n).filter { !hot.removed[$0] }
            hot.firstFrame = keepIdx.map { hot.firstFrame[$0] }
            hot.unmatchCnt = keepIdx.map { hot.unmatchCnt[$0] }
            hot.keepAlive = keepIdx.map { hot.keepAlive[$0] }
            hot.lastOccl = keepIdx.map { hot.lastOccl[$0] }
            hot.overlap = keepIdx.map { i in keepIdx.map { j in hot.overlap[i][j] } }
            hot.removed = [Bool](repeating: false, count: keepIdx.count)
            hot.n = keepIdx.count
        }
        if !newIds.isEmpty {
            let nn = newIds.count
            let oldN = hot.n
            hot.firstFrame += [Int](repeating: frameIdx, count: nn)
            hot.unmatchCnt += [Int](repeating: 0, count: nn)
            hot.keepAlive += [Int](repeating: initKeepAlive, count: nn)
            hot.removed += [Bool](repeating: false, count: nn)
            hot.lastOccl += [Int](repeating: -1, count: nn)
            var ov = [[Int]](repeating: [Int](repeating: 0, count: oldN + nn), count: oldN + nn)
            for i in 0..<oldN { for j in 0..<oldN { ov[i][j] = hot.overlap[i][j] } }
            hot.overlap = ov
            hot.n = oldN + nn
        }
        removedObjIds.formUnion(removedNow)

        // Step 4: execution
        if !newFa.isEmpty { try addObjectsExecution(frameIdx, det, newFa, newIds) }
        if !removedNow.isEmpty { removeObjectsExecution(removedNow) }

        for i in 0..<prevIds.count { frameScores[prevIds[i]] = TM.sigmoid(trkScores[i]) }
        sam2ScoreFrame[frameIdx] = frameScores

        // Step 5: outputs
        var out = FrameOut()
        if nTrk > 0 {
            let vid = TM.interpBilinear(trkMasks, nTrk, MASK, MASK, vH, vW)
            for i in 0..<prevIds.count {
                var m = [Bool](repeating: false, count: vH * vW)
                vid.withUnsafeBufferPointer { vp in
                    for p in 0..<(vH * vW) where vp[i * vH * vW + p] > 0 { m[p] = true }
                }
                out.objIdToMask.append((prevIds[i], m))
            }
        }
        if !newFa.isEmpty {
            for k in 0..<newFa.count {
                let o = det.maskOffset(newFa[k])
                let up = TM.interpBilinear(Array(det.headY[o..<(o + MASK * MASK)]),
                                           1, MASK, MASK, vH, vW)
                var m = [Bool](repeating: false, count: vH * vW)
                for p in 0..<(vH * vW) where up[p] > 0 { m[p] = true }
                out.objIdToMask.append((newIds[k], m))
            }
        }
        out.removedNow = removedNow
        out.unconfirmed = objIdsAll.indices.filter { confStatus[$0] == 1 }.map { objIdsAll[$0] }
        out.sam2Scores = frameScores
        return out
    }

    // ================================================================ full-clip run
    struct FrameResult {
        let ids: [Int]
        let probs: [Float]
        let masks: [[UInt64]]
    }

    /// Track the prompt through the numbered jpgs in clipDir. Returns per-frame
    /// results (ids ascending, probs = first detection score, packed video-res
    /// binary masks). `progress` gets one line per processed frame.
    func track(clipDir: URL, progress: (String) -> Void) throws -> [Int: FrameResult] {
        let frameFiles = try FileManager.default.contentsOfDirectory(at: clipDir, includingPropertiesForKeys: nil)
            .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
            .sorted { Int($0.deletingPathExtension().lastPathComponent)! <
                Int($1.deletingPathExtension().lastPathComponent)! }
        numFrames = frameFiles.count
        // Frames are decoded on demand (sequential access, frame 0 twice): keeping
        // all 48 fp16-decoded frames resident costs ~290 MB and the app already
        // brushes the per-app memory ceiling during the vision graph's first run.
        var frameCache: (idx: Int, data: [UInt16])?
        func frameData(_ fi: Int) throws -> [UInt16] {
            if let c = frameCache, c.idx == fi { return c.data }
            let d = try loadFrame(frameFiles[fi])
            frameCache = (fi, d)
            return d
        }

        let (textMem, pad) = textMemPad

        func runFrame(_ fi: Int) throws -> FrameOut {
            try runVision(frameData(fi))
            let det = try runDetection(textMem, pad)
            return try detTrackOneFrame(fi, det)
        }

        let delay = confThresh - 1
        var unconfirmedPerFrame: [Int: [Int]] = [:]
        var outs: [Int: FrameOut] = [:]
        var removedSnapshotOf: [Int: Set<Int>] = [:]
        var hotRemoved = Set<Int>()

        _ = try runFrame(0)                                // add_prompt(frame 0)
        for fi in 0..<numFrames {                          // propagate_in_video forward
            let snap = graphSnapshot()
            let t0 = Date()
            let out = try runFrame(fi)
            outs[fi] = out
            hotRemoved.formUnion(out.removedNow)
            unconfirmedPerFrame[fi] = out.unconfirmed
            if fi == numFrames - 1 {
                for yf in outs.keys where removedSnapshotOf[yf] == nil {
                    removedSnapshotOf[yf] = hotRemoved
                }
            } else if fi >= hotstartDelay - 1 {
                removedSnapshotOf[fi - (hotstartDelay - 1)] = hotRemoved
            }
            // heap relief: the memory bank only reaches cond + t-6..t-1 back, so the
            // big per-frame tensors of older non-cond entries are dead (obj_ptr is
            // still read up to 15 frames back and is kept).
            for st in states {
                for (f, e) in st.outputNonCond where f <= fi - 6 {
                    e.maskmem = nil
                    e.imageFeatures = nil
                }
            }
            progress("f\(fi) \(Int(Date().timeIntervalSince(t0) * 1000))ms  \(graphDelta(snap))")
        }

        var results: [Int: FrameResult] = [:]
        for fi in 0..<numFrames {
            let out = outs[fi]!
            let snapshot = removedSnapshotOf[fi] ?? hotRemoved
            let ids = out.objIdToMask.map { $0.0 }.sorted()
            if ids.isEmpty {
                results[fi] = FrameResult(ids: [], probs: [], masks: [])
                continue
            }
            var maskOf: [Int: [Bool]] = [:]
            for (id, m) in out.objIdToMask { maskOf[id] = m }
            let fUnc = max(0, min(fi + delay, numFrames - 1))
            var hide = snapshot
            hide.formUnion(unconfirmedPerFrame[fUnc] ?? [])
            let kept = ids.filter { id in
                let m = maskOf[id]!
                return m.contains(true) && !hide.contains(id)
            }
            let masksB = kept.map { maskOf[$0]! }
            let sam2 = kept.map { out.sam2Scores[$0] ?? 0 }
            // object-wise non-overlap at video res, scored by the per-frame sam2 probs
            let px = vH * vW
            var packed: [[UInt64]]
            if kept.count > 1 {
                // obj_wise_non_overlap: per pixel the highest-scoring claimant keeps it
                // (argmax ties -> first), losers lose the pixel.
                var fin = [[UInt64]](repeating: [UInt64](repeating: 0, count: TM.packedWords(px)),
                                     count: kept.count)
                for p in 0..<px {
                    var arg = 0
                    var mx: Float = masksB[0][p] ? sam2[0] : 0
                    for i in 1..<kept.count {
                        let v: Float = masksB[i][p] ? sam2[i] : 0
                        if v > mx {
                            mx = v
                            arg = i
                        }
                    }
                    if mx > 0 && masksB[arg][p] {
                        fin[arg][p >> 6] |= 1 << UInt64(p & 63)
                    }
                }
                packed = fin
            } else {
                packed = masksB.map { m in
                    var w = [UInt64](repeating: 0, count: TM.packedWords(px))
                    for p in 0..<px where m[p] { w[p >> 6] |= 1 << UInt64(p & 63) }
                    return w
                }
            }
            results[fi] = FrameResult(ids: kept, probs: kept.map { objIdToScore[$0]! }, masks: packed)
        }
        return results
    }
}

// Free helpers so the init body can resolve graph paths before `self` is ready.
private func rootFileStatic(_ modelsRoot: URL, _ name: String) throws -> URL {
    let f = modelsRoot.appendingPathComponent(name)
    guard FileManager.default.fileExists(atPath: f.path) else {
        throw LiteRTError.interface("missing \(name) — stage the tracker models first")
    }
    return f
}

private func graphFileStatic(_ modelsRoot: URL, _ trackerDir: URL, _ name: String) throws -> URL {
    let g = trackerDir.appendingPathComponent("graphs/\(name)")
    return FileManager.default.fileExists(atPath: g.path) ? g : try rootFileStatic(modelsRoot, name)
}

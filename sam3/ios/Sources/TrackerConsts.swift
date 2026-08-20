import Foundation

/// Host-side constants and flags for the tracker loop, produced by
/// scripts/dump_tracker_device_assets.py (consts/<name>.bin raw LE float32 +
/// consts/manifest.json name->shape, flags.json).
final class TrackerConsts {
    private var tensors: [String: [Float]] = [:]
    private let flags: [String: Any]

    init(trackerDir: URL) throws {
        let cdir = trackerDir.appendingPathComponent("consts")
        let manifestData = try Data(contentsOf: cdir.appendingPathComponent("manifest.json"))
        guard let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: [String: Any]]
        else { throw LiteRTError.interface("consts manifest is not an object") }
        for (name, entry) in manifest {
            guard let file = entry["file"] as? String else {
                throw LiteRTError.interface("const \(name) has no file")
            }
            let data = try Data(contentsOf: cdir.appendingPathComponent(file))
            var arr = [Float](repeating: 0, count: data.count / 4)
            _ = arr.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            tensors[name] = arr
        }
        let flagsData = try Data(contentsOf: trackerDir.appendingPathComponent("flags.json"))
        guard let f = try JSONSerialization.jsonObject(with: flagsData) as? [String: Any]
        else { throw LiteRTError.interface("flags.json is not an object") }
        flags = f
    }

    subscript(name: String) -> [Float] {
        guard let t = tensors[name] else { preconditionFailure("missing const \(name)") }
        return t
    }

    func flagInt(_ name: String) -> Int {
        guard let v = flags[name] as? NSNumber else { preconditionFailure("missing flag \(name)") }
        return v.intValue
    }

    func flagFloat(_ name: String) -> Float {
        guard let v = flags[name] as? NSNumber else { preconditionFailure("missing flag \(name)") }
        return v.floatValue
    }

    func flagBool(_ name: String) -> Bool {
        guard let v = flags[name] as? NSNumber else { preconditionFailure("missing flag \(name)") }
        return v.boolValue
    }

    /// 3-layer ReLU MLP 256->256 (obj_ptr_proj / interactive_obj_ptr_proj).
    func mlp3(_ x: [Float], _ m: Int, _ prefix: String) -> [Float] {
        var v = x
        for i in 0..<3 {
            v = TM.linear(v, m, 256, self["\(prefix).\(i).w"], 256, self["\(prefix).\(i).b"])
            if i < 2 { for j in 0..<v.count where v[j] < 0 { v[j] = 0 } }
        }
        return v
    }

    func noObjPtrBlend(_ ptr: [Float], _ m: Int, _ lam: [Float]) -> [Float] {
        let alt = TM.linear(ptr, m, 256, self["no_obj_ptr_linear.w"], 256, self["no_obj_ptr_linear.b"])
        var out = [Float](repeating: 0, count: m * 256)
        for r in 0..<m {
            for c in 0..<256 {
                out[r * 256 + c] = lam[r] * ptr[r * 256 + c] + (1 - lam[r]) * alt[r * 256 + c]
            }
        }
        return out
    }
}

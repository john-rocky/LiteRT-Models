import Foundation

/// Swift wrapper over the LiteRT Next C API (symbols from CLiteRTLM.framework)
/// for the SAM3 pipeline: one shared environment, per-graph compiled models with
/// a choice of accelerator, flat single-input / single-output float inference
/// with reusable host buffers.
enum LiteRTError: Error, CustomStringConvertible {
    case status(String, Int32)
    case interface(String)
    var description: String {
        switch self {
        case .status(let what, let code): return "\(what) failed (status \(code))"
        case .interface(let why): return why
        }
    }
}

private func check(_ s: LiteRtStatus, _ what: String) throws {
    if s != kLiteRtStatusOk {
        Sam3Log.log("ERROR \(what) failed (status \(s.rawValue))")
        throw LiteRTError.status(what, Int32(s.rawValue))
    }
}

/// Status log every subsystem appends to; mirrored to Documents/demo_status.txt so a
/// host can pull progress and failures off the device without anyone reading the screen.
enum Sam3Log {
    private static let url = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("demo_status.txt")
    private static let queue = DispatchQueue(label: "sam3.log")

    static func log(_ line: String) {
        print("SAM3 \(line)")
        queue.sync {
            let stamped = "\(Date().timeIntervalSince1970) \(line)\n"
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(stamped.data(using: .utf8)!)
                try? handle.close()
            } else {
                try? stamped.data(using: .utf8)!.write(to: url)
            }
        }
    }
}

/// The one LiteRT environment shared by every compiled model in the process.
final class LiteRTEnvHolder {
    static let shared = LiteRTEnvHolder()
    private(set) var env: LiteRtEnvironment?
    private init() {
        var e: LiteRtEnvironment?
        if LiteRtCreateEnvironment(0, nil, &e) == kLiteRtStatusOk { env = e }
    }
}

/// How a graph is compiled and executed.
enum Sam3Accel {
    /// Metal, delegate-default fp16 arithmetic (fastest).
    case gpu
    /// Metal with enforce_f32: float32 arithmetic, exact (the CLIP-L text stream
    /// at |x|~1.2e3 and the head's borderline scores need this).
    case gpuF32
    /// XNNPACK CPU (exact; used where the GPU graph is unnecessary).
    case cpu

    var label: String {
        switch self {
        case .gpu: return "GPU fp16"
        case .gpuF32: return "GPU f32"
        case .cpu: return "CPU"
        }
    }
}

/// One compiled .tflite with a flat float32 single-input / single-output
/// signature (all three SAM3 image-side graphs are exported this way).
final class Sam3Graph {
    private var model: LiteRtModel?
    private var options: LiteRtOptions?
    private var compiled: LiteRtCompiledModel?
    private var inDimsList: [[Int32]] = []
    private var inputSizes: [Int] = []
    private var outDims: [Int32] = []
    private var loggedBuffers = false
    /// When false (default), I/O buffers are created on the first run() and reused —
    /// per-run create/destroy churns wired GPU memory that the runtime frees lazily,
    /// which accumulates until the device runs out. Per-run mode remains for graphs
    /// that run rarely (initdec) to keep the ~512 MB tensor-buffer pool under budget.
    let perRunBuffers: Bool
    private var cachedInputs: [LiteRtTensorBuffer?] = []
    private var cachedOutput: LiteRtTensorBuffer?
    private var cachedOwnedMem: [UnsafeMutableRawPointer] = []

    let name: String
    let accel: Sam3Accel
    /// Total float count across all inputs; a flat host array is written across
    /// the input tensors in signature order (they are concatenation slices).
    private(set) var inputCount = 0
    private(set) var outputCount = 0
    private(set) var fullyAccelerated = false
    private(set) var compileSeconds = 0.0
    /// run + readback of the last call (GPU compute blocks inside the read lock).
    private(set) var lastRunMs = 0.0

    init(path: String, accel: Sam3Accel, perRunBuffers: Bool = false) throws {
        self.name = (path as NSString).lastPathComponent
        self.accel = accel
        self.perRunBuffers = perRunBuffers
        guard let env = LiteRTEnvHolder.shared.env else {
            throw LiteRTError.status("CreateEnvironment", -1)
        }
        try check(LiteRtCreateModelFromFile(env, path, &model), "CreateModelFromFile")
        let (inDimsList, outDims) = try interface()
        self.inDimsList = inDimsList
        self.outDims = outDims
        inputSizes = inDimsList.map { $0.reduce(1) { $0 * Int(max($1, 1)) } }
        inputCount = inputSizes.reduce(0, +)
        outputCount = outDims.reduce(1) { $0 * Int(max($1, 1)) }

        try check(LiteRtCreateOptions(&options), "CreateOptions")
        let hw = accel == .cpu ? kLiteRtHwAcceleratorCpu : kLiteRtHwAcceleratorGpu
        try check(
            LiteRtSetOptionsHardwareAccelerators(options, LiteRtHwAcceleratorSet(hw.rawValue)),
            "SetHardwareAccelerators")
        if accel == .gpuF32 {
            // Same native path as the Mac-verified Python `gpu_enforce_f32=True`.
            try check(Sam3AddGpuOptions(options, true), "AddGpuOptions")
        }

        let t0 = Date()
        try check(LiteRtCreateCompiledModel(env, model, options, &compiled), "CreateCompiledModel \(name)")
        compileSeconds = Date().timeIntervalSince(t0)
        var fully = false
        try check(LiteRtCompiledModelIsFullyAccelerated(compiled, &fully), "IsFullyAccelerated")
        fullyAccelerated = fully
        Sam3Log.log("\(name): compiled \(accel.label) fully=\(fullyAccelerated) "
            + "in \(String(format: "%.1f", compileSeconds))s; in=\(inDimsList) out=\(outDims)")

        // I/O buffers are created per run() call: the Metal tensor-buffer pool is
        // ~512 MB and the tracker's persistent buffer set (~640 MB) exhausts it.
    }

    /// Dims of every input + output 0 of signature 0, falling back to subgraph 0
    /// (litert-torch exports sometimes carry no signature defs). Multiple inputs are
    /// slices of one flat host array, in signature order.
    private func interface() throws -> (inDimsList: [[Int32]], outDims: [Int32]) {
        func dims(of tensor: LiteRtTensor?) throws -> [Int32] {
            var type = LiteRtRankedTensorType()
            try check(LiteRtGetRankedTensorType(tensor, &type), "GetRankedTensorType")
            guard Sam3ElemType(&type) == 1 else {  // kLiteRtElementTypeFloat32
                throw LiteRTError.interface("\(name): non-float32 interface tensor")
            }
            return (0..<Sam3Rank(&type)).map { Sam3Dim(&type, $0) }
        }
        var numSigs: LiteRtParamIndex = 0
        try check(LiteRtGetNumModelSignatures(model, &numSigs), "GetNumModelSignatures")
        if numSigs > 0 {
            var sig: LiteRtSignature?
            try check(LiteRtGetModelSignature(model, 0, &sig), "GetModelSignature")
            var n: LiteRtParamIndex = 0
            try check(LiteRtGetNumSignatureInputs(sig, &n), "GetNumSignatureInputs")
            var ins: [[Int32]] = []
            for i in 0..<n {
                var tin: LiteRtTensor?
                try check(LiteRtGetSignatureInputTensorByIndex(sig, i, &tin), "GetSignatureInputTensor")
                ins.append(try dims(of: tin))
            }
            try check(LiteRtGetNumSignatureOutputs(sig, &n), "GetNumSignatureOutputs")
            guard n == 1 else { throw LiteRTError.interface("\(name): expected 1 output, got \(n)") }
            var tout: LiteRtTensor?
            try check(LiteRtGetSignatureOutputTensorByIndex(sig, 0, &tout), "GetSignatureOutputTensor")
            return (ins, try dims(of: tout))
        }
        var subgraph: LiteRtSubgraph?
        try check(LiteRtGetModelSubgraph(model, 0, &subgraph), "GetModelSubgraph")
        var n: LiteRtParamIndex = 0
        try check(LiteRtGetNumSubgraphInputs(subgraph, &n), "GetNumSubgraphInputs")
        var ins: [[Int32]] = []
        for i in 0..<n {
            var tin: LiteRtTensor?
            try check(LiteRtGetSubgraphInput(subgraph, i, &tin), "GetSubgraphInput")
            ins.append(try dims(of: tin))
        }
        try check(LiteRtGetNumSubgraphOutputs(subgraph, &n), "GetNumSubgraphOutputs")
        guard n == 1 else { throw LiteRTError.interface("\(name): expected 1 output, got \(n)") }
        var tout: LiteRtTensor?
        try check(LiteRtGetSubgraphOutput(subgraph, 0, &tout), "GetSubgraphOutput")
        return (ins, try dims(of: tout))
    }

    /// Create the buffer the compiled model actually asks for. Plain host-memory
    /// buffers are rejected above ~112 MiB by the Metal-backed runtime (status 3),
    /// so honor the model's buffer requirements — the same thing the Kotlin/Python
    /// `createInputBuffers()` conveniences do.
    private func makeBuffer(dims: [Int32], what: String, index: Int,
                            ownedMem: inout [UnsafeMutableRawPointer]) throws -> LiteRtTensorBuffer? {
        guard let env = LiteRTEnvHolder.shared.env else {
            throw LiteRTError.status("Environment", -1)
        }
        var reqs: LiteRtTensorBufferRequirements?
        if what == "input" {
            try check(
                LiteRtGetCompiledModelInputBufferRequirements(
                    compiled, 0, LiteRtParamIndex(index), &reqs),
                "\(name): \(what) buffer requirements")
        } else {
            try check(
                LiteRtGetCompiledModelOutputBufferRequirements(
                    compiled, 0, LiteRtParamIndex(index), &reqs),
                "\(name): \(what) buffer requirements")
        }
        var nTypes: Int32 = 0
        try check(
            LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(reqs, &nTypes),
            "\(name): \(what) supported type count")
        var supported: [LiteRtTensorBufferType] = []
        for i in 0..<nTypes {
            var t = kLiteRtTensorBufferTypeHostMemory
            try check(
                LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(reqs, i, &t),
                "\(name): \(what) supported type[\(i)]")
            supported.append(t)
        }
        var reqBytes = 0
        try check(
            LiteRtGetTensorBufferRequirementsBufferSize(reqs, &reqBytes),
            "\(name): \(what) required size")
        var type = dims.withUnsafeBufferPointer {
            Sam3MakeType(1, $0.baseAddress, UInt32(dims.count))
        }
        let f32Bytes = dims.reduce(1) { $0 * Int(max($1, 1)) } * MemoryLayout<Float>.stride

        let hostBytes = max(reqBytes, f32Bytes)

        // The Metal delegate accepts ONLY the buffer types it lists (host-memory
        // buffers compile but the run fails with status 3), so create exactly what
        // the requirements ask for, at exactly the size they ask for — Lock/Unlock
        // does the canonical-layout conversion, same as the CL buffers on Android.
        if let devType = supported.first, devType != kLiteRtTensorBufferTypeHostMemory {
            var buffer: LiteRtTensorBuffer?
            let status = LiteRtCreateManagedTensorBuffer(env, devType, &type, reqBytes, &buffer)
            if status == kLiteRtStatusOk {
                if !loggedBuffers {
                    Sam3Log.log("\(name): \(what)[\(index)] device type=\(devType.rawValue) "
                        + "req=\(reqBytes) B (f32 \(f32Bytes) B)")
                }
                return buffer
            }
            Sam3Log.log("\(name): \(what)[\(index)] device type=\(devType.rawValue) "
                + "req=\(reqBytes) B failed (status \(status.rawValue)); trying host")
        }

        // Host-memory fallback: wrap memory WE allocate (managed host buffers come
        // out of the GPU environment's finite pool, which the tracker set exhausts).
        do {
            let mem = UnsafeMutableRawPointer.allocate(
                byteCount: hostBytes, alignment: 64)
            memset(mem, 0, hostBytes)
            var buffer: LiteRtTensorBuffer?
            let status = LiteRtCreateTensorBufferFromHostMemory(&type, mem, hostBytes, nil, &buffer)
            if status == kLiteRtStatusOk {
                ownedMem.append(mem)
                if !loggedBuffers {
                    Sam3Log.log("\(name): \(what)[\(index)] wrapped-host \(hostBytes) B "
                        + "(supported=\(supported.map { $0.rawValue }))")
                }
                return buffer
            }
            mem.deallocate()
            Sam3Log.log("\(name): \(what)[\(index)] wrapped-host failed (status \(status.rawValue)); "
                + "falling back to managed")
        }

        var candidates: [(LiteRtTensorBufferType, Int)] = [
            (kLiteRtTensorBufferTypeHostMemory, hostBytes)
        ]
        if supported.contains(kLiteRtTensorBufferTypeMetalBuffer) {
            candidates.append((kLiteRtTensorBufferTypeMetalBuffer, hostBytes))
        }
        var lastError: Error?
        for (t, bytes) in candidates {
            var buffer: LiteRtTensorBuffer?
            let status = LiteRtCreateManagedTensorBuffer(env, t, &type, bytes, &buffer)
            if status == kLiteRtStatusOk {
                Sam3Log.log("\(name): \(what)[\(index)] type=\(t.rawValue) \(bytes) B "
                    + "(supported=\(supported.map { $0.rawValue }))")
                return buffer
            }
            lastError = LiteRTError.status(
                "\(name): create \(what)[\(index)] type=\(t.rawValue) dims=\(dims) (\(bytes) B, "
                    + "supported=\(supported.map { $0.rawValue }))",
                Int32(status.rawValue))
            Sam3Log.log("\(lastError!)")
        }
        throw lastError ?? LiteRTError.interface("\(name): no usable buffer type")
    }

    /// Run the graph on `input` (row-major float32, `inputCount` values, written
    /// across the input tensors in signature order) and return `outputCount` floats.
    /// I/O buffers are created and destroyed per call — the Metal tensor-buffer pool
    /// is ~512 MB, so only one graph's buffer set may be alive at a time.
    func run(_ input: [Float]) throws -> [Float] {
        precondition(input.count == inputCount, "\(name): input \(input.count) != \(inputCount)")
        let t0 = Date()
        var ownedMem: [UnsafeMutableRawPointer] = []
        var inputBuffers: [LiteRtTensorBuffer?] = []
        var outputBuffer: LiteRtTensorBuffer?
        defer {
            if perRunBuffers {
                for b in inputBuffers where b != nil { LiteRtDestroyTensorBuffer(b) }
                if let b = outputBuffer { LiteRtDestroyTensorBuffer(b) }
                for m in ownedMem { m.deallocate() }
            }
        }

        if !perRunBuffers && !cachedInputs.isEmpty {
            inputBuffers = cachedInputs
            outputBuffer = cachedOutput
            var offset = 0
            for (i, buffer) in inputBuffers.enumerated() {
                var addr: UnsafeMutableRawPointer?
                try check(
                    LiteRtLockTensorBuffer(buffer, &addr, kLiteRtTensorBufferLockModeWrite),
                    "LockInput[\(i)]")
                _ = input.withUnsafeBytes {
                    memcpy(addr, $0.baseAddress! + offset * MemoryLayout<Float>.stride,
                           inputSizes[i] * MemoryLayout<Float>.stride)
                }
                try check(LiteRtUnlockTensorBuffer(buffer), "UnlockInput[\(i)]")
                offset += inputSizes[i]
            }
        } else {
            var offset = 0
            for (i, dims) in inDimsList.enumerated() {
                let buffer = try makeBuffer(dims: dims, what: "input", index: i, ownedMem: &ownedMem)
                var addr: UnsafeMutableRawPointer?
                try check(
                    LiteRtLockTensorBuffer(buffer, &addr, kLiteRtTensorBufferLockModeWrite),
                    "LockInput[\(i)]")
                _ = input.withUnsafeBytes {
                    memcpy(addr, $0.baseAddress! + offset * MemoryLayout<Float>.stride,
                           inputSizes[i] * MemoryLayout<Float>.stride)
                }
                try check(LiteRtUnlockTensorBuffer(buffer), "UnlockInput[\(i)]")
                inputBuffers.append(buffer)
                offset += inputSizes[i]
            }
            outputBuffer = try makeBuffer(dims: outDims, what: "output", index: 0, ownedMem: &ownedMem)
            loggedBuffers = true
            if !perRunBuffers {
                cachedInputs = inputBuffers
                cachedOutput = outputBuffer
                cachedOwnedMem.append(contentsOf: ownedMem)
                ownedMem = []
            }
        }

        var inputs: [LiteRtTensorBuffer?] = inputBuffers
        var outputs: [LiteRtTensorBuffer?] = [outputBuffer]
        try check(
            LiteRtRunCompiledModel(compiled, 0, inputs.count, &inputs, 1, &outputs), "\(name): run")

        var readAddr: UnsafeMutableRawPointer?
        try check(
            LiteRtLockTensorBuffer(outputBuffer, &readAddr, kLiteRtTensorBufferLockModeRead),
            "LockOutput")
        var result = [Float](repeating: 0, count: outputCount)
        _ = result.withUnsafeMutableBytes {
            memcpy($0.baseAddress, readAddr, outputCount * MemoryLayout<Float>.stride)
        }
        try check(LiteRtUnlockTensorBuffer(outputBuffer), "UnlockOutput")
        lastRunMs = Date().timeIntervalSince(t0) * 1000
        Sam3Log.log("\(name): run \(String(format: "%.0f", lastRunMs)) ms")
        return result
    }

    deinit {
        for b in cachedInputs where b != nil { LiteRtDestroyTensorBuffer(b) }
        if let b = cachedOutput { LiteRtDestroyTensorBuffer(b) }
        for m in cachedOwnedMem { m.deallocate() }
        if let c = compiled { LiteRtDestroyCompiledModel(c) }
        if let o = options { LiteRtDestroyOptions(o) }
        if let m = model { LiteRtDestroyModel(m) }
    }
}

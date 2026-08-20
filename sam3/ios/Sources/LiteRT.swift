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
    if s != kLiteRtStatusOk { throw LiteRTError.status(what, Int32(s.rawValue)) }
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
    private var inputBuffers: [LiteRtTensorBuffer?] = []
    private var inputSizes: [Int] = []
    private var outputBuffer: LiteRtTensorBuffer?

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

    init(path: String, accel: Sam3Accel) throws {
        self.name = (path as NSString).lastPathComponent
        self.accel = accel
        guard let env = LiteRTEnvHolder.shared.env else {
            throw LiteRTError.status("CreateEnvironment", -1)
        }
        try check(LiteRtCreateModelFromFile(env, path, &model), "CreateModelFromFile")
        let (inDimsList, outDims) = try interface()
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
        print("SAM3 \(name): compiled \(accel.label) fully=\(fullyAccelerated) "
            + "in \(String(format: "%.1f", compileSeconds))s; in=\(inDimsList) out=\(outDims)")

        for (i, dims) in inDimsList.enumerated() {
            inputBuffers.append(try makeBuffer(dims: dims, what: "input", index: i))
        }
        outputBuffer = try makeBuffer(dims: outDims, what: "output", index: 0)
        print("SAM3 \(name): buffers ready")
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
    private func makeBuffer(dims: [Int32], what: String, index: Int) throws -> LiteRtTensorBuffer? {
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

        // Candidates we can fill with plain float32 host writes, tried in order:
        // host memory (interops with every accelerator), then a plain Metal f32
        // buffer if the model lists it. Packed/fp16 types need layout conversion
        // and are not attempted.
        var candidates: [(LiteRtTensorBufferType, Int)] = [
            (kLiteRtTensorBufferTypeHostMemory, max(reqBytes, f32Bytes))
        ]
        if supported.contains(kLiteRtTensorBufferTypeMetalBuffer) {
            candidates.append((kLiteRtTensorBufferTypeMetalBuffer, max(reqBytes, f32Bytes)))
        }
        var lastError: Error?
        for (t, bytes) in candidates {
            var buffer: LiteRtTensorBuffer?
            let status = LiteRtCreateManagedTensorBuffer(env, t, &type, bytes, &buffer)
            if status == kLiteRtStatusOk {
                print("SAM3 \(name): \(what)[\(index)] type=\(t.rawValue) \(bytes) B "
                    + "(supported=\(supported.map { $0.rawValue }))")
                return buffer
            }
            lastError = LiteRTError.status(
                "\(name): create \(what)[\(index)] type=\(t.rawValue) dims=\(dims) (\(bytes) B, "
                    + "supported=\(supported.map { $0.rawValue }))",
                Int32(status.rawValue))
            print("SAM3 \(lastError!)")
        }
        throw lastError ?? LiteRTError.interface("\(name): no usable buffer type")
    }

    /// Run the graph on `input` (row-major float32, `inputCount` values, written
    /// across the input tensors in signature order) and return `outputCount` floats.
    func run(_ input: [Float]) throws -> [Float] {
        precondition(input.count == inputCount, "\(name): input \(input.count) != \(inputCount)")
        let t0 = Date()
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

        var inputs: [LiteRtTensorBuffer?] = inputBuffers
        var outputs: [LiteRtTensorBuffer?] = [outputBuffer]
        try check(
            LiteRtRunCompiledModel(compiled, 0, inputs.count, &inputs, 1, &outputs), "Run")

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
        return result
    }

    deinit {
        for b in inputBuffers where b != nil { LiteRtDestroyTensorBuffer(b) }
        if let b = outputBuffer { LiteRtDestroyTensorBuffer(b) }
        if let c = compiled { LiteRtDestroyCompiledModel(c) }
        if let o = options { LiteRtDestroyOptions(o) }
        if let m = model { LiteRtDestroyModel(m) }
    }
}

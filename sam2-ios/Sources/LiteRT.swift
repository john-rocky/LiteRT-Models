import Foundation

/// Thin Swift wrapper over the LiteRT Next C API (symbols from CLiteRTLM.framework).
///
/// Loads a .tflite, compiles it on the GPU (Metal/WebGPU on iOS), reports whether the
/// whole graph landed on the accelerator, and runs single-input / single-output float
/// inference — enough to time the SAM2 encoder and mask decoder on Apple silicon and
/// compare against MLX on the same device.
final class LiteRTModel {
    private var env: LiteRtEnvironment?
    private var model: LiteRtModel?
    private var options: LiteRtOptions?
    private var compiled: LiteRtCompiledModel?

    enum LiteRTError: Error { case status(String, Int32), notCompiled }

    private func check(_ s: LiteRtStatus, _ what: String) throws {
        if s != kLiteRtStatusOk { throw LiteRTError.status(what, Int32(s.rawValue)) }
    }

    /// Load `path` and compile it on the GPU. Returns whether it is fully accelerated
    /// (no CPU-fallback ops) and how long the compile took (seconds).
    func compileOnGPU(path: String) throws -> (fullyAccelerated: Bool, compileSeconds: Double) {
        try check(LiteRtCreateEnvironment(0, nil, &env), "CreateEnvironment")
        try check(LiteRtCreateModelFromFile(env, path, &model), "CreateModelFromFile")
        try check(LiteRtCreateOptions(&options), "CreateOptions")
        try check(
            LiteRtSetOptionsHardwareAccelerators(
                options, LiteRtHwAcceleratorSet(kLiteRtHwAcceleratorGpu.rawValue)),
            "SetOptionsHardwareAccelerators")

        let start = Date()
        try check(LiteRtCreateCompiledModel(env, model, options, &compiled), "CreateCompiledModel")
        let seconds = Date().timeIntervalSince(start)

        var fully = false
        try check(LiteRtCompiledModelIsFullyAccelerated(compiled, &fully), "IsFullyAccelerated")
        return (fully, seconds)
    }

    /// Run the single-signature, single-input, single-output graph on `input` (row-major
    /// floats matching `inputDims`) and return `outputCount` floats shaped `outputDims`.
    func run(input: [Float], inputDims: [Int32], outputCount: Int, outputDims: [Int32]) throws -> [Float] {
        guard let env = env, let compiled = compiled else { throw LiteRTError.notCompiled }

        var inType = inputDims.withUnsafeBufferPointer {
            Sam2MakeFloat32Type($0.baseAddress, UInt32(inputDims.count))
        }
        var outType = outputDims.withUnsafeBufferPointer {
            Sam2MakeFloat32Type($0.baseAddress, UInt32(outputDims.count))
        }
        let inBytes = input.count * MemoryLayout<Float>.stride
        let outBytes = outputCount * MemoryLayout<Float>.stride

        var inputBuffer: LiteRtTensorBuffer?
        var outputBuffer: LiteRtTensorBuffer?
        try check(
            LiteRtCreateManagedTensorBuffer(
                env, kLiteRtTensorBufferTypeHostMemory, &inType, inBytes, &inputBuffer),
            "CreateInputBuffer")
        try check(
            LiteRtCreateManagedTensorBuffer(
                env, kLiteRtTensorBufferTypeHostMemory, &outType, outBytes, &outputBuffer),
            "CreateOutputBuffer")
        defer {
            LiteRtDestroyTensorBuffer(inputBuffer)
            LiteRtDestroyTensorBuffer(outputBuffer)
        }

        var writeAddr: UnsafeMutableRawPointer?
        try check(
            LiteRtLockTensorBuffer(inputBuffer, &writeAddr, kLiteRtTensorBufferLockModeWrite),
            "LockInput")
        _ = input.withUnsafeBytes { memcpy(writeAddr, $0.baseAddress, inBytes) }
        try check(LiteRtUnlockTensorBuffer(inputBuffer), "UnlockInput")

        var inputs: [LiteRtTensorBuffer?] = [inputBuffer]
        var outputs: [LiteRtTensorBuffer?] = [outputBuffer]
        try check(
            LiteRtRunCompiledModel(compiled, 0, 1, &inputs, 1, &outputs),
            "RunCompiledModel")

        var readAddr: UnsafeMutableRawPointer?
        try check(
            LiteRtLockTensorBuffer(outputBuffer, &readAddr, kLiteRtTensorBufferLockModeRead),
            "LockOutput")
        var result = [Float](repeating: 0, count: outputCount)
        _ = result.withUnsafeMutableBytes { memcpy($0.baseAddress, readAddr, outBytes) }
        try check(LiteRtUnlockTensorBuffer(outputBuffer), "UnlockOutput")
        return result
    }

    deinit {
        if let c = compiled { LiteRtDestroyCompiledModel(c) }
        if let o = options { LiteRtDestroyOptions(o) }
        if let m = model { LiteRtDestroyModel(m) }
        if let e = env { LiteRtDestroyEnvironment(e) }
    }
}

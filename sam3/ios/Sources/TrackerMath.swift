import Foundation

/// Numeric primitives for the tracker host loop, ported 1:1 from the Kotlin port
/// (app/src/main/java/com/sam3/TrackerMath.kt), itself ported from
/// scripts/tracker_host_loop.py. Tensors are row-major float arrays with explicit
/// dims (NCHW like numpy). Binary masks use 64-bit packed bitsets (LSB-first bit
/// order, matching np.packbits(bitorder="little") read little-endian).
enum TM {
    static func sigmoid(_ x: Float) -> Float {
        Float(1.0 / (1.0 + exp(-Double(x))))
    }

    // ---------------- fp16 / bf16 (arm64 Float16 rounds RNE, same as numpy)
    static func toHalfBits(_ v: Float) -> UInt16 { Float16(v).bitPattern }

    static func halfBitsToFloat(_ h: UInt16) -> Float { Float(Float16(bitPattern: h)) }

    /// float32 -> bfloat16 -> float32 in place (round-to-nearest-even, like torch).
    static func bf16InPlace(_ x: inout [Float]) {
        x.withUnsafeMutableBufferPointer { p in
            for i in 0..<p.count {
                let b = p[i].bitPattern
                let r = ((b >> 16) & 1) &+ 0x7FFF
                p[i] = Float(bitPattern: (b &+ r) & 0xFFFF_0000)
            }
        }
    }

    // ---------------- bilinear interpolation (align_corners=False, no antialias)
    private struct AxisIdx {
        let i0: [Int]
        let i1: [Int]
        let lam: [Float]
    }

    private static func axisIdx(_ iSz: Int, _ oSz: Int) -> AxisIdx {
        var i0 = [Int](repeating: 0, count: oSz)
        var i1 = [Int](repeating: 0, count: oSz)
        var lam = [Float](repeating: 0, count: oSz)
        for o in 0..<oSz {
            var src = (Double(o) + 0.5) * (Double(iSz) / Double(oSz)) - 0.5
            if src < 0 { src = 0 }
            let f = min(Int(src.rounded(.down)), iSz - 1)
            i0[o] = f
            i1[o] = min(f + 1, iSz - 1)
            lam[o] = Float(src - Double(f))
        }
        return AxisIdx(i0: i0, i1: i1, lam: lam)
    }

    /// torch F.interpolate(mode="bilinear", align_corners=False), planes-major input
    /// (n*c, ih, iw) -> (n*c, oh, ow). Rows first, then columns, f32 arithmetic —
    /// the same evaluation order as the numpy reference.
    static func interpBilinear(_ x: [Float], _ planes: Int, _ ih: Int, _ iw: Int,
                               _ oh: Int, _ ow: Int) -> [Float] {
        if ih == oh && iw == ow { return x }
        let ay = axisIdx(ih, oh)
        let ax = axisIdx(iw, ow)
        var out = [Float](repeating: 0, count: planes * oh * ow)
        var top = [Float](repeating: 0, count: iw)
        x.withUnsafeBufferPointer { xp in
            out.withUnsafeMutableBufferPointer { op in
                top.withUnsafeMutableBufferPointer { tp in
                    for p in 0..<planes {
                        let base = p * ih * iw
                        let obase = p * oh * ow
                        for y in 0..<oh {
                            let r0 = base + ay.i0[y] * iw
                            let r1 = base + ay.i1[y] * iw
                            let ly = ay.lam[y]
                            for xw in 0..<iw { tp[xw] = xp[r0 + xw] * (1 - ly) + xp[r1 + xw] * ly }
                            let orow = obase + y * ow
                            for xo in 0..<ow {
                                let lx = ax.lam[xo]
                                op[orow + xo] = tp[ax.i0[xo]] * (1 - lx) + tp[ax.i1[xo]] * lx
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    // ---------------- antialiased bilinear (triangle filter, PIL-style)
    private struct AAWeights {
        let xmin: [Int]
        let ws: [[Float]]
    }

    private static var aaCache = [Int64: AAWeights]()

    private static func aaWeights(_ iSz: Int, _ oSz: Int) -> AAWeights {
        let key = Int64(iSz) << 32 | Int64(oSz)
        if let c = aaCache[key] { return c }
        let scale = Double(iSz) / Double(oSz)
        let support = max(scale, 1.0)
        var xmin = [Int](repeating: 0, count: oSz)
        var ws = [[Float]]()
        ws.reserveCapacity(oSz)
        for o in 0..<oSz {
            let center = (Double(o) + 0.5) * scale
            let lo = max(0, Int(center - support + 0.5))
            let hi = min(iSz, Int(center + support + 0.5))
            xmin[o] = lo
            var w = [Float](repeating: 0, count: hi - lo)
            var s = 0.0
            for k in lo..<hi {
                let v = max(1.0 - abs((Double(k) + 0.5 - center) / support), 0.0)
                w[k - lo] = Float(v)
                s += v
            }
            if s > 0 { for i in 0..<w.count { w[i] = Float(Double(w[i]) / s) } }
            ws.append(w)
        }
        let r = AAWeights(xmin: xmin, ws: ws)
        aaCache[key] = r
        return r
    }

    /// torch F.interpolate(..., antialias=True), planes-major (n*c, ih, iw).
    static func interpBilinearAA(_ x0: [Float], _ planes: Int, _ ih: Int, _ iw: Int,
                                 _ oh: Int, _ ow: Int) -> [Float] {
        var x = x0
        var h = ih
        if ih != oh {
            let a = aaWeights(ih, oh)
            var out = [Float](repeating: 0, count: planes * oh * iw)
            x.withUnsafeBufferPointer { xp in
                out.withUnsafeMutableBufferPointer { op in
                    for p in 0..<planes {
                        let base = p * ih * iw
                        let obase = p * oh * iw
                        for o in 0..<oh {
                            let w = a.ws[o]
                            let lo = a.xmin[o]
                            let orow = obase + o * iw
                            for k in 0..<w.count {
                                let wk = w[k]
                                let row = base + (lo + k) * iw
                                for c in 0..<iw { op[orow + c] += wk * xp[row + c] }
                            }
                        }
                    }
                }
            }
            x = out
            h = oh
        }
        if iw != ow {
            let a = aaWeights(iw, ow)
            var out = [Float](repeating: 0, count: planes * h * ow)
            x.withUnsafeBufferPointer { xp in
                out.withUnsafeMutableBufferPointer { op in
                    for p in 0..<planes {
                        let base = p * h * iw
                        let obase = p * h * ow
                        for r in 0..<h {
                            let row = base + r * iw
                            let orow = obase + r * ow
                            for o in 0..<ow {
                                let w = a.ws[o]
                                let lo = a.xmin[o]
                                var acc: Float = 0
                                for k in 0..<w.count { acc += w[k] * xp[row + lo + k] }
                                op[orow + o] = acc
                            }
                        }
                    }
                }
            }
            x = out
        }
        return x
    }

    // ---------------- GELU (libm erf: correctly rounded, matches numpy)
    static func geluInPlace(_ x: inout [Float]) {
        for i in 0..<x.count {
            let v = Double(x[i])
            x[i] = Float(0.5 * v * (1.0 + erf(v / 2.0.squareRoot())))
        }
    }

    // ---------------- small NN ops (host-side conv chain, LN, linears)
    /// Conv2d with stride == kernel (non-overlapping blocks), NCHW.
    static func convBlock(_ x: [Float], _ n: Int, _ ci: Int, _ h: Int, _ w: Int,
                          _ wgt: [Float], _ co: Int, _ k: Int, _ bias: [Float]) -> [Float] {
        let oh = h / k
        let ow = w / k
        var out = [Float](repeating: 0, count: n * co * oh * ow)
        x.withUnsafeBufferPointer { xp in
            wgt.withUnsafeBufferPointer { wp in
                out.withUnsafeMutableBufferPointer { op in
                    for b in 0..<n {
                        for o in 0..<co {
                            let obase = (b * co + o) * oh * ow
                            for y in 0..<oh {
                                for xw in 0..<ow {
                                    var acc: Float = 0
                                    for c in 0..<ci {
                                        let ibase = (b * ci + c) * h * w + y * k * w + xw * k
                                        let wbase = (o * ci + c) * k * k
                                        for ky in 0..<k {
                                            for kx in 0..<k {
                                                acc += xp[ibase + ky * w + kx] * wp[wbase + ky * k + kx]
                                            }
                                        }
                                    }
                                    op[obase + y * ow + xw] = acc + bias[o]
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// LayerNorm over the channel axis of NCHW (eps 1e-6), in place.
    static func layerNorm2dInPlace(_ x: inout [Float], _ n: Int, _ c: Int, _ h: Int, _ w: Int,
                                   _ wgt: [Float], _ bias: [Float]) {
        let hw = h * w
        x.withUnsafeMutableBufferPointer { p in
            for b in 0..<n {
                for px in 0..<hw {
                    var mu: Float = 0
                    for ch in 0..<c { mu += p[(b * c + ch) * hw + px] }
                    mu /= Float(c)
                    var va: Float = 0
                    for ch in 0..<c {
                        let d = p[(b * c + ch) * hw + px] - mu
                        va += d * d
                    }
                    va /= Float(c)
                    let inv = Float(1.0 / (Double(va) + 1e-6).squareRoot())
                    for ch in 0..<c {
                        let i = (b * c + ch) * hw + px
                        p[i] = (p[i] - mu) * inv * wgt[ch] + bias[ch]
                    }
                }
            }
        }
    }

    /// x (m,inDim) @ w(out,inDim)^T + b -> (m,out).
    static func linear(_ x: [Float], _ m: Int, _ inDim: Int,
                       _ w: [Float], _ out: Int, _ b: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: m * out)
        x.withUnsafeBufferPointer { xp in
            w.withUnsafeBufferPointer { wp in
                y.withUnsafeMutableBufferPointer { yp in
                    for r in 0..<m {
                        let xb = r * inDim
                        for o in 0..<out {
                            var acc: Float = 0
                            let wb = o * inDim
                            for i in 0..<inDim { acc += xp[xb + i] * wp[wb + i] }
                            yp[r * out + o] = acc + b[o]
                        }
                    }
                }
            }
        }
        return y
    }

    /// Sine PE of get_1d_sine_pe: pos (n) -> (n, dim), [sin | cos].
    static func sine1dPe(_ pos: [Float], _ dim: Int) -> [Float] {
        let peDim = dim / 2
        var out = [Float](repeating: 0, count: pos.count * dim)
        for r in 0..<pos.count {
            for i in 0..<peDim {
                let dimT = pow(10000.0, 2.0 * Double(i / 2) / Double(peDim))
                let v = Double(pos[r]) / dimT
                out[r * dim + i] = Float(sin(v))
                out[r * dim + peDim + i] = Float(cos(v))
            }
        }
        return out
    }

    // ---------------- packed binary masks
    static func packedWords(_ px: Int) -> Int { (px + 63) >> 6 }

    /// Pack (src[offset + i] > thresh) for i in 0..<px, LSB-first.
    static func pack(_ src: [Float], _ offset: Int, _ px: Int, _ thresh: Float) -> [UInt64] {
        var w = [UInt64](repeating: 0, count: packedWords(px))
        src.withUnsafeBufferPointer { sp in
            w.withUnsafeMutableBufferPointer { wp in
                for i in 0..<px {
                    if sp[offset + i] > thresh { wp[i >> 6] |= 1 << UInt64(i & 63) }
                }
            }
        }
        return w
    }

    static func popcount(_ a: [UInt64]) -> Int {
        var s = 0
        for w in a { s += w.nonzeroBitCount }
        return s
    }

    static func popcountAnd(_ a: [UInt64], _ b: [UInt64]) -> Int {
        var s = 0
        for i in 0..<a.count { s += (a[i] & b[i]).nonzeroBitCount }
        return s
    }

    static func popcountOr(_ a: [UInt64], _ b: [UInt64]) -> Int {
        var s = 0
        for i in 0..<a.count { s += (a[i] | b[i]).nonzeroBitCount }
        return s
    }

    static func testBit(_ a: [UInt64], _ i: Int) -> Bool {
        (a[i >> 6] >> UInt64(i & 63)) & 1 == 1
    }
}

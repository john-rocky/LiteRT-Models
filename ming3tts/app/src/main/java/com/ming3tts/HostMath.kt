package com.ming3tts

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** Host-side constants and small math of the graph contract (contract.json, host_mirror.py). */
object HostMath {
    const val HIDDEN = 896
    const val LATENT = 64
    const val KV = 48                 // 24 layers x 2 KV heads, packed
    const val HEAD_DIM = 64
    const val LM_PAST = 511           // LM cache slots (prompt + steps <= 512)
    const val MAX_CACHE_LEN = 512
    const val DEC_PAST = 64           // decoder cache slots per layer (sliding window 64)
    const val DEC_CHUNK = 8           // latents per decoder call
    const val POS_PER_LATENT = 4      // decoder positions per latent (x4 upsample)
    const val HEAD_WIDTH = 3530       // ISTFT head row: 1765 log-magnitudes + 1765 phases
    const val SAMPLE_RATE = 44100
    const val PREFILL_CHUNK = 64
    const val FLOW_STEPS = 10
    const val TEMB = 1024
    const val MASK_NEG = -30000f

    // frame2 'frame' output row [8002]
    const val OUT_LATENT = 0
    const val OUT_ENEXT = 64
    const val OUT_STOP = 960
    const val OUT_Z = 962
    const val OUT_K = 1858
    const val OUT_V = 4930
    const val FRAME_OUT = 8002

    /** transformers Qwen2RotaryEmbedding inv_freq (theta 1e6, dim 64) as torch computes it. */
    val INV_FREQ = floatArrayOf(
        1.0f, 0.6493816375732422f, 0.4216965138912201f, 0.2738419771194458f,
        0.17782793939113617f, 0.11547820270061493f, 0.07498941570520401f, 0.04869675263762474f,
        0.03162277862429619f, 0.02053525112569332f, 0.01333521492779255f, 0.00865964312106371f,
        0.005623413249850273f, 0.003651741426438093f, 0.0023713738191872835f, 0.0015399265103042126f,
        0.0010000000474974513f, 0.0006493816617876291f, 0.00042169648804701865f, 0.00027384195709601045f,
        0.00017782794020604342f, 0.00011547819303814322f, 7.498942431993783e-05f, 4.869675467489287e-05f,
        3.162277425872162e-05f, 2.0535249859676696e-05f, 1.333521413471317e-05f, 8.659643754072022e-06f,
        5.62341347176698e-06f, 3.651741280918941e-06f, 2.3713737391517498e-06f, 1.5399265294036013e-06f,
    )

    /** cos, sin rows [n * 64] for positions start..start+n-1: float32 angle, float64 cos/sin. */
    fun rope(start: Int, n: Int, cosOut: FloatArray, sinOut: FloatArray) {
        for (i in 0 until n) {
            val p = (start + i).toFloat()
            for (j in 0 until 32) {
                val ang = (p * INV_FREQ[j]).toDouble()
                val c = cos(ang).toFloat()
                val s = sin(ang).toFloat()
                cosOut[i * 64 + j] = c
                cosOut[i * 64 + j + 32] = c
                sinOut[i * 64 + j] = s
                sinOut[i * 64 + j + 32] = s
            }
        }
    }

    /** Additive LM mask [nq, 511 + nq]: past slot s valid for s < start, new key j <= query i. */
    fun lmMask(start: Int, nq: Int, out: FloatArray = FloatArray(nq * (LM_PAST + nq))): FloatArray {
        val w = LM_PAST + nq
        out.fill(MASK_NEG)
        for (i in 0 until nq) {
            for (s in 0 until start) out[i * w + s] = 0f
            for (j in 0..i) out[i * w + LM_PAST + j] = 0f
        }
        return out
    }

    /** Decoder chunk mask [nq, 64 + nq]: query p0+i sees absolute j with p0+i-64 < j <= p0+i. */
    fun decMask(p0: Int, nq: Int, keys: Int = 64): FloatArray {
        val w = DEC_PAST + nq
        val m = FloatArray(nq * w) { MASK_NEG }
        for (i in 0 until nq) {
            val q = p0 + i
            for (s in 0 until DEC_PAST) {
                val j = p0 - DEC_PAST + s
                if (j >= 0 && q - keys < j && j <= q) m[i * w + s] = 0f
            }
            for (s in 0 until nq) {
                val j = p0 + s
                if (q - keys < j && j <= q) m[i * w + DEC_PAST + s] = 0f
            }
        }
        return m
    }

    /** softmax(stop_logits)[1] > 0.5 and step > 3 (float32 softmax). Returns (stop, p1). */
    fun stopDecision(l0: Float, l1: Float, step: Int): Pair<Boolean, Float> {
        val m = max(l0, l1)
        val e0 = exp((l0 - m).toDouble()).toFloat()
        val e1 = exp((l1 - m).toDouble()).toFloat()
        val p1 = e1 / (e0 + e1)
        return (p1 > 0.5f && step > 3) to p1
    }

    /** [chunk + 2, 64]: lat[a-1] | lat[a..a+chunk) | lat[a+chunk], indices clamped to the list. */
    fun decLatentWindow(lat: List<FloatArray>, a: Int, chunk: Int, out: FloatArray) {
        val s = lat.size
        fun put(row: Int, idx: Int) = System.arraycopy(lat[idx], 0, out, row * LATENT, LATENT)
        put(0, max(a - 1, 0))
        for (i in 0 until chunk) put(1 + i, min(a + i, s - 1))
        put(chunk + 1, min(a + chunk, s - 1))
    }

    /** IEEE half -> float (exact). */
    fun halfToFloat(h: Int): Float {
        val sign = (h ushr 15) and 1
        val e = (h ushr 10) and 0x1f
        val f = h and 0x3ff
        val bits = when {
            e == 0 && f == 0 -> sign shl 31
            e == 0 -> {
                // subnormal half -> normal float
                var m = f
                var ee = -14
                while (m and 0x400 == 0) {
                    m = m shl 1
                    ee--
                }
                (sign shl 31) or ((ee + 127) shl 23) or ((m and 0x3ff) shl 13)
            }
            e == 0x1f -> (sign shl 31) or (0xff shl 23) or (f shl 13)
            else -> (sign shl 31) or ((e - 15 + 127) shl 23) or (f shl 13)
        }
        return java.lang.Float.intBitsToFloat(bits)
    }

    fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(b.size / 4)
        bb.get(out)
        return out
    }

    fun writeF32(f: File, a: FloatArray, n: Int = a.size) {
        val bb = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN)
        bb.asFloatBuffer().put(a, 0, n)
        f.writeBytes(bb.array())
    }
}

/** stop_head (Linear 896 -> 2) on the host for the split graphs: float64 dot, float32 bias add. */
class StopHead(file: File) {
    private val w: FloatArray
    private val b: FloatArray

    init {
        val all = HostMath.readF32(file)
        require(all.size == 2 * HostMath.HIDDEN + 2) { "stop_head.bin: ${all.size} floats" }
        w = all.copyOfRange(0, 2 * HostMath.HIDDEN)
        b = all.copyOfRange(2 * HostMath.HIDDEN, all.size)
    }

    fun logits(z: FloatArray, zOff: Int = 0): FloatArray {
        val out = FloatArray(2)
        for (o in 0 until 2) {
            var acc = 0.0
            for (i in 0 until HostMath.HIDDEN) acc += w[o * HostMath.HIDDEN + i].toDouble() * z[zOff + i]
            out[o] = acc.toFloat() + b[o]
        }
        return out
    }
}

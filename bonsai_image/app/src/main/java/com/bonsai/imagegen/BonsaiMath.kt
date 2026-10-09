// Copyright 2026 Daisuke Majima. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Host-side math for the Bonsai pipeline (port of the iOS app's
// BonsaiMath.swift), kept Android-free so it can be cross-checked on the JVM
// against the recorded pipeline fixtures. Every function takes the patch grid
// of the output size (512x512 -> 32, 256x256 -> 16); the defaults are the
// original 512x512 constants.

package com.bonsai.imagegen

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

object BonsaiMath {
    const val SEQ = 256
    const val TOKENS = 1024        // 512x512 -> 32x32 patch grid
    const val LAT_GRID = 32
    const val PACKED_CHANNELS = 128
    const val LATENT_CHANNELS = 32 // VAE latent channels; 2x2 patchify packs them to 128

    /** Patch-grid side for an output size: one image token per 16x16 pixels
     *  (8x VAE downsampling, then 2x2 patchify). 512 -> 32, 256 -> 16. */
    fun gridFor(size: Int): Int {
        require(size > 0 && size % 16 == 0) { "output size must be a multiple of 16: $size" }
        return size / 16
    }

    /** FLUX.2-klein sigma schedule (generate.py flowmatch_sigmas): linspace
     *  shifted by the empirical mu (a function of the image-token count),
     *  exponential time-shift, timestep == sigma. steps=4 at 1024 tokens
     *  reproduces the device-fixture manifest sigmas exactly. */
    fun sigmas(steps: Int, tokens: Int = TOKENS): FloatArray {
        val m200 = 0.00016927 * tokens + 0.45666666
        val m10 = 8.73809524e-05 * tokens + 1.89833333
        val a = (m200 - m10) / 190.0
        val mu = a * steps + (m200 - 200.0 * a)
        val out = FloatArray(steps + 1)
        for (i in 0 until steps) {
            val lin = 1.0 - i * (1.0 - 1.0 / steps) / maxOf(steps - 1, 1)
            out[i] = (exp(mu) / (exp(mu) + (1.0 / lin - 1.0))).toFloat()
        }
        return out
    }

    /** Image-token position ids: [0, h, w, 0] over the grid x grid patch grid
     *  -> (grid*grid, 4). */
    fun imgIds(grid: Int = LAT_GRID): FloatArray {
        val out = FloatArray(grid * grid * 4)
        for (h in 0 until grid) for (w in 0 until grid) {
            val base = (h * grid + w) * 4
            out[base + 1] = h.toFloat()
            out[base + 2] = w.toFloat()
        }
        return out
    }

    /** Text-token position ids: [0, 0, 0, i] -> (256, 4). */
    fun txtIds(): FloatArray {
        val out = FloatArray(SEQ * 4)
        for (i in 0 until SEQ) out[i * 4 + 3] = i.toFloat()
        return out
    }

    /** Seeded standard-normal noise (1, tokens, 128). SplitMix64 + Box-Muller —
     *  the SAME stream as the iOS app (identical algorithm and constants), so
     *  (prompt, seed, steps) reproduces the same image across platforms. A
     *  smaller size takes the head of the same stream: noise(seed, 256) ==
     *  the first 256*128 values of noise(seed, 1024). */
    fun noise(seed: Long, tokens: Int = TOKENS): FloatArray {
        var state = seed
        fun next(): Long {
            state += -0x61c8864680b583ebL          // 0x9E3779B97F4A7C15
            var z = state
            z = (z xor (z ushr 30)) * -0x40a7b892e31b1a47L   // 0xBF58476D1CE4E5B9
            z = (z xor (z ushr 27)) * -0x6b2fb644ecceee15L   // 0x94D049BB133111EB
            return z xor (z ushr 31)
        }
        // (0, 1], never 0 for ln(); (next() ushr 11) is uniform in [0, 2^53)
        fun uniform(): Double = ((next() ushr 11) + 1.0) / 9007199254740993.0
        val n = tokens * PACKED_CHANNELS
        val out = FloatArray(n)
        var i = 0
        while (i < n) {
            val r = sqrt(-2.0 * ln(uniform()))
            val theta = 2.0 * Math.PI * uniform()
            out[i] = (r * cos(theta)).toFloat()
            if (i + 1 < n) out[i + 1] = (r * sin(theta)).toFloat()
            i += 2
        }
        return out
    }

    /** (grid*grid, 128) packed tokens -> (1, 32, 2*grid, 2*grid) VAE latent:
     *  per-PACKED-channel BN affine first, then the 2x2 patch unfold (packed
     *  channel m = c*4 + i*2 + j lands at z[c, 2h+i, 2w+j] for token (h, w)). */
    fun unpatchify(
        lat: FloatArray, scale: FloatArray, shift: FloatArray, grid: Int = LAT_GRID,
    ): FloatArray {
        require(lat.size == grid * grid * PACKED_CHANNELS) {
            "latent has ${lat.size} values, grid $grid needs ${grid * grid * PACKED_CHANNELS}"
        }
        val side = 2 * grid
        val plane = side * side
        val z = FloatArray(LATENT_CHANNELS * plane)
        for (h in 0 until grid) for (w in 0 until grid) {
            val base = (h * grid + w) * PACKED_CHANNELS
            for (c in 0 until LATENT_CHANNELS) for (i in 0..1) for (j in 0..1) {
                val m = c * 4 + i * 2 + j
                z[c * plane + (2 * h + i) * side + (2 * w + j)] =
                    scale[m] * lat[base + m] + shift[m]
            }
        }
        return z
    }
}

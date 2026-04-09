package com.yolotracking

/**
 * 8-state linear Kalman filter for bounding box tracking.
 * State: [cx, cy, a, h, vx, vy, va, vh]
 *   (cx,cy) = center, a = aspect ratio, h = height, v* = velocities
 * Measurement: [cx, cy, a, h]
 *
 * Follows the original DeepSORT kalman_filter.py implementation.
 */
class KalmanFilter {

    companion object {
        private const val N = 8   // state dim
        private const val M = 4   // measurement dim

        // Noise weights relative to current height
        private const val STD_WEIGHT_POS = 1f / 20f
        private const val STD_WEIGHT_VEL = 1f / 160f

        // Chi-square 95th percentile for 4-DOF (gating threshold)
        const val CHI2_THRESHOLD_4D = 9.4877f
    }

    /** State vector [8] */
    val x = FloatArray(N)

    /** Covariance matrix [8x8] stored row-major */
    val P = FloatArray(N * N)

    fun initiate(cx: Float, cy: Float, a: Float, h: Float) {
        x[0] = cx; x[1] = cy; x[2] = a; x[3] = h
        x[4] = 0f; x[5] = 0f; x[6] = 0f; x[7] = 0f

        val stdPos = floatArrayOf(
            2f * STD_WEIGHT_POS * h,
            2f * STD_WEIGHT_POS * h,
            1e-2f,
            2f * STD_WEIGHT_POS * h,
        )
        val stdVel = floatArrayOf(
            10f * STD_WEIGHT_VEL * h,
            10f * STD_WEIGHT_VEL * h,
            1e-5f,
            10f * STD_WEIGHT_VEL * h,
        )

        P.fill(0f)
        for (i in 0 until M) P[i * N + i] = stdPos[i] * stdPos[i]
        for (i in 0 until M) P[(M + i) * N + (M + i)] = stdVel[i] * stdVel[i]
    }

    fun predict() {
        // x = F * x  (constant velocity: pos += vel)
        for (i in 0 until M) x[i] += x[M + i]

        val h = x[3]
        val stdPos = floatArrayOf(
            STD_WEIGHT_POS * h,
            STD_WEIGHT_POS * h,
            1e-2f,
            STD_WEIGHT_POS * h,
        )
        val stdVel = floatArrayOf(
            STD_WEIGHT_VEL * h,
            STD_WEIGHT_VEL * h,
            1e-5f,
            STD_WEIGHT_VEL * h,
        )

        // P = F * P * F^T + Q
        // F is identity with upper-right I4 block: F[i][M+i] = 1
        // P' = F P F^T computed in-place
        val tmp = FloatArray(N * N)

        // tmp = F * P
        for (r in 0 until N) {
            for (c in 0 until N) {
                var v = P[r * N + c]
                if (r < M) v += P[(r + M) * N + c]
                tmp[r * N + c] = v
            }
        }
        // P = tmp * F^T
        for (r in 0 until N) {
            for (c in 0 until N) {
                var v = tmp[r * N + c]
                if (c < M) v += tmp[r * N + (c + M)]
                P[r * N + c] = v
            }
        }

        // Add process noise Q (diagonal)
        for (i in 0 until M) P[i * N + i] += stdPos[i] * stdPos[i]
        for (i in 0 until M) P[(M + i) * N + (M + i)] += stdVel[i] * stdVel[i]
    }

    fun update(cx: Float, cy: Float, a: Float, h: Float) {
        val z = floatArrayOf(cx, cy, a, h)

        // Measurement noise R (diagonal)
        val stdR = floatArrayOf(
            STD_WEIGHT_POS * x[3],
            STD_WEIGHT_POS * x[3],
            1e-1f,
            STD_WEIGHT_POS * x[3],
        )

        // Innovation y = z - H * x  (H picks first M states)
        val y = FloatArray(M) { z[it] - x[it] }

        // S = H * P * H^T + R  (M x M)
        val S = FloatArray(M * M)
        for (r in 0 until M) for (c in 0 until M) S[r * M + c] = P[r * N + c]
        for (i in 0 until M) S[i * M + i] += stdR[i] * stdR[i]

        // K = P * H^T * S^{-1}  (N x M)
        val Sinv = invert4x4(S)
        val K = FloatArray(N * M)
        for (r in 0 until N) {
            for (c in 0 until M) {
                var v = 0f
                for (k in 0 until M) v += P[r * N + k] * Sinv[k * M + c]
                K[r * M + c] = v
            }
        }

        // x = x + K * y
        for (r in 0 until N) {
            var v = 0f
            for (c in 0 until M) v += K[r * M + c] * y[c]
            x[r] += v
        }

        // P = (I - K * H) * P
        val IKH = FloatArray(N * N)
        for (i in 0 until N) IKH[i * N + i] = 1f
        for (r in 0 until N) for (c in 0 until M) IKH[r * N + c] -= K[r * M + c]

        val Pnew = FloatArray(N * N)
        for (r in 0 until N) for (c in 0 until N) {
            var v = 0f
            for (k in 0 until N) v += IKH[r * N + k] * P[k * N + c]
            Pnew[r * N + c] = v
        }
        System.arraycopy(Pnew, 0, P, 0, N * N)
    }

    /**
     * Squared Mahalanobis distance between measurement and predicted state.
     */
    fun mahalanobisDistance(cx: Float, cy: Float, a: Float, h: Float): Float {
        val z = floatArrayOf(cx, cy, a, h)
        val y = FloatArray(M) { z[it] - x[it] }

        val stdR = floatArrayOf(
            STD_WEIGHT_POS * x[3],
            STD_WEIGHT_POS * x[3],
            1e-1f,
            STD_WEIGHT_POS * x[3],
        )
        val S = FloatArray(M * M)
        for (r in 0 until M) for (c in 0 until M) S[r * M + c] = P[r * N + c]
        for (i in 0 until M) S[i * M + i] += stdR[i] * stdR[i]

        val Sinv = invert4x4(S)
        // d^2 = y^T S^{-1} y
        var dist = 0f
        for (i in 0 until M) {
            var v = 0f
            for (j in 0 until M) v += Sinv[i * M + j] * y[j]
            dist += y[i] * v
        }
        return dist
    }

    private fun invert4x4(m: FloatArray): FloatArray {
        // Analytical 4x4 matrix inversion via adjugate
        val inv = FloatArray(16)
        val a = m; val n = 4

        inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10]
        inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10]
        inv[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9]
        inv[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9]

        inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10]
        inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10]
        inv[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9]
        inv[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9]

        inv[2]  =  a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6]
        inv[6]  = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6]
        inv[10] =  a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5]
        inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5]

        inv[3]  = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]  + a[9]*a[3]*a[6]
        inv[7]  =  a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]  - a[8]*a[3]*a[6]
        inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9]  + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]  + a[8]*a[3]*a[5]
        inv[15] =  a[0]*a[5]*a[10] - a[0]*a[6]*a[9]  - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]  - a[8]*a[2]*a[5]

        val det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12]
        if (det == 0f) return FloatArray(16) // singular
        val invDet = 1f / det
        for (i in 0 until 16) inv[i] *= invDet
        return inv
    }
}

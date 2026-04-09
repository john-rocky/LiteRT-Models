package com.yolotracking

/**
 * Hungarian algorithm (Munkres) for optimal assignment.
 * Given an NxM cost matrix, finds the minimum-cost assignment.
 * Returns list of (row, col) pairs.
 *
 * Implementation uses the classic O(n^3) algorithm.
 */
object HungarianAlgorithm {

    fun solve(costMatrix: Array<FloatArray>): List<Pair<Int, Int>> {
        val nRows = costMatrix.size
        if (nRows == 0) return emptyList()
        val nCols = costMatrix[0].size
        if (nCols == 0) return emptyList()

        // Pad to square
        val n = maxOf(nRows, nCols)
        val cost = Array(n) { r ->
            FloatArray(n) { c ->
                if (r < nRows && c < nCols) costMatrix[r][c] else 0f
            }
        }

        val u = FloatArray(n + 1)  // potential for rows
        val v = FloatArray(n + 1)  // potential for cols
        val p = IntArray(n + 1)    // col -> row assignment
        val way = IntArray(n + 1)  // augmenting path

        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = FloatArray(n + 1) { Float.MAX_VALUE }
            val used = BooleanArray(n + 1)

            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Float.MAX_VALUE
                var j1 = 0

                for (j in 1..n) {
                    if (used[j]) continue
                    val cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if (cur < minv[j]) {
                        minv[j] = cur
                        way[j] = j0
                    }
                    if (minv[j] < delta) {
                        delta = minv[j]
                        j1 = j
                    }
                }

                for (j in 0..n) {
                    if (used[j]) {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }
                j0 = j1
            } while (p[j0] != 0)

            // Augment path
            do {
                val j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while (j0 != 0)
        }

        // Extract results (only original-sized assignments)
        val result = mutableListOf<Pair<Int, Int>>()
        for (j in 1..n) {
            if (p[j] != 0 && p[j] - 1 < nRows && j - 1 < nCols) {
                result.add(p[j] - 1 to j - 1)
            }
        }
        return result
    }
}

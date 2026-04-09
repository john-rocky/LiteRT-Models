package com.yolotracking

/**
 * Single tracked object with Kalman state and appearance feature gallery.
 */
class Track(
    val trackId: Int,
    detection: Detection,
    private val nnBudget: Int = 100,
) {
    enum class State { TENTATIVE, CONFIRMED, DELETED }

    var state = State.TENTATIVE
        private set

    val kf = KalmanFilter()

    var classId: Int = detection.classId
        private set
    var className: String = detection.className
        private set

    var timeSinceUpdate = 0
        private set
    var hits = 1
        private set

    // Appearance feature gallery
    private val features = mutableListOf<FloatArray>()

    // Trail of recent center positions for visualization
    val trail = ArrayDeque<Pair<Float, Float>>(TRAIL_LENGTH + 1)

    init {
        val a = detection.w / detection.h.coerceAtLeast(1f)
        kf.initiate(detection.cx, detection.cy, a, detection.h)
        detection.feature?.let { features.add(it.copyOf()) }
        trail.addLast(detection.cx to detection.cy)
    }

    fun predict() {
        kf.predict()
        timeSinceUpdate++
    }

    fun update(detection: Detection) {
        val a = detection.w / detection.h.coerceAtLeast(1f)
        kf.update(detection.cx, detection.cy, a, detection.h)
        detection.feature?.let {
            features.add(it.copyOf())
            if (features.size > nnBudget) features.removeAt(0)
        }
        classId = detection.classId
        className = detection.className
        hits++
        timeSinceUpdate = 0
        trail.addLast(detection.cx to detection.cy)
        if (trail.size > TRAIL_LENGTH) trail.removeFirst()

        if (state == State.TENTATIVE && hits >= N_INIT) {
            state = State.CONFIRMED
        }
    }

    fun markDeleted() {
        state = State.DELETED
    }

    fun shouldDelete(maxAge: Int): Boolean {
        return state == State.DELETED ||
            (state == State.TENTATIVE && timeSinceUpdate > 0) ||
            timeSinceUpdate > maxAge
    }

    /** Predicted bounding box from Kalman state: [cx, cy, w, h] */
    val predictedBox: FloatArray
        get() {
            val cx = kf.x[0]; val cy = kf.x[1]
            val a = kf.x[2]; val h = kf.x[3].coerceAtLeast(1f)
            val w = a * h
            return floatArrayOf(cx, cy, w, h)
        }

    /**
     * Minimum cosine distance between given feature and stored gallery.
     * Lower = more similar.
     */
    fun cosineDistance(queryFeature: FloatArray): Float {
        if (features.isEmpty()) return 1f
        var minDist = Float.MAX_VALUE
        for (stored in features) {
            val dist = 1f - cosineSimilarity(queryFeature, stored)
            if (dist < minDist) minDist = dist
        }
        return minDist
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var normA = 0f; var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB)
        return if (denom > 1e-8f) dot / denom else 0f
    }

    companion object {
        const val N_INIT = 3
        const val TRAIL_LENGTH = 30
    }
}

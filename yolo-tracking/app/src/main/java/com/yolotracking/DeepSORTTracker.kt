package com.yolotracking

/**
 * DeepSORT multi-object tracker.
 *
 * Pipeline per frame:
 *   1. predict() all existing tracks (Kalman)
 *   2. Matching cascade: appearance (cosine) + Mahalanobis gate on confirmed tracks
 *   3. IOU matching on remaining unconfirmed + unmatched confirmed tracks
 *   4. Create new tracks for unmatched detections
 *   5. Delete stale tracks
 */
class DeepSORTTracker(
    private val maxAge: Int = 30,
    private val maxCosineDist: Float = 0.4f,
    private val maxIouDist: Float = 0.7f,
) {
    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    val activeTracks: List<Track>
        get() = tracks.filter { it.state != Track.State.DELETED }

    fun update(detections: List<Detection>): List<Detection> {
        // 1. Predict all tracks
        for (track in tracks) track.predict()

        // 2. Split tracks
        val confirmed = tracks.filter { it.state == Track.State.CONFIRMED }
        val unconfirmed = tracks.filter { it.state == Track.State.TENTATIVE }

        // 3. Matching cascade on confirmed tracks (appearance + Mahalanobis gate)
        val (matchedA, unmatchedTracksA, unmatchedDetsA) =
            matchingCascade(confirmed, detections)

        // 4. IOU matching on remaining
        val iouCandidateTracks = unmatchedTracksA.filter { it.timeSinceUpdate == 1 } + unconfirmed
        val iouCandidateDets = unmatchedDetsA

        val (matchedB, unmatchedTracksB, unmatchedDetsB) =
            iouMatching(iouCandidateTracks, detections, iouCandidateDets)

        // 5. Apply matches
        for ((track, detIdx) in matchedA) track.update(detections[detIdx])
        for ((track, detIdx) in matchedB) track.update(detections[detIdx])

        // 6. Delete unmatched tracks
        val toDeleteFromA = unmatchedTracksA.filter { it.timeSinceUpdate > 0 || it.state == Track.State.TENTATIVE }
        for (track in toDeleteFromA) if (track.shouldDelete(maxAge)) track.markDeleted()
        for (track in unmatchedTracksB) if (track.shouldDelete(maxAge)) track.markDeleted()

        // 7. Create new tracks for unmatched detections
        for (detIdx in unmatchedDetsB) {
            tracks.add(Track(nextId++, detections[detIdx]))
        }

        // 8. Remove deleted tracks
        tracks.removeAll { it.state == Track.State.DELETED }

        // 9. Build output: confirmed tracks mapped to Detection with trackId
        return tracks.filter { it.state == Track.State.CONFIRMED }.map { track ->
            val box = track.predictedBox
            val h = box[3].coerceAtLeast(1f)
            val w = box[2]
            Detection(
                classId = track.classId,
                className = track.className,
                score = 1f,
                xMin = box[0] - w / 2f,
                yMin = box[1] - h / 2f,
                xMax = box[0] + w / 2f,
                yMax = box[1] + h / 2f,
                trackId = track.trackId,
            )
        }
    }

    // --- Matching cascade (DeepSORT core) ---

    private data class MatchResult(
        val matched: List<Pair<Track, Int>>,     // (track, detectionIndex)
        val unmatchedTracks: List<Track>,
        val unmatchedDetIndices: List<Int>,
    )

    private fun matchingCascade(
        tracks: List<Track>,
        detections: List<Detection>,
    ): MatchResult {
        val matched = mutableListOf<Pair<Track, Int>>()
        var unmatchedDetIndices = detections.indices.toMutableList()
        val matchedTrackIds = mutableSetOf<Int>()

        // Iterate by cascade depth: recently seen tracks get priority
        for (level in 0 until maxAge) {
            if (unmatchedDetIndices.isEmpty()) break

            val levelTracks = tracks.filter {
                it.timeSinceUpdate == 1 + level && it.trackId !in matchedTrackIds
            }
            if (levelTracks.isEmpty()) continue

            val (levelMatched, _, remainingDets) =
                minCostMatching(levelTracks, detections, unmatchedDetIndices, useAppearance = true)

            for ((track, detIdx) in levelMatched) {
                matched.add(track to detIdx)
                matchedTrackIds.add(track.trackId)
            }
            unmatchedDetIndices = remainingDets.toMutableList()
        }

        val unmatchedTracks = tracks.filter { it.trackId !in matchedTrackIds }
        return MatchResult(matched, unmatchedTracks, unmatchedDetIndices)
    }

    private fun iouMatching(
        tracks: List<Track>,
        allDetections: List<Detection>,
        detIndices: List<Int>,
    ): MatchResult {
        return minCostMatching(tracks, allDetections, detIndices, useAppearance = false)
    }

    private fun minCostMatching(
        tracks: List<Track>,
        allDetections: List<Detection>,
        detIndices: List<Int>,
        useAppearance: Boolean,
    ): MatchResult {
        if (tracks.isEmpty() || detIndices.isEmpty()) {
            return MatchResult(emptyList(), tracks, detIndices)
        }

        val threshold = if (useAppearance) maxCosineDist else maxIouDist

        // Build cost matrix [tracks x detIndices]
        val costMatrix = Array(tracks.size) { t ->
            FloatArray(detIndices.size) { d ->
                val det = allDetections[detIndices[d]]
                if (useAppearance) {
                    val feature = det.feature
                    val cosDist = if (feature != null) tracks[t].cosineDistance(feature) else 1f
                    // Gate with Mahalanobis distance
                    val a = det.w / det.h.coerceAtLeast(1f)
                    val mahaDist = tracks[t].kf.mahalanobisDistance(det.cx, det.cy, a, det.h)
                    if (mahaDist > KalmanFilter.CHI2_THRESHOLD_4D) INFTY_COST else cosDist
                } else {
                    // IOU distance
                    1f - iou(tracks[t], det)
                }
            }
        }

        val assignments = HungarianAlgorithm.solve(costMatrix)

        val matched = mutableListOf<Pair<Track, Int>>()
        val matchedTrackIdx = mutableSetOf<Int>()
        val matchedDetIdx = mutableSetOf<Int>()

        for ((tIdx, dIdx) in assignments) {
            if (costMatrix[tIdx][dIdx] <= threshold) {
                matched.add(tracks[tIdx] to detIndices[dIdx])
                matchedTrackIdx.add(tIdx)
                matchedDetIdx.add(dIdx)
            }
        }

        val unmatchedTracks = tracks.filterIndexed { idx, _ -> idx !in matchedTrackIdx }
        val unmatchedDets = detIndices.filterIndexed { idx, _ -> idx !in matchedDetIdx }

        return MatchResult(matched, unmatchedTracks, unmatchedDets)
    }

    private fun iou(track: Track, det: Detection): Float {
        val box = track.predictedBox
        val h = box[3].coerceAtLeast(1f); val w = box[2]
        val tXMin = box[0] - w / 2f; val tYMin = box[1] - h / 2f
        val tXMax = box[0] + w / 2f; val tYMax = box[1] + h / 2f

        val x1 = maxOf(tXMin, det.xMin); val y1 = maxOf(tYMin, det.yMin)
        val x2 = minOf(tXMax, det.xMax); val y2 = minOf(tYMax, det.yMax)
        val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val areaT = (tXMax - tXMin) * (tYMax - tYMin)
        val areaD = det.w * det.h
        return inter / (areaT + areaD - inter + 1e-6f)
    }

    companion object {
        private const val INFTY_COST = 1e5f
    }
}

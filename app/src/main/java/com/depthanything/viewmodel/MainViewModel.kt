package com.depthanything.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.depthanything.ml.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

data class BenchmarkEntry(
    val mode: InferenceMode,
    val initTimeMs: Long,
    val firstRunMs: Long,
    val avgMs: Long,
    val depthBitmap: Bitmap?,
    val error: String? = null
)

data class UiState(
    val isLoading: Boolean = false,
    val loadingMessage: String = "",
    val inputBitmap: Bitmap? = null,
    val depthBitmap: Bitmap? = null,
    val coloredDepthBitmap: Bitmap? = null,
    val inferenceTimeMs: Long = 0,
    val currentMode: InferenceMode = InferenceMode.ONNX_CPU,
    val availableModes: List<InferenceMode> = emptyList(),
    val error: String? = null,
    val benchmarkEntries: List<BenchmarkEntry> = emptyList()
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "DepthAnything"
    }

    private val _uiState = MutableStateFlow(UiState())
    val uiState: StateFlow<UiState> = _uiState

    private var estimator: DepthEstimator? = null

    init {
        val modes = DepthEstimatorFactory.availableModes(application)
        Log.i(TAG, "Available modes: ${modes.map { it.label }}")
        val defaultMode = when {
            InferenceMode.ONNX_CPU in modes -> InferenceMode.ONNX_CPU
            modes.isNotEmpty() -> modes.first()
            else -> InferenceMode.ONNX_CPU
        }
        _uiState.value = _uiState.value.copy(
            availableModes = modes,
            currentMode = defaultMode
        )
    }

    fun switchMode(mode: InferenceMode) {
        if (mode == _uiState.value.currentMode && estimator != null) return
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null,
                loadingMessage = "Loading ${mode.label}...")
            try {
                estimator?.close()
                val initStart = System.nanoTime()
                estimator = DepthEstimatorFactory.create(getApplication(), mode)
                val initMs = (System.nanoTime() - initStart) / 1_000_000
                Log.i(TAG, "[${mode.label}] Model initialized in ${initMs} ms")
                _uiState.value = _uiState.value.copy(
                    currentMode = mode,
                    isLoading = false
                )
                _uiState.value.inputBitmap?.let { runPrediction(it) }
            } catch (e: Exception) {
                Log.e(TAG, "[${mode.label}] Init failed", e)
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Failed to init ${mode.label}: ${e.message}"
                )
            }
        }
    }

    fun loadImage(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                @Suppress("DEPRECATION")
                val bitmap = MediaStore.Images.Media.getBitmap(
                    getApplication<Application>().contentResolver, uri
                )
                Log.i(TAG, "Image loaded: ${bitmap.width}x${bitmap.height}")
                _uiState.value = _uiState.value.copy(inputBitmap = bitmap)
                ensureEstimator()
                runPrediction(bitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load image", e)
                _uiState.value = _uiState.value.copy(error = "Failed to load image: ${e.message}")
            }
        }
    }

    fun processFrame(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            ensureEstimator()
            runPrediction(bitmap)
        }
    }

    fun runBenchmark() {
        val inputBitmap = _uiState.value.inputBitmap ?: return
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(
                isLoading = true, benchmarkEntries = emptyList()
            )
            val entries = mutableListOf<BenchmarkEntry>()

            Log.i(TAG, "========== BENCHMARK START ==========")
            Log.i(TAG, "Input: ${inputBitmap.width}x${inputBitmap.height}")

            for (mode in _uiState.value.availableModes) {
                _uiState.value = _uiState.value.copy(
                    loadingMessage = "Benchmarking ${mode.label}..."
                )
                Log.i(TAG, "--- ${mode.label} ---")
                try {
                    // Init
                    val initStart = System.nanoTime()
                    val est = DepthEstimatorFactory.create(getApplication(), mode)
                    val initMs = (System.nanoTime() - initStart) / 1_000_000
                    Log.i(TAG, "  Init: ${initMs} ms")

                    // First run (cold)
                    val firstResult = est.predict(inputBitmap)
                    Log.i(TAG, "  1st run (cold): ${firstResult.inferenceTimeMs} ms")

                    // Subsequent runs
                    val runs = 3
                    var totalMs = 0L
                    var lastResult = firstResult
                    for (i in 1..runs) {
                        lastResult = est.predict(inputBitmap)
                        totalMs += lastResult.inferenceTimeMs
                        Log.i(TAG, "  Run ${i + 1}: ${lastResult.inferenceTimeMs} ms")
                    }
                    val avgMs = totalMs / runs
                    Log.i(TAG, "  Avg (runs 2-4): ${avgMs} ms")

                    val colored = Colormap.applyInferno(lastResult.depthMap)
                    entries.add(BenchmarkEntry(
                        mode = mode,
                        initTimeMs = initMs,
                        firstRunMs = firstResult.inferenceTimeMs,
                        avgMs = avgMs,
                        depthBitmap = colored
                    ))
                    est.close()
                } catch (e: Exception) {
                    Log.e(TAG, "  FAILED: ${e.message}", e)
                    entries.add(BenchmarkEntry(
                        mode = mode,
                        initTimeMs = -1,
                        firstRunMs = -1,
                        avgMs = -1,
                        depthBitmap = null,
                        error = e.message
                    ))
                }
            }

            Log.i(TAG, "========== BENCHMARK SUMMARY ==========")
            for (entry in entries) {
                if (entry.error != null) {
                    Log.i(TAG, "  ${entry.mode.label}: FAILED (${entry.error})")
                } else {
                    Log.i(TAG, "  ${entry.mode.label}: init=${entry.initTimeMs}ms, " +
                            "1st=${entry.firstRunMs}ms, avg=${entry.avgMs}ms")
                }
            }
            Log.i(TAG, "=======================================")

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                benchmarkEntries = entries
            )
        }
    }

    private fun ensureEstimator() {
        if (estimator == null) {
            val mode = _uiState.value.currentMode
            Log.i(TAG, "Creating estimator: ${mode.label}")
            estimator = DepthEstimatorFactory.create(getApplication(), mode)
        }
    }

    private fun runPrediction(bitmap: Bitmap) {
        try {
            val est = estimator ?: return
            Log.i(TAG, "[${est.mode.label}] Running prediction...")
            val result = est.predict(bitmap)
            Log.i(TAG, "[${est.mode.label}] Inference: ${result.inferenceTimeMs} ms")
            val colored = Colormap.applyInferno(result.depthMap)
            _uiState.value = _uiState.value.copy(
                depthBitmap = result.depthMap,
                coloredDepthBitmap = colored,
                inferenceTimeMs = result.inferenceTimeMs,
                error = null
            )
        } catch (e: Exception) {
            Log.e(TAG, "Prediction failed", e)
            _uiState.value = _uiState.value.copy(
                error = "Inference error: ${e.message}"
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        estimator?.close()
    }
}

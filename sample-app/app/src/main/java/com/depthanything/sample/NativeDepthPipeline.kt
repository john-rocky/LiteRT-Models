package com.depthanything.sample

import android.content.res.AssetManager
import android.util.Log

/**
 * JNI bridge to C++ zero-copy depth pipeline.
 * Uses Kotlin CompiledModel (FP32) handle + C API SSBO tensor buffers.
 */
class NativeDepthPipeline : AutoCloseable {

    /**
     * Read output TensorBuffer via C++ LiteRtLockTensorBuffer (bypass Kotlin readFloat),
     * apply Inferno colormap, write ARGB pixels to outputPixels array.
     * Returns lock time in ms (to compare with Kotlin readFloat's 340ms).
     */
    external fun nativeLockAndColormap(
        tensorBufferHandle: Long,
        outputPixels: IntArray,
        width: Int, height: Int
    ): Long

    // Keep old methods for compatibility
    external fun nativeInitGl(inputW: Int, inputH: Int, outputW: Int, outputH: Int): Boolean
    external fun nativeSetHandles(envHandle: Long, compiledModelHandle: Long): Boolean
    external fun nativeProcessFrame(pixels: IntArray, width: Int, height: Int, rotation: Int)
    external fun nativeRender(viewWidth: Int, viewHeight: Int)
    external fun nativeDestroy()
    external fun nativeIsInitialized(): Boolean
    external fun nativeIsZeroCopy(): Boolean

    override fun close() = nativeDestroy()

    companion object {
        init {
            System.loadLibrary("depth_pipeline")
        }

        /** Extract native handle from JniHandle subclass via reflection */
        fun getNativeHandle(obj: Any): Long {
            return try {
                val field = obj.javaClass.superclass?.getDeclaredField("handle")
                    ?: obj.javaClass.getDeclaredField("handle")
                field.isAccessible = true
                field.getLong(obj)
            } catch (e: Exception) {
                Log.e("NativeDepthPipeline", "Failed to get native handle", e)
                0L
            }
        }
    }
}

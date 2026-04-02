package com.depthanything.sample

import android.content.res.AssetManager
import android.util.Log

class NativeDepthPipeline : AutoCloseable {

    external fun nativeInitGl(inputW: Int, inputH: Int, outputW: Int, outputH: Int): Boolean
    external fun nativeInitLiteRT(
        assetManager: AssetManager, modelPath: String,
        inputW: Int, inputH: Int, outputW: Int, outputH: Int
    ): Boolean
    external fun nativeProcessFrame(pixels: IntArray, width: Int, height: Int, rotation: Int)
    external fun nativeRender(viewWidth: Int, viewHeight: Int)
    external fun nativeDestroy()
    external fun nativeIsInitialized(): Boolean
    external fun nativeIsZeroCopy(): Boolean

    external fun nativeLockAndColormap(
        tensorBufferHandle: Long, outputPixels: IntArray, width: Int, height: Int
    ): Long

    override fun close() = nativeDestroy()

    companion object {
        init { System.loadLibrary("depth_pipeline") }

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

package com.depthanything.sample

import android.content.res.AssetManager

/**
 * JNI bridge to C++ zero-copy depth pipeline.
 * Uses LiteRT C API + OpenGL SSBO for GPU-resident inference.
 */
class NativeDepthPipeline : AutoCloseable {

    external fun nativeInit(
        assetManager: AssetManager, modelPath: String,
        inputW: Int, inputH: Int, outputW: Int, outputH: Int
    ): Boolean

    external fun nativeProcessFrame(
        pixels: IntArray, width: Int, height: Int, rotation: Int
    )

    external fun nativeRender(viewWidth: Int, viewHeight: Int)
    external fun nativeDestroy()
    external fun nativeIsInitialized(): Boolean
    external fun nativeIsZeroCopy(): Boolean

    override fun close() = nativeDestroy()

    companion object {
        init {
            System.loadLibrary("depth_pipeline")
        }
    }
}

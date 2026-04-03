package com.depthanything.sample

import android.content.res.AssetManager

class NcnnDepthPipeline : AutoCloseable {
    external fun nativeInit(
        assetManager: AssetManager, paramPath: String, binPath: String,
        targetSize: Int, useGpu: Boolean
    ): Boolean
    external fun nativeInfer(pixels: IntArray, w: Int, h: Int, rotation: Int): IntArray?
    external fun nativeDestroy()
    external fun nativeIsVulkan(): Boolean

    override fun close() = nativeDestroy()

    companion object {
        init { System.loadLibrary("depth_pipeline") }
    }
}

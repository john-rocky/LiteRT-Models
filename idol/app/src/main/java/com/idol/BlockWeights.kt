package com.idol

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

/**
 * The weights of the blocks one shared graph runs (the denoiser's middle joint blocks in `den_joint_mid`, the Gaussian
 * decoder's middle graphs in `gs_mid`): a little-endian float32 side file, block after block, each block's weight
 * inputs in input-name order (from args_<first_weight_input> on). The graph is compiled once per photo instead of once
 * per block; before each call the block's weights are copied from the mapped file into the graph's input buffers.
 */
class BlockWeights(file: File, private val spec: GraphSpec) : AutoCloseable {
    private val raf = RandomAccessFile(file, "r")
    private val floats: FloatBuffer

    /** For each weight input: its index in the signature's input list, its offset in one block (floats), its size. */
    private val slots: List<Triple<Int, Int, Int>>
    private val perBlock: Int

    /** One array per distinct input size, reused slot after slot (a write copies it into the input buffer). */
    private val scratch: Map<Int, FloatArray>

    init {
        try {
            val sig = spec.signature()
            val indexOf = sig.inputs.withIndex().associate { it.value.name to it.index }
            val names = sig.inputs.map { it.name }.filter { trailing(it) >= spec.firstWeightInput }
                .sortedBy { trailing(it) }
            var offset = 0
            slots = names.map { n ->
                val i = indexOf.getValue(n)
                val size = sig.inputs[i].size
                Triple(i, offset, size).also { offset += size }
            }
            perBlock = offset
            val expect = perBlock.toLong() * spec.sharedBlocks.size * 4
            require(file.length() == expect) { "${file.name}: ${file.length()} bytes, expected $expect" }
            floats = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            scratch = slots.map { it.third }.distinct().associateWith { FloatArray(it) }
        } catch (e: Throwable) {
            raf.close()
            throw e
        }
    }

    /** Writes block [block]'s weights into [g]'s inputs (tagged: a block already in the buffers is not copied again). */
    fun write(g: Graph, block: Int) {
        val pos = spec.sharedBlocks.indexOf(block)
        require(pos >= 0) { "block $block is not one of ${spec.name}'s shared blocks" }
        val base = pos.toLong() * perBlock
        for ((k, slot) in slots.withIndex()) {
            val (index, offset, size) = slot
            val a = scratch.getValue(size)
            floats.position((base + offset).toInt())
            floats.get(a, 0, size)
            g.write(index, a, "w${block}_$k")
        }
    }

    override fun close() {
        raf.close()
    }

    private fun trailing(name: String): Int = name.substringAfterLast('_').toIntOrNull() ?: -1
}

package com.audio8tts

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** codes.npy as "Record my voice" writes it reads back through the loader the bundled voices use. */
class NpyTest {
    @get:Rule val tmp = TemporaryFolder()

    @Test
    fun u2RoundTrip() {
        val shape = intArrayOf(10, 7)
        val data = IntArray(70) { (it * 613) % 4096 }.also { it[69] = 65535 }
        val f = File(tmp.root, "codes.npy")
        Npy.saveU2(f, shape, data)
        val back = Npy.loadInts(f)
        assertArrayEquals(shape, back.shape)
        assertArrayEquals(data, back.data)
        // Version 1.0 layout: the data starts at a multiple of 64 bytes.
        assertEquals(0L, (f.length() - 2 * data.size) % 64)
    }

    @Test(expected = IllegalArgumentException::class)
    fun aValueOutsideU2IsRefused() {
        Npy.saveU2(File(tmp.root, "bad.npy"), intArrayOf(1), intArrayOf(70_000))
    }
}

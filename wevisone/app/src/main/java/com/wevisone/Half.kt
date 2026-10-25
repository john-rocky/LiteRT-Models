package com.wevisone

/** IEEE 754 binary16 to float32 (exact: every fp16 value is a float32 value). */
object Half {
    fun toFloat(bits: Int): Float {
        val h = bits and 0xFFFF
        val sign = (h ushr 15) and 1
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x3FF
        val magnitude = when (exp) {
            0 -> mant * SUBNORMAL_UNIT                      // zero and subnormals: mant * 2^-24, exact in float32
            0x1F -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
            else -> java.lang.Float.intBitsToFloat(((exp - 15 + 127) shl 23) or (mant shl 13))
        }
        return if (sign == 1) -magnitude else magnitude
    }

    private val SUBNORMAL_UNIT = java.lang.Float.intBitsToFloat((127 - 24) shl 23)   // 2^-24
}

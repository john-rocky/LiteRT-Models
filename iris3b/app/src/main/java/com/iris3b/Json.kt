package com.iris3b

/**
 * A small JSON reader (objects -> Map, arrays -> List, numbers -> Double, strings, booleans, null).
 * Pure Kotlin so the rows / config parsing is the same code in the app and in the JVM check.
 */
object Json {

    fun parse(text: String): Any? {
        val reader = Reader(text)
        val value = reader.value()
        reader.skipSpace()
        require(reader.pos == text.length) { "trailing text at ${reader.pos}" }
        return value
    }

    private class Reader(val s: String) {
        var pos = 0

        fun skipSpace() {
            while (pos < s.length && s[pos] in " \t\r\n") pos++
        }

        fun value(): Any? {
            skipSpace()
            require(pos < s.length) { "unexpected end" }
            return when (val c = s[pos]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> word("true", true)
                'f' -> word("false", false)
                'n' -> word("null", null)
                else -> if (c == '-' || c in '0'..'9') num() else throw IllegalArgumentException("bad char '$c' at $pos")
            }
        }

        private fun word(w: String, v: Any?): Any? {
            require(s.startsWith(w, pos)) { "bad literal at $pos" }
            pos += w.length
            return v
        }

        private fun obj(): Map<String, Any?> {
            val out = LinkedHashMap<String, Any?>()
            pos++
            skipSpace()
            if (s[pos] == '}') {
                pos++
                return out
            }
            while (true) {
                skipSpace()
                val key = str()
                skipSpace()
                require(s[pos] == ':') { "expected ':' at $pos" }
                pos++
                out[key] = value()
                skipSpace()
                when (s[pos]) {
                    ',' -> pos++
                    '}' -> {
                        pos++
                        return out
                    }
                    else -> throw IllegalArgumentException("expected ',' or '}' at $pos")
                }
            }
        }

        private fun arr(): List<Any?> {
            val out = ArrayList<Any?>()
            pos++
            skipSpace()
            if (s[pos] == ']') {
                pos++
                return out
            }
            while (true) {
                out.add(value())
                skipSpace()
                when (s[pos]) {
                    ',' -> pos++
                    ']' -> {
                        pos++
                        return out
                    }
                    else -> throw IllegalArgumentException("expected ',' or ']' at $pos")
                }
            }
        }

        private fun str(): String {
            require(s[pos] == '"') { "expected a string at $pos" }
            pos++
            val sb = StringBuilder()
            while (true) {
                val c = s[pos++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        when (val e = s[pos++]) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                sb.append(s.substring(pos, pos + 4).toInt(16).toChar())
                                pos += 4
                            }
                            else -> throw IllegalArgumentException("bad escape '\\$e' at $pos")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun num(): Double {
            val start = pos
            while (pos < s.length && s[pos] in "+-0123456789.eE") pos++
            return s.substring(start, pos).toDouble()
        }
    }

    /** JSON text of a value made of maps, lists, strings, numbers, booleans and nulls. */
    fun write(value: Any?): String = StringBuilder().also { emit(value, it) }.toString()

    private fun emit(value: Any?, sb: StringBuilder) {
        when (value) {
            null -> sb.append("null")
            is String -> quote(value, sb)
            is Boolean -> sb.append(value)
            is Int, is Long -> sb.append(value)
            is Float -> if (value.isFinite()) sb.append(value.toDouble()) else sb.append("null")
            is Double -> if (value.isFinite()) sb.append(value) else sb.append("null")
            is Map<*, *> -> {
                sb.append('{')
                var first = true
                for ((k, v) in value) {
                    if (!first) sb.append(',')
                    first = false
                    quote(k.toString(), sb)
                    sb.append(':')
                    emit(v, sb)
                }
                sb.append('}')
            }
            is Iterable<*> -> {
                sb.append('[')
                var first = true
                for (v in value) {
                    if (!first) sb.append(',')
                    first = false
                    emit(v, sb)
                }
                sb.append(']')
            }
            is IntArray -> emit(value.toList(), sb)
            is DoubleArray -> emit(value.toList(), sb)
            else -> quote(value.toString(), sb)
        }
    }

    private fun quote(s: String, sb: StringBuilder) {
        sb.append('"')
        for (c in s) {
            when {
                c == '"' -> sb.append("\\\"")
                c == '\\' -> sb.append("\\\\")
                c == '\n' -> sb.append("\\n")
                c == '\r' -> sb.append("\\r")
                c == '\t' -> sb.append("\\t")
                c < ' ' -> sb.append(String.format("\\u%04x", c.code))
                else -> sb.append(c)
            }
        }
        sb.append('"')
    }
}

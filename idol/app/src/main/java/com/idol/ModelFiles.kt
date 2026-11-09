package com.idol

import android.content.Context
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * The files `scripts/install_to_device.sh` puts in the app's `files/models/`: `manifest.json`, every graph it lists
 * (sizes and SHA-256 from the manifest), the side files (float32, size from their shapes) and the JSON files. The
 * app checks them before it builds a graph.
 */
object ModelFiles {
    /** The two matting graphs; a run needs only the one the setting picks. */
    val MATTING_GRAPHS = setOf("bir", "dis")

    fun dir(filesDir: File): File = File(filesDir, "models")

    /** A file problem the app reports instead of building a graph. */
    class Problem(val file: String, val message: String)

    /** The graphs a run with [matting] needs, in the manifest's order. */
    fun neededGraphs(manifest: Manifest, matting: String): List<String> =
        manifest.graphs.keys.filter { it == matting || it !in MATTING_GRAPHS }

    /** Reads the manifest, or returns null when it is missing or unreadable (reported by [check]). */
    fun manifest(dir: File): Manifest? = runCatching { Manifest.read(File(dir, Manifest.FILE)) }.getOrNull()

    /**
     * Missing files and wrong sizes (cheap); with [hash] also the SHA-256 of every graph the manifest gives one for,
     * once per install: a match is remembered by size and modification time in the app's preferences.
     * [progress] gets (files checked, files to check) while hashing.
     */
    fun check(
        context: Context,
        dir: File,
        matting: String,
        hash: Boolean,
        progress: (Int, Int) -> Unit = { _, _ -> },
    ): List<Problem> {
        val manifestFile = File(dir, Manifest.FILE)
        if (!manifestFile.isFile) return listOf(Problem(Manifest.FILE, "${Manifest.FILE} (missing)"))
        val manifest = manifest(dir) ?: return listOf(Problem(Manifest.FILE, "${Manifest.FILE} (unreadable)"))
        val out = ArrayList<Problem>()
        if (matting !in manifest.graphs) out.add(Problem(matting, "the manifest has no matting graph \"$matting\""))
        val graphs = neededGraphs(manifest, matting).map { manifest.graph(it) }
        for (g in graphs) sizeProblem(File(dir, g.file), g.bytes)?.let { out.add(it) }
        for (s in manifest.sideFiles.values) sizeProblem(File(dir, s.name), s.bytes)?.let { out.add(it) }
        for (j in manifest.jsonFiles) {
            val f = File(dir, j)
            if (!f.isFile || f.length() == 0L) out.add(Problem(j, "$j (missing)"))
        }
        if (!hash || out.isNotEmpty()) return out
        val prefs = context.getSharedPreferences("verified", Context.MODE_PRIVATE)
        val toHash = graphs.filter { it.sha256 != null }
        for ((k, g) in toHash.withIndex()) {
            progress(k, toHash.size)
            val f = File(dir, g.file)
            val key = "${f.length()}:${f.lastModified()}:${g.sha256}"
            if (prefs.getString(g.file, null) == key) continue
            val digest = sha256(f)
            if (digest != g.sha256) {
                out.add(Problem(g.file, "${g.file} (SHA-256 ${digest.take(12)}..., expected ${g.sha256!!.take(12)}...)"))
            } else {
                prefs.edit().putString(g.file, key).apply()
            }
        }
        progress(toHash.size, toHash.size)
        return out
    }

    private fun sizeProblem(f: File, bytes: Long): Problem? = when {
        !f.isFile -> Problem(f.name, "${f.name} (missing)")
        f.length() != bytes -> Problem(f.name, "${f.name} (${f.length()} bytes, expected $bytes)")
        else -> null
    }

    fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /** The message the Home screen shows for [problems]. */
    fun message(problems: List<Problem>): String =
        "Model files are missing. Run scripts/install_to_device.sh (see README).\n" +
            problems.take(12).joinToString("\n") { "• ${it.message}" } +
            if (problems.size > 12) "\n• ... and ${problems.size - 12} more" else ""
}

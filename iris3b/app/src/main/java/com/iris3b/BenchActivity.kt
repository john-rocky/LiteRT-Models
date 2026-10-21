package com.iris3b

import android.app.Activity
import android.app.ActivityManager
import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import com.google.ai.edge.litert.Environment
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream

/**
 * The measurement entry of the iris-3b engine (devicebench's app contract, generic legs):
 *
 *     am start -W -n com.iris3b/.BenchActivity --es graph <entry graph> --es accel <...>
 *         --es precision <...> --es rows <rows.json> --es report <tag>.json --es mode generic
 *
 * Reads files/<rows> and the config it names ([RunFiles]), generates one 768x768 image per row
 * ([IrisPipeline] on [LiteRtExec]) and appends, per row, the image [3, 768, 768] (and the config's
 * dumps) as little-endian float32 to files/out_<report stem>.f32. files/<report> gets the status
 * (DONE / FAILED), out_bytes, and per row the stage timeline and memory readings. files/STOP ends
 * the run before the next graph (the row in progress is dropped). The graph files, every graph's
 * accel and precision, and the residency come from the config, not from the extras (the extras
 * are recorded); the `graph` extra names the entry graph the harness swaps in. The activity stays
 * until force-stopped.
 */
class BenchActivity : Activity() {
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            if (Build.VERSION.SDK_INT >= 27) {
                setShowWhenLocked(true)
                setTurnScreenOn(true)
            }
        }
        status = TextView(this).apply { text = "iris-3b: starting" }
        setContentView(status)
        val extras = mapOf(
            "graph" to intent.getStringExtra("graph").orEmpty(),
            "accel" to (intent.getStringExtra("accel") ?: "gpu"),
            "precision" to intent.getStringExtra("precision").orEmpty(),
            "rows" to intent.getStringExtra("rows").orEmpty(),
            "report" to (intent.getStringExtra("report") ?: "iris3b_report.json"),
            "mode" to (intent.getStringExtra("mode") ?: "generic"),
        )
        Thread({ runAll(extras) }, "Iris3bBench").start()
    }

    private fun show(text: String) {
        Log.i(TAG, text)
        runOnUiThread { status.text = text }
    }

    private val stopFile: File
        get() = File(filesDir, "STOP")

    private fun runAll(extras: Map<String, String>) {
        val report = LinkedHashMap<String, Any?>()
        report.putAll(extras)
        report["rows_file"] = extras["rows"]
        report["litert"] = LITERT_VERSION
        report["device"] = Build.MODEL
        report["android"] = Build.VERSION.SDK_INT
        report["app"] = "com.iris3b engine r4"
        val stem = extras.getValue("report").removeSuffix(".json")
        val outFile = File(filesDir, "out_$stem.f32")
        val rowsOut = ArrayList<Map<String, Any?>>()
        var env: Environment? = null
        var exec: LiteRtExec? = null
        stopFile.delete()
        outFile.delete()
        try {
            require(extras["mode"] == "generic") { "mode ${extras["mode"]}: this app runs generic legs only" }
            require(extras.getValue("rows").isNotEmpty()) { "extra rows is required" }
            val files = RunFiles(filesDir, extras.getValue("rows"))
            report["config_file"] = files.configName
            report["config"] = files.config
            val missing = (files.missingInputs() + files.graphFiles().filter { !File(filesDir, it).isFile }).distinct()
            require(missing.isEmpty()) { "missing input files: $missing" }
            val config = files.engineConfig()
            report["residency"] = config.residency.name
            if (config.residency == Residency.P4) {
                report["prefetch_rule"] = mapOf("factor" to config.prefetch.factor, "margin_bytes" to config.prefetch.marginBytes)
            }
            report["graphs"] = files.graphFiles()
            report["tokenizer_sha256"] = files.materializeTokenizer()
            val outputs = files.outputs()
            report["outputs"] = outputs.map { mapOf("name" to it.first, "count" to it.second) }
            report["out_bytes_per_row"] = outputs.sumOf { it.second.toLong() } * 4
            report["memory_before"] = memoryNow()
            val t0 = System.nanoTime()
            val environment = Environment.create(this)
            env = environment
            report["environment_ms"] = (System.nanoTime() - t0) / 1e6
            val cache = if (files.programCache) File(filesDir, "gpu_cache").also { it.mkdirs() } else null
            var cacheRoles = files.programCacheRoles
            report["data_free_before"] = filesDir.usableSpace
            if (cache != null) {
                // The cache also stores the weights: start without it when the planned writes
                // (graph size x CACHE_FACTOR for every graph without a cache file yet) would leave
                // less than CACHE_MARGIN free on the volume.
                val cached = (config.textEncoder + config.adapter + config.dual + config.single + config.pixel)
                    .filter { s -> cacheRoles.any { s.role.startsWith(it) } }
                    .distinctBy { it.file }
                val planned = cached.filter { !LiteRtExec.cacheFile(cache, it).isFile }
                    .sumOf { File(filesDir, it.file).length() * LiteRtExec.CACHE_FACTOR }
                val free = cache.usableSpace
                val on = planned <= free - LiteRtExec.CACHE_MARGIN
                report["program_cache_plan"] = mapOf(
                    "roles" to cacheRoles, "graphs" to cached.map { it.file }, "planned_bytes" to planned,
                    "free_bytes" to free, "margin_bytes" to LiteRtExec.CACHE_MARGIN, "on" to on,
                )
                if (!on) cacheRoles = emptyList()
            }
            report["program_cache_dir"] = cache?.name
            report["program_cache_files_before"] = cache?.let { listing(it) }
            exec = LiteRtExec(environment, filesDir, cache?.takeIf { cacheRoles.isNotEmpty() }, cacheRoles)
            PromptTokens(File(filesDir, "tok")).use { tokens ->
                val pipeline = IrisPipeline(exec!!, config, files.host, tokens, availableBytes = ::memAvailableBytes) {
                    stopFile.exists()
                }
                report["prefetch_errors"] = pipeline.prefetchErrors
                BufferedOutputStream(FileOutputStream(outFile), 1 shl 20).use { out ->
                    for ((i, row) in files.rows.withIndex()) {
                        if (stopFile.exists()) {
                            report["stopped_early"] = true
                            break
                        }
                        show("row ${i + 1} / ${files.rows.size}: ${row.key}")
                        val log = StageLog(System::nanoTime, ::memoryNow) { e ->
                            Log.i(TAG, "EVT ${row.key} ${e.stage} ${e.graph ?: "-"} s${e.step} ${e.ms} ${e.memory}")
                        }
                        val wall0 = System.currentTimeMillis()
                        val start = System.nanoTime()
                        val result = try {
                            pipeline.generate(row.prompt, files.host[row.noise], log)
                        } catch (stop: StopRequested) {
                            report["stopped_early"] = true
                            rowsOut.add(rowRecord(row, i, log, null, (System.nanoTime() - start) / 1e6, wall0))
                            break
                        }
                        val wallMs = (System.nanoTime() - start) / 1e6
                        out.write(HostOps.littleEndian(result.image))
                        for ((name, _) in outputs.drop(1)) {
                            out.write(HostOps.littleEndian(requireNotNull(result.dumps[name]) { "no dump $name" }))
                        }
                        out.flush()
                        rowsOut.add(rowRecord(row, i, log, result, wallMs, wall0))
                        show("row ${i + 1} done: ${"%.1f".format(wallMs / 1000)} s")
                    }
                }
            }
            report["program_cache_files_after"] = cache?.let { listing(it) }
            report["xnnpack_cache_files_after"] =
                listing(filesDir).filter { (it["name"] as String).endsWith(LiteRtExec.XNNPACK_CACHE_SUFFIX) }
            report["program_cache_decisions"] = exec?.cacheDecisions
            report["data_free_after"] = filesDir.usableSpace
            report["memory_at_end"] = memoryNow()
            report["out_file"] = outFile.name
            report["out_bytes"] = outFile.length()
            report["status"] = "DONE"
        } catch (failure: Throwable) {
            Log.e(TAG, "IRIS3B failed", failure)
            report["status"] = "FAILED"
            report["error"] = failure.toString()
            report["stack"] = failure.stackTrace.take(12).map { it.toString() }
            show("failed: $failure")
        } finally {
            runCatching { env?.close() }
        }
        report["rows"] = rowsOut
        report["count"] = rowsOut.count { it["done"] == true }
        val tmp = File(filesDir, extras.getValue("report") + ".partial")
        tmp.writeText(Json.write(report))
        tmp.renameTo(File(filesDir, extras.getValue("report")))
        Log.i(TAG, "IRIS3B report ${extras["report"]} ${report["status"]}")
        if (report["status"] == "DONE") show("done: ${extras["report"]}")
    }

    /** One row of the report: the totals per stage and the whole timeline. */
    private fun rowRecord(row: RunFiles.Row, index: Int, log: StageLog, result: PipelineResult?, wallMs: Double, wall0: Long): Map<String, Any?> {
        fun sum(stage: String, prefix: String? = null, step: Int? = null, field: String = "ms") =
            log.events.filter { e ->
                e.stage == stage && (prefix == null || (e.graph ?: "").startsWith(prefix)) && (step == null || e.step == step)
            }.sumOf { it.ms[field] ?: 0.0 }
        val steps = (0 until 4).map { s ->
            mapOf(
                "load_ms" to sum("load", "dit_", s),
                "compile_ms" to sum("load", "dit_", s, "compile"),
                "prefetched" to sum("load", "dit_", s, "prefetched"),
                "run_ms" to sum("run", "dit_", s),
                "read_ms" to sum("run", "dit_", s, "read"),
                "close_ms" to sum("close", "dit_", s),
                "host_ms" to sum("mods", step = s) + sum("concat", step = s) + sum("slice", step = s),
                "sampler_ms" to sum("sampler", step = s),
            )
        }
        return linkedMapOf(
            "key" to row.key,
            "index" to index,
            "prompt" to row.prompt,
            "noise" to row.noise,
            "done" to (result != null),
            "t_start_ms" to wall0,
            "wall_ms" to wallMs,
            "real_len" to result?.realLen,
            "caption_tokens" to result?.captionTokens,
            "nonfinite" to result?.let { HostOps.nonFinite(it.image) },
            "stage_ms" to linkedMapOf(
                "tokenize" to sum("tokenize"),
                "embed" to sum("embed"),
                "te_mask" to sum("te_mask"),
                "te_load" to sum("load", "te"),
                "te_run" to sum("run", "te"),
                "te_read" to sum("run", "te", field = "read"),
                "te_close" to sum("close", "te"),
                "te_pack" to sum("te_pack"),
                "adapter_bias" to sum("adapter_bias"),
                "adapter_load" to sum("load", "dit_adapter"),
                "adapter_run" to sum("run", "dit_adapter"),
                "dit_load" to sum("load", "dit_") - sum("load", "dit_adapter"),
                "dit_run" to sum("run", "dit_") - sum("run", "dit_adapter"),
                "dit_close" to sum("close", "dit_"),
                "clamp" to sum("clamp"),
                "assemble" to sum("assemble"),
            ),
            // load above = the pipeline thread's time per load (under P4 the wait for a compile
            // ahead); compile = the compiles themselves; prefetched = how many came from ahead
            "compile_ms" to linkedMapOf(
                "te" to sum("load", "te", field = "compile"),
                "adapter" to sum("load", "dit_adapter", field = "compile"),
                "dit" to sum("load", "dit_", field = "compile") - sum("load", "dit_adapter", field = "compile"),
            ),
            "prefetched" to linkedMapOf(
                "te" to sum("load", "te", field = "prefetched"),
                "adapter" to sum("load", "dit_adapter", field = "prefetched"),
                "dit" to sum("load", "dit_", field = "prefetched") - sum("load", "dit_adapter", field = "prefetched"),
            ),
            "steps" to steps,
            "memory_peak" to linkedMapOf(
                "VmHWM_kb" to log.events.maxOfOrNull { it.memory["VmHWM_kb"] ?: -1L },
                "VmRSS_kb_max" to log.events.maxOfOrNull { it.memory["VmRSS_kb"] ?: -1L },
                "kgsl_page_alloc_max" to log.events.maxOfOrNull { it.memory["kgsl_page_alloc"] ?: -1L },
            ),
            "events" to log.events.map { e ->
                linkedMapOf("stage" to e.stage, "graph" to e.graph, "step" to e.step, "ms" to e.ms, "mem" to e.memory)
            },
        )
    }

    /**
     * VmHWM / VmRSS / VmSwap of this process and the system's MemAvailable in kB, and the GPU's
     * page_alloc (bytes) when readable.
     */
    private fun memoryNow(): Map<String, Long> {
        val out = LinkedHashMap<String, Long>()
        runCatching {
            for (line in File("/proc/self/status").readLines()) {
                val parts = line.split(Regex("\\s+"))
                if (parts.size >= 2 && parts[0] in listOf("VmHWM:", "VmRSS:", "VmSwap:")) {
                    out[parts[0].removeSuffix(":") + "_kb"] = parts[1].toLong()
                }
            }
        }
        memInfoAvailableKb()?.let { out["MemAvailable_kb"] = it }
        runCatching { out["kgsl_page_alloc"] = File("/sys/class/kgsl/kgsl/page_alloc").readText().trim().toLong() }
        return out
    }

    /** MemAvailable from /proc/meminfo in kB, or null when it cannot be read. */
    private fun memInfoAvailableKb(): Long? = runCatching {
        File("/proc/meminfo").useLines { lines ->
            lines.first { it.startsWith("MemAvailable:") }.split(Regex("\\s+"))[1].toLong()
        }
    }.getOrNull()

    /** The system's available memory for P4's rule: /proc/meminfo's MemAvailable, else ActivityManager's. */
    private fun memAvailableBytes(): Long {
        memInfoAvailableKb()?.let { return it * 1024 }
        val info = ActivityManager.MemoryInfo()
        (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
        return info.availMem
    }

    private fun listing(dir: File): List<Map<String, Any?>> =
        (dir.listFiles() ?: emptyArray()).sortedBy { it.name }.map { mapOf("name" to it.name, "bytes" to it.length()) }

    companion object {
        private const val TAG = "IRIS3B"
        private const val LITERT_VERSION = "2.2.0"
    }
}

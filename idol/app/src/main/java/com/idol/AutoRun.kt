package com.idol

import android.os.Build
import android.os.Process
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.CompiledModel
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Debug builds only: runs a plan of rows through the app and writes what each produced, for the device lines of
 * CONTRACT.md ((a) parity, (b) the .ply, (c) speed, (d) failure paths, (e) memory).
 *
 *   adb shell am start -n com.idol/.MainActivity --es autorun plans/<plan>.json --es out runs/<name>
 *
 * Paths are under the app's files dir. Plan: {"preload": true, "reserve_mb": 2500, "mem_check": true,
 * "handoff": "host" | "direct", "matting": "bir" | "dis", "cpu_threads": 6, "gpu_priority": "default", "rows": [...]}
 * (every key optional; a row may override "preload"). Rows:
 *   {"id", "kind": "e2e", "fixture": "fx/<tag>", "runs": 4, "preload": true}: the app's whole pipeline from the
 *       fixture's raw image and mode with its noise arrays; <id>.<run>.ply + <id>.<run>.json per run (run 0 is the
 *       warm-up), <id>.alpha.u8 once (raw uint8 matting alpha), <id>.json for the row.
 *   {"id", "kind": "parity", "fixture": "fx/<tag>"}: each part on the fixture's reference input, <id>.<name>.f32.
 *   {"id", "kind": "image", "image": "<path>", "mode": "real"}: a normal run of a picture (failure paths);
 *       <id>.json with the outcome and the message on the screen.
 *   {"id", "kind": "snippet", "fixture": "fx/<tag>"}: the model card's Kotlin block (CardSnippet.kt) on the
 *       fixture's picture; its line goes to <id>.json, DONE.json and logcat.
 * Output: <out>/header.json (device, options, the load and compile seconds of each preload), the row files,
 * <out>/DONE.json; logcat tag IdolAutoRun: START / ROW / DONE lines. A file <out>/STOP ends the plan after the
 * current run.
 */
class AutoRun(private val activity: MainActivity, private val planFile: File, private val outDir: File) {
    private val header = JSONObject()
    private val loads = JSONArray()

    fun start() {
        Thread({
            try {
                runAll()
            } catch (e: Throwable) {
                Log.e(TAG, "IDOL_AUTORUN FAILED", e)
                runCatching {
                    outDir.mkdirs()
                    File(outDir, "DONE.json").writeText(
                        JSONObject().put("status", "FAILED").put("error", e.toString()).toString(1),
                    )
                }
                Log.i(TAG, "IDOL_AUTORUN DONE status=FAILED pid=${Process.myPid()}")
            } finally {
                activity.autorunFinished()
            }
        }, "IdolAutoRun").start()
    }

    private fun stopped() = File(outDir, "STOP").exists()

    private fun writeHeader() = File(outDir, "header.json").writeText(header.toString(1))

    private fun runAll() {
        val plan = JSONObject(planFile.readText())
        outDir.mkdirs()
        File(outDir, "DONE.json").delete()
        header.put("plan", planFile.name).put("plan_json", plan)
            .put("device", DeviceInfo.displayName()).put("model", Build.MODEL).put("android_sdk", Build.VERSION.SDK_INT)
            .put("fingerprint", Build.FINGERPRINT).put("litert", BuildConfig.LITERT_VERSION)
            .put("app_version", BuildConfig.VERSION_NAME).put("pid", Process.myPid())
            .put("mem_total_kb", DeviceInfo.memTotalKb()).put("mem_available_kb", DeviceInfo.memAvailableKb())
            .put("thermal", DeviceInfo.thermalStatus(activity)).put("cpuset", DeviceInfo.cpuset())
            .put("start_ms", System.currentTimeMillis()).put("load", loads)
        writeHeader()
        Log.i(TAG, "IDOL_AUTORUN START plan=${planFile.name} out=${outDir.name} pid=${Process.myPid()}")
        val rows = plan.getJSONArray("rows")
        val summary = JSONArray()
        val done = JSONObject().put("status", "DONE")
        for (r in 0 until rows.length()) {
            if (stopped()) {
                done.put("stopped", true)
                break
            }
            val row = rows.getJSONObject(r)
            val id = row.getString("id")
            val kind = row.optString("kind", "e2e")
            Log.i(TAG, "IDOL_AUTORUN ROW_START id=$id kind=$kind")
            val rec = try {
                when (kind) {
                    "e2e" -> e2e(plan, row)
                    "parity" -> parity(plan, row)
                    "image" -> image(plan, row)
                    "snippet" -> snippet(row)
                    else -> JSONObject().put("id", id).put("kind", kind).put("status", "unknown_kind")
                }
            } catch (e: Throwable) {
                Log.e(TAG, "row $id failed", e)
                JSONObject().put("id", id).put("kind", kind).put("status", "failed").put("error", e.toString())
            }
            summary.put(rec)
            if (kind == "snippet") done.put("snippet", rec.optString("line"))
            Log.i(TAG, "IDOL_AUTORUN ROW id=$id kind=$kind status=${rec.optString("status")} ${rec.optString("summary")}")
        }
        done.put("rows", summary).put("pid", Process.myPid()).put("end_ms", System.currentTimeMillis())
            .put("self_status_kb", JSONObject(DeviceInfo.selfStatusKb() as Map<*, *>))
        File(outDir, "DONE.json").writeText(done.toString(1))
        Log.i(TAG, "IDOL_AUTORUN DONE rows=${summary.length()} pid=${Process.myPid()}")
    }

    /** The engine options of [row] in [plan] (the row's keys first). */
    private fun options(plan: JSONObject, row: JSONObject): EngineOptions {
        fun has(k: String) = row.has(k) || plan.has(k)
        fun src(k: String) = if (row.has(k)) row else plan
        val defaults = EngineOptions()
        val manifestMatting = ModelFiles.manifest(ModelFiles.dir(activity.filesDir))?.defaultMatting ?: "bir"
        val priority = when (if (has("gpu_priority")) src("gpu_priority").getString("gpu_priority") else "default") {
            "low" -> CompiledModel.GpuOptions.Priority.LOW
            "normal" -> CompiledModel.GpuOptions.Priority.NORMAL
            "high" -> CompiledModel.GpuOptions.Priority.HIGH
            else -> null
        }
        return EngineOptions(
            preload = if (has("preload")) src("preload").getBoolean("preload") else defaults.preload,
            reserveMb = if (has("reserve_mb")) src("reserve_mb").getLong("reserve_mb") else defaults.reserveMb,
            memoryCheck = if (has("mem_check")) src("mem_check").getBoolean("mem_check") else defaults.memoryCheck,
            directHandoff = has("handoff") && src("handoff").getString("handoff") == "direct",
            matting = if (has("matting")) src("matting").getString("matting") else manifestMatting,
            cpuThreads = if (has("cpu_threads")) src("cpu_threads").getInt("cpu_threads") else defaults.cpuThreads,
            gpuPriority = priority,
            disFloor = if (has("dis_floor")) src("dis_floor").getInt("dis_floor") else defaults.disFloor,
        )
    }

    private fun json(o: EngineOptions) = JSONObject().put("preload", o.preload).put("reserve_mb", o.reserveMb)
        .put("mem_check", o.memoryCheck).put("handoff", if (o.directHandoff) "direct" else "host")
        .put("matting", o.matting).put("cpu_threads", o.cpuThreads)
        .put("gpu_priority", o.gpuPriority?.name ?: "default").put("dis_floor", o.disFloor)

    private fun outcomeFields(rec: JSONObject, outcome: Outcome) {
        when (outcome) {
            is Outcome.Done -> rec.put("status", "done")
            is Outcome.NoObject -> rec.put("status", "no_object").put("reason", outcome.reason)
            is Outcome.Missing -> rec.put("status", "missing_models").put("message", ModelFiles.message(outcome.problems))
            is Outcome.Failed -> rec.put("status", outcome.kind).put("message", outcome.message)
        }
    }

    private fun e2e(plan: JSONObject, row: JSONObject): JSONObject {
        val id = row.getString("id")
        val fx = Fixture(File(activity.filesDir, row.getString("fixture")))
        val runs = row.optInt("runs", 4)
        val opts = options(plan, row)
        val rowRec = JSONObject().put("id", id).put("kind", "e2e").put("fixture", row.getString("fixture"))
            .put("mode", fx.mode).put("runs", runs).put("options", json(opts))
        val problems = activity.ensureEngine(opts)
        if (problems.isNotEmpty()) {
            return rowRec.put("status", "missing_models").put("message", ModelFiles.message(problems))
        }
        if (opts.preload) {
            val load = activity.onModel { requireNotNull(Idol.engine).preload(activity.progress) }
            loads.put(JSONObject().put("row", id).put("record", load))
            writeHeader()
            rowRec.put("load_total_s", load.optDouble("total_s")).put("load_loaded", load.optInt("loaded"))
                .put("load_of", load.optInt("of"))
        }
        val photo = Photo.fromFile(fx.image)
        val noise = fx.noise()
        val statuses = JSONArray()
        val measured = ArrayList<Double>()
        var alphaWritten = false
        for (k in 0 until runs) {
            if (stopped()) {
                rowRec.put("stopped_after", k)
                break
            }
            val ply = File(outDir, "$id.$k.ply")
            val rec = JSONObject().put("id", id).put("run", k).put("warmup", k == 0)
                .put("fixture", row.getString("fixture")).put("mode", fx.mode).put("noise", noise.source)
                .put("options", json(opts))
                .put("thermal_start", DeviceInfo.thermalStatus(activity))
                .put("mem_available_kb_start", DeviceInfo.memAvailableKb())
                .put("vmhwm_reset", DeviceInfo.resetPeakRss())
                .put("cpuset", DeviceInfo.cpuset())
            val sampler = GpuPageSampler().also { it.start() }
            val t0 = SystemClock.elapsedRealtimeNanos()
            val outcome = activity.runAndWait(photo, fx.mode, noise, ply, showResult = false)
            val wall = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9
            val gpuMax = sampler.stop()
            val st = DeviceInfo.selfStatusKb()
            rec.put("wall_s", wall).put("thermal_end", DeviceInfo.thermalStatus(activity))
                .put("vmhwm_kb", st["VmHWM"] ?: -1L).put("vmrss_kb", st["VmRSS"] ?: -1L)
                .put("vmswap_kb", st["VmSwap"] ?: -1L).put("gpu_page_alloc_max_bytes", gpuMax)
                .put("mem_available_kb_end", DeviceInfo.memAvailableKb())
            outcomeFields(rec, outcome)
            if (outcome is Outcome.Done) {
                val r = outcome.result
                val stages = JSONObject()
                for ((key, v) in r.times) if (key != "generate" && key != "all") stages.put(key, v)
                rec.put("stages", stages).put("generate", r.times["generate"]).put("total", r.times["all"])
                    .put("ply", ply.name).put("ply_bytes", ply.length()).put("gaussians", r.gaussians)
                    .put("box", JSONArray(r.pre.box.toList()))
                    .put("alpha_size", JSONArray(listOf(r.pre.alphaWidth, r.pre.alphaHeight)))
                    .put("own_alpha", r.pre.ownAlpha).put("handoff", r.handoff)
                    .put("compiles_in_run", JSONObject().put("count", r.compiles.size)
                        .put("seconds", r.compiles.sumOf { it.seconds })
                        .put("graphs", JSONArray(r.compiles.map { it.json() })))
                    .put("memory_events", JSONArray(r.events))
                if (!alphaWritten) {
                    File(outDir, "$id.alpha.u8").writeBytes(r.pre.alphaRaw)
                    rowRec.put("alpha", JSONObject().put("file", "$id.alpha.u8").put("width", r.pre.alphaWidth)
                        .put("height", r.pre.alphaHeight).put("box", JSONArray(r.pre.box.toList())))
                    alphaWritten = true
                }
                if (k > 0) r.times["generate"]?.let { measured.add(it) }
            }
            File(outDir, "$id.$k.json").writeText(rec.toString(1))
            statuses.put(rec.optString("status"))
            Log.i(TAG, "IDOL_AUTORUN RUN id=$id run=$k warmup=${k == 0} status=${rec.optString("status")} " +
                "generate_s=${rec.optDouble("generate", -1.0)} vmhwm_kb=${rec.optLong("vmhwm_kb")}")
        }
        val allDone = (0 until statuses.length()).all { statuses.getString(it) == "done" }
        rowRec.put("statuses", statuses).put("status", if (allDone && statuses.length() > 0) "done" else "incomplete")
        if (measured.isNotEmpty()) {
            val sorted = measured.sorted()
            val median = if (sorted.size % 2 == 1) sorted[sorted.size / 2]
            else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2
            rowRec.put("generate_measured_s", JSONArray(measured)).put("generate_median_s", median)
                .put("summary", "generate median %.2f s over %d measured runs".format(median, measured.size))
        }
        File(outDir, "$id.json").writeText(rowRec.toString(1))
        return rowRec
    }

    private fun parity(plan: JSONObject, row: JSONObject): JSONObject {
        val id = row.getString("id")
        val fx = Fixture(File(activity.filesDir, row.getString("fixture")))
        val opts = options(plan, row).copy(preload = false)
        val problems = activity.ensureEngine(opts)
        if (problems.isNotEmpty()) {
            return JSONObject().put("id", id).put("kind", "parity").put("status", "missing_models")
                .put("message", ModelFiles.message(problems))
        }
        activity.showRunScreen("Part agreement: ${fx.dir.name}")
        val t0 = SystemClock.elapsedRealtimeNanos()
        val rec = activity.onModel { requireNotNull(Idol.engine).parity(fx, outDir, id, activity.progress) }
        rec.put("kind", "parity").put("status", "done").put("options", json(opts))
            .put("wall_s", (SystemClock.elapsedRealtimeNanos() - t0) / 1e9)
            .put("summary", "${rec.getJSONObject("arrays").length()} arrays")
        File(outDir, "$id.json").writeText(rec.toString(1))
        return rec
    }

    private fun image(plan: JSONObject, row: JSONObject): JSONObject {
        val id = row.getString("id")
        val path = row.getString("image")
        val mode = row.optString("mode", "real")
        val rec = JSONObject().put("id", id).put("kind", "image").put("image", path).put("mode", mode)
        activity.ensureEngine(options(plan, row))
        val photo = try {
            Photo.fromFile(File(activity.filesDir, path))
        } catch (e: Throwable) {
            return rec.put("status", "decode_failed").put("error", e.toString())
        }
        val outcome = activity.runAndWait(
            photo, mode, Noise.random(MainActivity.DEFAULT_SEED), File(outDir, "$id.ply"),
            showResult = row.optBoolean("show_result", false),
        )
        outcomeFields(rec, outcome)
        if (outcome is Outcome.Done) rec.put("generate", outcome.result.times["generate"])
        // the outcome is on the screen once the UI thread has run the posts before this one
        rec.put("message_on_screen", activity.homeMessageText()).put("pid", Process.myPid())
        rec.put("summary", rec.optString("message_on_screen").lineSequence().firstOrNull() ?: "")
        File(outDir, "$id.json").writeText(rec.toString(1))
        return rec
    }

    private fun snippet(row: JSONObject): JSONObject {
        val id = row.getString("id")
        val fx = Fixture(File(activity.filesDir, row.getString("fixture")))
        val photo = Photo.fromFile(fx.image)
        val rgb = Preprocess.resized(photo.image).rgb()
        val image = Matting.birefnetInput(rgb)
        val dir = ModelFiles.dir(activity.filesDir)
        val t0 = SystemClock.elapsedRealtimeNanos()
        val line = activity.onModel {
            // the card's block builds its own environment and graph; free the engine's graphs first
            Idol.engine?.pool?.releaseAll()
            val bir = cardSnippet(dir, image)
            if (!fx.has("pixels") || !fx.has("feature1")) return@onModel bir
            // the DINOv3 card's block on the fixture's preprocessed picture, against the reference's feature1
            val got = dinoCardSnippet(dir, fx.floats("pixels"))
            val want = fx.floats("feature1")
            var dot = 0.0
            var ng = 0.0
            var nw = 0.0
            for (i in want.indices) {
                dot += got[i].toDouble() * want[i]
                ng += got[i].toDouble() * got[i]
                nw += want[i].toDouble() * want[i]
            }
            "$bir | dino card snippet: ${got.size} values, cos vs the reference feature1 %.6f"
                .format(dot / Math.sqrt(ng * nw))
        }
        Log.i(TAG, "IDOL_AUTORUN SNIPPET $line")
        val rec = JSONObject().put("id", id).put("kind", "snippet").put("status", "done").put("line", line)
            .put("picture", JSONArray(listOf(rgb.width, rgb.height)))
            .put("seconds", (SystemClock.elapsedRealtimeNanos() - t0) / 1e9).put("summary", line)
        File(outDir, "$id.json").writeText(rec.toString(1))
        return rec
    }

    companion object {
        const val TAG = "IdolAutoRun"
    }
}

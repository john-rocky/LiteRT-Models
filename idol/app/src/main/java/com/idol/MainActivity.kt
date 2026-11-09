package com.idol

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.ComponentCallbacks2
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.OpenableColumns
import android.util.Log
import android.util.TypedValue
import android.view.View
import android.view.WindowManager
import android.webkit.WebView
import android.widget.Button
import android.widget.CheckBox
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioGroup
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** How a run ended, for the screen and the autorun. */
sealed class Outcome {
    class Done(val result: RunResult) : Outcome()
    class NoObject(val reason: String) : Outcome()
    class Missing(val problems: List<ModelFiles.Problem>) : Outcome()
    class Failed(val kind: String, val message: String) : Outcome()
}

/** The model state of the process: one model thread and one engine, kept across activity instances. */
object Idol {
    /** Every LiteRT call (compile, run, close) happens on this one thread. */
    val model: ExecutorService = Executors.newSingleThreadExecutor { r -> Thread(r, "IdolModel") }
    @Volatile var engine: IdolEngine? = null
    @Volatile var problems: List<ModelFiles.Problem>? = null
    @Volatile var started = false
}

/**
 * Home (pick or take a photo) -> Run (the stages with their seconds) -> Result (the splat viewer). Every run, from a
 * button or from the debug autorun ([AutoRun]), goes through [startRun]; the work is on the model thread, never on
 * the UI thread.
 */
class MainActivity : ComponentActivity() {
    private lateinit var home: View
    private lateinit var runScreen: View
    private lateinit var resultScreen: View
    private lateinit var modeGroup: RadioGroup
    private lateinit var disCheck: CheckBox
    private lateinit var homeMessage: TextView
    private lateinit var homeStatus: TextView
    private lateinit var picture: ImageView
    private lateinit var cutout: ImageView
    private lateinit var stageList: LinearLayout
    private lateinit var totalText: TextView
    private lateinit var runStatus: TextView
    private lateinit var viewerBox: FrameLayout
    private var web: WebView? = null

    private val ui = Handler(Looper.getMainLooper())
    private val decoder = Executors.newSingleThreadExecutor { r -> Thread(r, "IdolDecode") }
    private val stageRows = ArrayList<TextView>()

    private class StageState {
        var detail: String? = null
        var seconds: Double? = null
        var startedAt = 0L
        var running = false
        var done = false
    }

    private val stages = Array(Stage.entries.size) { StageState() }
    @Volatile private var running = false
    @Volatile var autorunActive = false
        private set
    private var lastPly: File? = null
    private var deviceLine = ""

    /** Engine progress, posted to the screen. */
    val progress = object : Progress {
        override fun stage(stage: Stage, detail: String?, seconds: Double?, done: Boolean) {
            ui.post { onStage(stage, detail, seconds, done) }
        }

        override fun status(text: String) {
            ui.post {
                runStatus.text = text
                homeStatus.text = if (text.isEmpty()) deviceLine else "$text\n$deviceLine"
            }
        }

        override fun cutout(composite: Image8) {
            val bmp = Photo.displayBitmap(composite, 512)
            ui.post {
                cutout.setImageBitmap(bmp)
                cutout.visibility = View.VISIBLE
            }
        }
    }

    private val pickPhoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) startFromPicture { Photo.fromUri(contentResolver, uri, displayName(uri)) }
    }

    private val takePhoto = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        // the camera app can take enough memory for Android to stop this process while it is open; the activity is
        // then created again for the result, so the photo's path is a fixed one, not a field
        val file = cameraFile()
        if (ok && file.isFile) startFromPicture { Photo.fromFile(file) }
    }

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) launchCamera() else showHome(getString(R.string.camera_denied))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (isDebuggable()) {
            // the autorun drives a phone nobody unlocked: stay in front (cpuset top-app) with the screen on
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            if (Build.VERSION.SDK_INT >= 27) {
                setShowWhenLocked(true)
                setTurnScreenOn(true)
            }
        }
        setContentView(R.layout.activity_main)
        // targetSdk 35 draws edge to edge: keep the screens clear of the status and navigation bars
        val root = findViewById<View>(R.id.root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        home = findViewById(R.id.home)
        runScreen = findViewById(R.id.run)
        resultScreen = findViewById(R.id.result)
        modeGroup = findViewById(R.id.mode)
        disCheck = findViewById(R.id.dis)
        homeMessage = findViewById(R.id.home_message)
        homeStatus = findViewById(R.id.home_status)
        picture = findViewById(R.id.picture)
        cutout = findViewById(R.id.cutout)
        stageList = findViewById(R.id.stages)
        totalText = findViewById(R.id.total)
        runStatus = findViewById(R.id.run_status)
        viewerBox = findViewById(R.id.viewer)

        for (s in Stage.entries) {
            val tv = TextView(this)
            tv.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16f)
            tv.setPadding(0, dp(6), 0, dp(6))
            stageList.addView(tv)
            stageRows.add(tv)
        }

        findViewById<Button>(R.id.pick).setOnClickListener {
            pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<Button>(R.id.camera).setOnClickListener {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                launchCamera()
            } else {
                cameraPermission.launch(Manifest.permission.CAMERA)
            }
        }
        findViewById<Button>(R.id.share).setOnClickListener { share() }
        findViewById<Button>(R.id.new_run).setOnClickListener { showHome(null) }

        val prefs = getSharedPreferences("settings", MODE_PRIVATE)
        disCheck.isChecked = prefs.getBoolean("dis", false)
        disCheck.setOnCheckedChangeListener { _, checked ->
            prefs.edit().putBoolean("dis", checked).apply()
            if (!running && !autorunActive) Idol.model.execute { setup(uiOptions(), preload = true) }
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = onBack()
        })

        deviceLine = "${DeviceInfo.displayName()} · LiteRT ${BuildConfig.LITERT_VERSION}"
        homeStatus.text = deviceLine
        showHome(null)
        val autorun = isDebuggable() && intent?.getStringExtra("autorun") != null
        if (!Idol.started) {
            Idol.started = true
            // an autorun builds its own engine with its plan's options: no preload here then
            val opts = uiOptions()
            Idol.model.execute { setup(opts, preload = !autorun) }
        } else {
            Idol.problems?.takeIf { it.isNotEmpty() }?.let { showMissing(it) }
        }
        handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val plan = intent?.getStringExtra("autorun") ?: return
        if (!isDebuggable() || autorunActive) return
        autorunActive = true
        AutoRun(this, File(filesDir, plan), File(filesDir, intent.getStringExtra("out") ?: "runs/autorun")).start()
    }

    fun autorunFinished() {
        ui.post { autorunActive = false }
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        // In the background (the camera app in front, for one) the resident graphs make this process the first one
        // Android stops: give their memory back while nothing runs; the next run compiles them again.
        if (level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND && !running && !autorunActive) {
            Idol.model.execute { if (!running) Idol.engine?.pool?.releaseAll() }
        }
    }

    override fun onDestroy() {
        ResultViewer.destroy(viewerBox, web)
        web = null
        if (isFinishing && !running && !autorunActive) {
            Idol.model.execute {
                Idol.engine?.close()
                Idol.engine = null
                Idol.started = false
            }
        }
        super.onDestroy()
    }

    private fun isDebuggable() = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0

    private fun dp(v: Int) = (v * resources.displayMetrics.density + 0.5f).toInt()

    // ------------------------------------------------------------------------------------------------ model thread
    /** The options of a run from the buttons: preload on, the matting the setting picks (else the manifest's). */
    private fun uiOptions(): EngineOptions {
        val matting = if (getSharedPreferences("settings", MODE_PRIVATE).getBoolean("dis", false)) "dis" else ""
        return EngineOptions(preload = true, matting = matting)
    }

    /**
     * On the model thread: checks the model files (sizes; SHA-256 once per install), then builds the engine with
     * [options] (an empty matting = the manifest's default) and, with [preload], compiles the graphs.
     */
    private fun setup(options: EngineOptions, preload: Boolean) {
        Idol.engine?.close()
        Idol.engine = null
        val dir = ModelFiles.dir(filesDir)
        val manifest = ModelFiles.manifest(dir)
        val matting = options.matting.ifEmpty { manifest?.defaultMatting ?: "bir" }
        val problems = ModelFiles.check(this, dir, matting, hash = true) { k, n ->
            progress.status("Checking model files ${k + 1}/$n")
        }
        Idol.problems = problems
        if (problems.isNotEmpty() || manifest == null) {
            progress.status("")
            ui.post { if (!running) showMissing(problems) }
            return
        }
        ui.post { if (homeMessage.text.startsWith("Model files")) homeMessage.text = "" }
        val engine = try {
            IdolEngine(dir, manifest, options.copy(matting = matting))
        } catch (e: Throwable) {
            Log.e(TAG, "engine", e)
            progress.status("")
            ui.post { showHome("The model files could not be read: ${e.message}") }
            return
        }
        Idol.engine = engine
        if (preload && options.preload) {
            try {
                engine.preload(progress)
            } catch (e: Throwable) {
                Log.e(TAG, "preload", e)
                progress.status("Loading stopped: ${e.message}")
            }
        } else {
            progress.status("")
        }
    }

    /** Runs [block] on the model thread and waits for it (the autorun's engine calls). */
    fun <T> onModel(block: () -> T): T = Idol.model.submit(Callable { block() }).get()

    /** The autorun's engine: rebuilt (without preloading) unless the current one has [options]. */
    fun ensureEngine(options: EngineOptions): List<ModelFiles.Problem> = onModel {
        val e = Idol.engine
        if (e != null && e.options == options) {
            emptyList()
        } else {
            setup(options, preload = false)
            Idol.problems ?: emptyList()
        }
    }

    private fun generate(photo: Photo, mode: String, noise: Noise, ply: File): Outcome {
        val e = Idol.engine ?: return Idol.problems?.takeIf { it.isNotEmpty() }?.let { Outcome.Missing(it) }
            ?: Outcome.Failed("not_ready", "The models are not loaded.")
        return try {
            ply.parentFile?.mkdirs()
            Outcome.Done(e.generate(photo.image, mode, noise, ply, progress))
        } catch (x: NoObjectException) {
            Outcome.NoObject(x.reason)
        } catch (x: NotEnoughMemoryException) {
            Outcome.Failed("memory", x.message ?: "Not enough memory.")
        } catch (x: CancelledException) {
            Outcome.Failed("cancelled", "Stopped.")
        } catch (x: OutOfMemoryError) {
            Log.e(TAG, "run", x)
            Outcome.Failed("memory", "Not enough memory for this picture.")
        } catch (x: Throwable) {
            Log.e(TAG, "run", x)
            Outcome.Failed("error", "Something went wrong: ${x.message ?: x.javaClass.simpleName}")
        } finally {
            progress.status("")
        }
    }

    // ------------------------------------------------------------------------------------------------ runs
    private fun startFromPicture(decode: () -> Photo) {
        if (running) return
        val mode = currentMode()
        showRun(null)
        decoder.execute {
            val photo = try {
                decode()
            } catch (e: Throwable) {
                Log.e(TAG, "decode", e)
                ui.post { showHome("This picture could not be opened: ${e.message ?: e.javaClass.simpleName}") }
                return@execute
            }
            ui.post { startRun(photo, mode, Noise.random(DEFAULT_SEED), newResultFile(), showResult = true) {} }
        }
    }

    /**
     * Starts a run of [photo] on the model thread; [done] gets the outcome on the UI thread. With [showResult] a
     * finished run opens the viewer; without (the autorun) the Run screen stays with its times.
     */
    fun startRun(photo: Photo, mode: String, noise: Noise, ply: File, showResult: Boolean, done: (Outcome) -> Unit) {
        running = true
        showRun(photo)
        Idol.model.execute {
            val outcome = generate(photo, mode, noise, ply)
            ui.post {
                running = false
                when (outcome) {
                    is Outcome.Done -> {
                        val t = outcome.result.times
                        totalText.text = "Total %.1f s (3D generation %.1f s)".format(t["all"] ?: 0.0, t["generate"] ?: 0.0)
                        if (showResult) showResult(outcome.result.ply)
                    }
                    is Outcome.NoObject -> showHome(getString(R.string.no_object))
                    is Outcome.Missing -> showMissing(outcome.problems)
                    is Outcome.Failed -> showHome(if (outcome.kind == "cancelled") null else outcome.message)
                }
                done(outcome)
            }
        }
    }

    /** The autorun's run: [startRun] from another thread, waiting for the outcome. */
    fun runAndWait(photo: Photo, mode: String, noise: Noise, ply: File, showResult: Boolean): Outcome {
        val latch = CountDownLatch(1)
        var result: Outcome? = null
        ui.post {
            if (running) {
                result = Outcome.Failed("busy", "A run is already in progress.")
                latch.countDown()
            } else {
                startRun(photo, mode, noise, ply, showResult) {
                    result = it
                    latch.countDown()
                }
            }
        }
        latch.await()
        return result!!
    }

    /** The Run screen for a run the autorun drives without a picture (the parity row). */
    fun showRunScreen(title: String) {
        ui.post {
            showRun(null)
            runStatus.text = title
        }
    }

    /** The message the Home screen shows now (the autorun records it). */
    fun homeMessageText(): String {
        val latch = CountDownLatch(1)
        var text = ""
        ui.post {
            text = if (home.visibility == View.VISIBLE) homeMessage.text.toString() else ""
            latch.countDown()
        }
        latch.await()
        return text
    }

    private fun currentMode() = if (modeGroup.checkedRadioButtonId == R.id.mode_render) "synthetic" else "real"

    private fun cameraFile() = File(File(cacheDir, "camera"), "photo.jpg")

    private fun launchCamera() {
        val file = cameraFile().apply { parentFile?.mkdirs() }
        try {
            takePhoto.launch(FileProvider.getUriForFile(this, "$packageName.files", file))
        } catch (e: ActivityNotFoundException) {
            showHome(getString(R.string.no_camera_app))
        }
    }

    private fun resultsDir() = File(filesDir, "results").apply { mkdirs() }

    /** A new file in files/results/ (the last few results are kept). */
    private fun newResultFile(): File {
        val dir = resultsDir()
        dir.listFiles { f -> f.name.endsWith(".ply") }?.sortedByDescending { it.lastModified() }?.drop(KEEP_RESULTS - 1)
            ?.forEach { it.delete() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(dir, "idol_$stamp.ply")
    }

    private fun displayName(uri: Uri): String =
        runCatching {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
        }.getOrNull() ?: uri.lastPathSegment ?: "photo"

    private fun share() {
        val ply = lastPly ?: return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", ply)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivity(Intent.createChooser(send, getString(R.string.share)))
        } catch (e: ActivityNotFoundException) {
            runStatus.text = getString(R.string.no_share_app)
        }
    }

    // ------------------------------------------------------------------------------------------------ screens
    private fun onBack() {
        when {
            resultScreen.visibility == View.VISIBLE -> showHome(null)
            runScreen.visibility == View.VISIBLE && running -> {
                Idol.engine?.cancel = true
                runStatus.text = getString(R.string.stopping)
            }
            runScreen.visibility == View.VISIBLE -> showHome(null)
            else -> finish()
        }
    }

    private fun showHome(message: String?) {
        ResultViewer.destroy(viewerBox, web)
        web = null
        stopTicker()
        home.visibility = View.VISIBLE
        runScreen.visibility = View.GONE
        resultScreen.visibility = View.GONE
        homeMessage.text = message ?: ""
        homeMessage.visibility = if (message.isNullOrEmpty()) View.GONE else View.VISIBLE
    }

    private fun showMissing(problems: List<ModelFiles.Problem>) {
        showHome(ModelFiles.message(problems))
    }

    private fun showRun(photo: Photo?) {
        ResultViewer.destroy(viewerBox, web)
        web = null
        home.visibility = View.GONE
        resultScreen.visibility = View.GONE
        runScreen.visibility = View.VISIBLE
        if (photo != null) picture.setImageBitmap(photo.displayBitmap()) else picture.setImageDrawable(null)
        cutout.setImageDrawable(null)
        cutout.visibility = View.INVISIBLE
        for (s in stages) {
            s.detail = null
            s.seconds = null
            s.running = false
            s.done = false
        }
        totalText.text = ""
        runStatus.text = ""
        renderStages()
        startTicker()
    }

    private fun showResult(ply: File) {
        stopTicker()
        lastPly = ply
        home.visibility = View.GONE
        runScreen.visibility = View.GONE
        resultScreen.visibility = View.VISIBLE
        web = ResultViewer.show(this, viewerBox, resultsDir(), ply)
    }

    private fun onStage(stage: Stage, detail: String?, seconds: Double?, done: Boolean) {
        val s = stages[stage.ordinal]
        s.detail = detail
        if (done) {
            s.running = false
            s.done = true
            s.seconds = seconds
        } else if (!s.running) {
            s.running = true
            s.startedAt = SystemClock.elapsedRealtime()
        }
        renderStages()
    }

    private fun renderStages() {
        val now = SystemClock.elapsedRealtime()
        for ((i, st) in Stage.entries.withIndex()) {
            val s = stages[i]
            val label = if (s.detail != null) "${st.label}, ${s.detail}" else st.label
            stageRows[i].text = when {
                s.done -> "✓  $label — %.1f s".format(s.seconds ?: 0.0)
                s.running -> "●  $label — %.1f s".format((now - s.startedAt) / 1000.0)
                else -> "○  $label"
            }
            stageRows[i].alpha = if (s.done || s.running) 1f else 0.5f
        }
    }

    private val ticker = object : Runnable {
        override fun run() {
            renderStages()
            ui.postDelayed(this, 250)
        }
    }

    private fun startTicker() {
        ui.removeCallbacks(ticker)
        ui.postDelayed(ticker, 250)
    }

    private fun stopTicker() {
        ui.removeCallbacks(ticker)
    }

    companion object {
        const val TAG = "Idol"
        /** The seed of a run from a picture (scripts/run_idol.py's default; the app draws with java.util.Random). */
        const val DEFAULT_SEED = 42L
        const val KEEP_RESULTS = 5
    }
}

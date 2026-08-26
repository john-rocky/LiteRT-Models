package com.edgeagent.lab

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.edgeagent.sdk.Agent
import com.edgeagent.sdk.Framing
import com.edgeagent.sdk.Grounded
import com.edgeagent.sdk.LiteRtGrounder
import com.edgeagent.sdk.ScreenAgent
import com.edgeagent.sdk.ScreenSource
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.system.measureTimeMillis

/**
 * Host for the screen agent: UI, permissions, and the two Android services that
 * satisfy the SDK's seams. The work itself lives in `com.edgeagent.sdk`.
 *
 * One capture consent covers a whole session — [CaptureService] holds the
 * projection so the agent can capture, act, and capture again without asking
 * the user anything in between.
 */
class MainActivity : Activity() {

    private companion object {
        const val REQ_PROJECTION = 1
        const val REQ_NOTIFICATIONS = 2
        /** Long enough for this app's window to be gone from the screen. */
        const val SETTLE_MS = 900L
        /** Long enough for the tapped screen to finish its transition. */
        const val AFTER_TAP_MS = 1400L
        const val MAX_STEPS = 6
    }

    private lateinit var status: TextView
    private lateinit var result: TextView
    private lateinit var overlay: OverlayView
    private lateinit var instruction: EditText
    private lateinit var runButton: Button
    private lateinit var loadButton: Button
    private lateinit var accessibilityButton: Button
    private lateinit var badgeButton: Button
    private lateinit var face: AgentFace
    private lateinit var speech: TextView
    private lateinit var robotPanel: LinearLayout
    private lateinit var gpuRadio: RadioButton
    private lateinit var tapRadio: RadioButton
    private lateinit var agentRadio: RadioButton
    private lateinit var actRadio: RadioButton
    private lateinit var askRadio: RadioButton

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var grounder: LiteRtGrounder? = null
    private var agent: ScreenAgent? = null
    private var modelFile: File? = null
    private var busy = false
    private var readyStatus = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val root = findViewById<LinearLayout>(R.id.root_layout)
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime()
            )
            view.setPadding(bars.left + 24, bars.top + 24, bars.right + 24, bars.bottom + 24)
            insets
        }

        status = findViewById(R.id.tv_status)
        result = findViewById(R.id.tv_result)
        overlay = findViewById(R.id.overlay)
        instruction = findViewById(R.id.et_instruction)
        runButton = findViewById(R.id.btn_capture)
        loadButton = findViewById(R.id.btn_load)
        accessibilityButton = findViewById(R.id.btn_accessibility)
        badgeButton = findViewById(R.id.btn_badge)
        face = findViewById(R.id.face)
        speech = findViewById(R.id.tv_speech)
        robotPanel = findViewById(R.id.robot_panel)
        gpuRadio = findViewById(R.id.rb_gpu)
        tapRadio = findViewById(R.id.rb_tap)
        agentRadio = findViewById(R.id.rb_agent)
        actRadio = findViewById(R.id.rb_act)
        askRadio = findViewById(R.id.rb_ask)

        // A previous run's overlay sitting next to a freshly typed instruction
        // reads as if it were the answer to it. Drop it when the question changes.
        instruction.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                if (busy) return
                overlay.clear()
                robotPanel.visibility = android.view.View.VISIBLE
                result.text = ""
                answer(s?.toString().orEmpty())
            }
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) = Unit
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) = Unit
        })

        findViewById<android.widget.RadioGroup>(R.id.rg_mode)
            .setOnCheckedChangeListener { _, _ ->
                if (!busy) answer(instruction.text.toString())
            }
        loadButton.setOnClickListener { loadModel() }
        runButton.setOnClickListener { onRun() }
        accessibilityButton.setOnClickListener { TapService.openSettings(this) }
        badgeButton.setOnClickListener { AgentOverlay.requestPermission(this) }
        runButton.isEnabled = false

        findModel()
        requestNotificationsIfNeeded()
    }

    override fun onResume() {
        super.onResume()
        accessibilityButton.visibility =
            if (TapService.isEnabled(this)) android.view.View.GONE else android.view.View.VISIBLE
        // The badge is optional: without it the agent still runs, it just runs
        // invisibly once the app steps back.
        badgeButton.visibility =
            if (AgentOverlay.isAllowed(this)) android.view.View.GONE else android.view.View.VISIBLE
    }

    override fun onDestroy() {
        CaptureService.onReady = null
        CaptureService.finish(this)
        grounder?.close()
        super.onDestroy()
    }

    // ---------- model ----------

    private fun findModel() {
        val dir = getExternalFilesDir(null)
        val bundles = dir?.listFiles { f -> f.name.endsWith(".litertlm") }?.sortedBy { it.name }
        if (bundles.isNullOrEmpty()) {
            status.text = "No .litertlm in ${dir?.absolutePath}"
            loadButton.isEnabled = false
            return
        }
        modelFile = bundles.first()
        status.text = "${bundles.first().name} " +
            "(${bundles.first().length() / (1024 * 1024)} MB) — press Load"
    }

    private fun loadModel() {
        val model = modelFile ?: return
        val backend = if (gpuRadio.isChecked) "gpu" else "cpu"
        loadButton.isEnabled = false
        status.text = "Loading ${model.name} on $backend…"
        scope.launch {
            var created: LiteRtGrounder? = null
            val millis = measureTimeMillis {
                created = withContext(Dispatchers.IO) {
                    LiteRtGrounder.create(model, backend, cacheDir)
                }
            }
            val ready = created
            if (ready == null) {
                status.text = "Engine failed to load ${model.name} (all backends)."
                loadButton.isEnabled = true
                return@launch
            }
            grounder?.close()
            grounder = ready
            agent = ScreenAgent(
                screen = ScreenSource { AgentOverlay.duringCapture { captureScreen() } },
                executor = ShownExecutor(AccessibilityExecutor(this@MainActivity)),
                grounder = ready,
                cacheDir = cacheDir,
                settleMillis = AFTER_TAP_MS,
            )
            readyStatus = "Ready — ${model.name} on ${ready.effectiveBackend} " +
                "(loaded in ${millis / 1000}s)"
            status.text = readyStatus
            runButton.isEnabled = true
        }
    }

    // ---------- running ----------

    private fun onRun() {
        if (agent == null || busy) return
        if (CaptureService.isRunning) {
            dispatch()
            return
        }
        // First run of the session: ask once, then keep the projection.
        val manager = getSystemService(MediaProjectionManager::class.java)
        startActivityForResult(manager.createScreenCaptureIntent(), REQ_PROJECTION)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_PROJECTION) return
        if (resultCode != RESULT_OK || data == null) {
            status.text = "Screen capture was declined."
            return
        }
        val bounds = getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds
        CaptureService.onReady = { ok ->
            CaptureService.onReady = null
            if (!ok) status.text = "Could not start screen capture." else dispatch()
        }
        CaptureService.start(
            context = this,
            resultCode = resultCode,
            data = data,
            width = bounds.width(),
            height = bounds.height(),
            density = resources.configuration.densityDpi,
        )
    }

    private fun dispatch() = when {
        agentRadio.isChecked -> runAgent(actions = false)
        actRadio.isChecked -> runAgent(actions = true)
        else -> runOnce()
    }

    /** One shot: locate, optionally press, then show the screen it produced. */
    private fun runOnce() {
        val agent = this.agent ?: return
        val grounder = this.grounder ?: return
        val question = instruction.text.toString().trim()
            .ifEmpty { "Point to the search bar." }
        val raw = askRadio.isChecked
        val shouldTap = tapRadio.isChecked && !raw
        startWork("Working…")

        scope.launch {
            moveTaskToBack(true)
            delay(SETTLE_MS)
            AgentOverlay.show(this@MainActivity, question)
            AgentOverlay.update(AgentFace.State.THINKING, "reading the screen")

            var sighting: ScreenAgent.Sighting? = null
            val millis = measureTimeMillis {
                sighting = if (raw) {
                    // Ask mode bypasses the grounding prompt entirely.
                    val frame = captureScreen()
                    if (frame == null) null else withContext(Dispatchers.IO) {
                        val view = Framing.wholeScreen(frame, cacheDir)
                        val out = grounder.ground(view, question, raw = true)
                        ScreenAgent.Sighting(emptyList(), frame, view, out.raw)
                    }
                } else {
                    agent.locate(question)
                }
            }
            val seen = sighting
            if (seen == null) {
                finishWork("No frame arrived — the projection may have been revoked.")
                return@launch
            }

            val marks = seen.targets.map { g ->
                val p = seen.view.toScreen(g.centre())
                OverlayView.Mark(g.label, PointF(p.x, p.y), g.box?.let { b ->
                    val tl = seen.view.toScreen(intArrayOf(b[0], b[1]))
                    val br = seen.view.toScreen(intArrayOf(b[2], b[3]))
                    RectF(tl.x, tl.y, br.x, br.y)
                })
            }

            var tapped = false
            var after: Bitmap? = null
            if (shouldTap) {
                seen.firstPoint()?.let { target ->
                    tapped = ShownExecutor(AccessibilityExecutor(this@MainActivity)).tap(target.x, target.y)
                    if (tapped) {
                        delay(AFTER_TAP_MS)
                        after = captureScreen()
                    }
                }
            }

            robotPanel.visibility = android.view.View.GONE
            overlay.show(after ?: seen.frame, marks)
            result.text = if (raw) "%.1fs — %s".format(millis / 1000f, seen.raw.take(300))
                else describe(seen, marks, millis, shouldTap, tapped, after != null)
            AgentOverlay.update(AgentFace.State.DONE, "done")
            delay(1500)
            AgentOverlay.dismiss()
            finishWork(null)
        }
    }

    /**
     * Goal in, a sequence of actions out. Renders each step as it happens.
     *
     * [actions] false is the tap-only loop; true adds the planner, so the agent
     * can scroll to something that is not on screen, back out of a dead end, or
     * type. That costs one extra model call per step.
     */
    private fun runAgent(actions: Boolean) {
        val agent = this.agent ?: return
        val goal = instruction.text.toString().trim().ifEmpty { "open battery settings" }
        if (!TapService.isEnabled(this)) {
            status.text = "Agent mode needs the accessibility service."
            face.state = AgentFace.State.STOPPED
            return
        }
        startWork(if (actions) "Act: $goal" else "Agent: $goal")

        scope.launch {
            // The badge goes up *after* the step-back transition has finished.
            // Added mid-transition the window sits at READY_TO_SHOW with
            // isReadyForDisplay()=false: drawn, but held hidden by the window
            // manager until some later transition makes it re-evaluate. That is
            // why an earlier build showed no badge until the first tap landed.
            moveTaskToBack(true)
            delay(SETTLE_MS)
            AgentOverlay.show(this@MainActivity, goal)
            AgentOverlay.update(AgentFace.State.LOOKING, "looking at the screen")
            val log = StringBuilder()
            val bounds = getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds
            val render: suspend (Agent.Step, Bitmap, List<Grounded>, Framing.View) -> Unit =
                { step, frame, found, view ->
                val marks = found.map { g ->
                    val p = view.toScreen(g.centre())
                    OverlayView.Mark(g.label, PointF(p.x, p.y), null)
                }
                robotPanel.visibility = android.view.View.GONE
                overlay.show(frame, marks)
                // The badge reports the step it just finished, and aims at the
                // point it pressed, so the eye and the tap agree.
                step.at?.let { AgentOverlay.lookAt(it.x, it.y, bounds.width(), bounds.height()) }
                AgentOverlay.update(AgentFace.State.ACTING, "${step.index}. ${step.label}")
                log.append("${step.index}. ${step.label} — ${step.note}\n")
                result.text = log.toString()
                // Whatever comes next begins with another model call.
                AgentOverlay.update(AgentFace.State.THINKING, "thinking about the next step")
            }
            val outcome = if (actions) agent.operate(goal, MAX_STEPS, render)
                else agent.pursue(goal, MAX_STEPS, render)
            log.append("stopped: ${outcome.stoppedBecause}")
            result.text = log.toString()
            AgentOverlay.update(
                if (outcome.stoppedBecause.startsWith("the model reported")) AgentFace.State.DONE
                else AgentFace.State.STOPPED,
                outcome.stoppedBecause.take(46),
            )
            delay(2500)
            AgentOverlay.dismiss()
            finishWork(null)
            status.text = "Agent finished — ${outcome.steps.size} step(s)"
            face.state = AgentFace.State.DONE
        }
    }

    private fun describe(
        seen: ScreenAgent.Sighting,
        marks: List<OverlayView.Mark>,
        millis: Long,
        shouldTap: Boolean,
        tapped: Boolean,
        haveAfter: Boolean,
    ): String {
        val seconds = "%.1f".format(millis / 1000f)
        if (marks.isEmpty()) return "Nothing matched (${seconds}s): ${seen.raw.take(120)}"
        val where = seen.targets.joinToString { "${it.label} ${it.centre().joinToString()}" }
        val tail = when {
            !shouldTap -> "not tapped (Point only)"
            !TapService.isEnabled(this) -> "not tapped — accessibility service is off"
            tapped && haveAfter -> "tapped; showing the screen after"
            tapped -> "tapped, but no frame came back after it"
            else -> "tap was refused"
        }
        return "${seconds}s — $where — $tail"
    }

    /**
     * The face's reply to whatever is in the box.
     *
     * It says back what it understood, which is the cheapest way to make an app
     * feel answerable: you see that the thing read your sentence before you
     * commit to it.
     */
    private fun answer(typed: String) {
        val goal = typed.trim()
        when {
            agent == null -> {
                face.state = AgentFace.State.IDLE
                speech.text = "Load a model and I'll get to work."
            }
            goal.isEmpty() -> {
                face.state = AgentFace.State.IDLE
                speech.text = "Type a goal, then press Run."
            }
            askRadio.isChecked -> {
                face.state = AgentFace.State.LOOKING
                speech.text = "I'll ask the model that, about whatever is on screen."
            }
            agentRadio.isChecked || actRadio.isChecked -> {
                face.state = AgentFace.State.LOOKING
                speech.text = "I'll work toward \"${goal.take(40)}\", a step at a time."
            }
            else -> {
                face.state = AgentFace.State.LOOKING
                speech.text = "I'll find \"${goal.take(40)}\"" +
                    if (tapRadio.isChecked) " and press it." else " and mark it."
            }
        }
    }

    private fun startWork(message: String) {
        busy = true
        runButton.isEnabled = false
        overlay.clear()
        result.text = ""
        status.text = message
        speech.text = message
        face.state = AgentFace.State.THINKING
    }

    private fun finishWork(error: String?) {
        busy = false
        runButton.isEnabled = agent != null
        status.text = error ?: readyStatus
        if (error != null) speech.text = error
        face.state = if (error == null) AgentFace.State.DONE else AgentFace.State.STOPPED
    }

    // ---------- permissions ----------

    private fun requestNotificationsIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
            == PackageManager.PERMISSION_GRANTED
        ) return
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQ_NOTIFICATIONS)
    }
}

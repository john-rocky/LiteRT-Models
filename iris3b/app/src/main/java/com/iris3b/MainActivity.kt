package com.iris3b

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.iris3b.view.IrisScreen
import com.iris3b.view.IrisTheme

/**
 * Type a prompt, press Generate, get a 768x768 image made on this phone's GPU. A thin Compose host:
 * [MainViewModel] owns the model and the image in progress.
 *
 * Debuggable builds also take extras, so a computer can drive the app (none are needed to use it):
 *
 *     adb shell am start -n com.iris3b/.MainActivity --es verify apple
 *
 *  - `verify` apple | punct_num | japanese: that example from the reference noise; writes
 *    files/verify_<key>.npy (float32 [1, 3, 768, 768]) and files/verify_<key>.json
 *  - `prompt` <text> or `prompt_b64` <base64 of UTF-8>: put a prompt in the editor
 *  - `seed` (int or long): put a seed in the seed field
 *  - `autoplay` true: press Generate
 *  - `guard_kb` (int or long): stop an image when MemAvailable falls below this many kB
 *  - `on_top` true: show over the lock screen and keep the screen on
 */
class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)
        if (savedInstanceState == null) handleExtras(intent)
        setContent {
            val state by viewModel.uiState.collectAsStateWithLifecycle()
            IrisTheme {
                IrisScreen(
                    state = state,
                    onPromptChange = viewModel::setPrompt,
                    onExample = viewModel::selectExample,
                    onSeedChange = viewModel::setSeed,
                    onRandomSeed = viewModel::randomSeed,
                    onGenerate = viewModel::generate,
                    onStop = viewModel::stop,
                    onSave = viewModel::save,
                    onCheckAgain = viewModel::checkModel,
                )
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleExtras(intent)
    }

    private fun handleExtras(intent: Intent?) {
        if (intent == null) return
        val extras = intent.extras ?: return
        val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
        // Reopened from recents: the old extras come back; never re-run them.
        if (!debuggable || intent.flags and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY != 0) return
        if (extras.getBoolean(EXTRA_ON_TOP)) showOnTop()
        number(extras, EXTRA_GUARD_KB)?.let(viewModel::setGuardKb)
        val prompt = extras.getString(EXTRA_PROMPT_B64)?.let { String(Base64.decode(it, Base64.DEFAULT), Charsets.UTF_8) }
            ?: extras.getString(EXTRA_PROMPT)
        prompt?.let(viewModel::setPrompt)
        number(extras, EXTRA_SEED)?.let { viewModel.setSeed(it.toString()) }
        val verify = extras.getString(EXTRA_VERIFY)
        if (verify != null) {
            val example = Example.entries.firstOrNull { it.key == verify }
            if (example != null) viewModel.verify(example) else Log.w(TAG, "VERIFY unknown key $verify")
        } else if (extras.getBoolean(EXTRA_AUTOPLAY)) {
            viewModel.autoplay()
        }
    }

    private fun showOnTop() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
    }

    /** An int or long extra as a long (am start's --ei and --el). */
    @Suppress("DEPRECATION")
    private fun number(extras: Bundle, key: String): Long? = (extras.get(key) as? Number)?.toLong()

    private companion object {
        const val TAG = "IRIS3B_APP"
        const val EXTRA_VERIFY = "verify"
        const val EXTRA_PROMPT = "prompt"
        const val EXTRA_PROMPT_B64 = "prompt_b64"
        const val EXTRA_SEED = "seed"
        const val EXTRA_AUTOPLAY = "autoplay"
        const val EXTRA_GUARD_KB = "guard_kb"
        const val EXTRA_ON_TOP = "on_top"
    }
}

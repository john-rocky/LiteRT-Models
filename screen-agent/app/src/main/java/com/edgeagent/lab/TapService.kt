package com.edgeagent.lab

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Context
import android.content.Intent
import android.graphics.Path
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.edgeagent.sdk.ActionExecutor
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * The Executor seam: turns a screen coordinate into a real tap.
 *
 * An accessibility service is the only way to synthesise input into other apps
 * without root, and the user has to enable it by hand in system settings — it
 * cannot be granted programmatically. [isEnabled] reports that state so the UI
 * can say so plainly instead of failing silently.
 *
 * This service does nothing on its own: it listens for no events and reacts to
 * no windows. It exists purely so the gestures below have a dispatcher. The one
 * exception is [typeText] — see the note there for why typing has to reach into
 * the accessibility tree and what it is limited to.
 */
class TapService : AccessibilityService() {

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "connected")
    }

    override fun onUnbind(intent: Intent?): Boolean {
        instance = null
        return super.onUnbind(intent)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    companion object {
        private const val TAG = "TapService"
        private const val TAP_DURATION_MS = 60L

        @Volatile
        private var instance: TapService? = null

        /** True once the user has switched the service on in system settings. */
        fun isEnabled(context: Context): Boolean {
            val enabled = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
            ) ?: return false
            val id = "${context.packageName}/${TapService::class.java.name}"
            return enabled.split(':').any { it.equals(id, ignoreCase = true) }
        }

        /** Sends the user to the screen where they can switch it on. */
        fun openSettings(context: Context) {
            context.startActivity(
                Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }

        /**
         * Taps at a screen pixel. Returns false if the service is not connected
         * or the system refused the gesture.
         */
        fun tap(x: Float, y: Float, onDone: (Boolean) -> Unit) {
            val path = Path().apply { moveTo(x, y) }
            dispatch(path, TAP_DURATION_MS, onDone)
        }

        /** Drags between two points — this is how the agent scrolls. */
        fun swipe(
            fromX: Float,
            fromY: Float,
            toX: Float,
            toY: Float,
            durationMs: Long,
            onDone: (Boolean) -> Unit,
        ) {
            val path = Path().apply {
                moveTo(fromX, fromY)
                lineTo(toX, toY)
            }
            dispatch(path, durationMs, onDone)
        }

        /** The system back gesture. Needs no coordinate and cannot miss. */
        fun back(): Boolean {
            val service = instance ?: return false
            return service.performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK)
        }

        /**
         * Writes [text] into the field that currently has input focus.
         *
         * This is the one place the service touches the accessibility tree, and
         * it is a write, not a read: it asks for the input-focused node and sets
         * its text. Nothing here enumerates windows or reads back content — the
         * screen is still understood from pixels by the model. Typing is not
         * reachable any other way, though: gestures can press keys but cannot
         * put characters into a field.
         */
        fun typeText(text: String): Boolean {
            val service = instance ?: return false
            val focused = service.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            if (focused == null) {
                Log.w(TAG, "type requested but no field has input focus")
                return false
            }
            return try {
                val args = Bundle().apply {
                    putCharSequence(
                        AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                        text,
                    )
                }
                focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
            } finally {
                @Suppress("DEPRECATION")
                focused.recycle()
            }
        }

        /** True when some editable field currently holds input focus. */
        fun hasInputFocus(): Boolean {
            val service = instance ?: return false
            val focused = service.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: return false
            return try {
                focused.isEditable
            } finally {
                @Suppress("DEPRECATION")
                focused.recycle()
            }
        }

        private fun dispatch(path: Path, durationMs: Long, onDone: (Boolean) -> Unit) {
            val service = instance
            if (service == null) {
                Log.w(TAG, "gesture requested but service is not connected")
                onDone(false)
                return
            }
            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0L, durationMs))
                .build()
            val accepted = service.dispatchGesture(
                gesture,
                object : AccessibilityService.GestureResultCallback() {
                    override fun onCompleted(description: GestureDescription?) = onDone(true)
                    override fun onCancelled(description: GestureDescription?) = onDone(false)
                },
                null,
            )
            if (!accepted) {
                Log.w(TAG, "dispatchGesture was refused")
                onDone(false)
            }
        }
    }
}

/** Adapts the accessibility service to the SDK's [ActionExecutor] seam. */
class AccessibilityExecutor(private val context: Context) : ActionExecutor {
    override fun isReady(): Boolean = TapService.isEnabled(context)

    override suspend fun tap(x: Float, y: Float): Boolean =
        suspendCancellableCoroutine { cont ->
            TapService.tap(x, y) { ok -> if (cont.isActive) cont.resume(ok) }
        }

    override suspend fun swipe(
        fromX: Float,
        fromY: Float,
        toX: Float,
        toY: Float,
        durationMs: Long,
    ): Boolean = suspendCancellableCoroutine { cont ->
        TapService.swipe(fromX, fromY, toX, toY, durationMs) { ok ->
            if (cont.isActive) cont.resume(ok)
        }
    }

    override suspend fun back(): Boolean = TapService.back()

    override suspend fun typeText(text: String): Boolean = TapService.typeText(text)

    override fun hasTextFocus(): Boolean = TapService.hasInputFocus()
}

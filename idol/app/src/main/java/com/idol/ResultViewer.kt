package com.idol

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.view.ViewGroup
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.webkit.WebViewAssetLoader
import java.io.File
import java.io.FileInputStream

/**
 * The splat viewer (antimatter15/splat, vendored in assets/viewer/) in a WebView. The page and the .ply come from
 * the app itself through WebViewAssetLoader (https://appassets.androidplatform.net/...): no network, no file:// access.
 */
object ResultViewer {
    private const val HOST = "https://appassets.androidplatform.net"

    @SuppressLint("SetJavaScriptEnabled")
    fun show(context: Context, container: FrameLayout, resultsDir: File, ply: File): WebView {
        val loader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context))
            .addPathHandler("/results/", PlyHandler(resultsDir))
            .build()
        val web = WebView(context)
        web.settings.javaScriptEnabled = true
        web.settings.allowFileAccess = false
        web.settings.allowContentAccess = false
        web.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                loader.shouldInterceptRequest(request.url)

            // the viewer has no links; stay on the page
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean = true
        }
        container.removeAllViews()
        container.addView(web, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        val plyUrl = "$HOST/results/" + Uri.encode(ply.name)
        web.loadUrl("$HOST/assets/viewer/index.html?url=" + Uri.encode(plyUrl))
        return web
    }

    fun destroy(container: FrameLayout, web: WebView?) {
        if (web == null) return
        container.removeView(web)
        web.stopLoading()
        web.destroy()
    }

    /** Serves files of [dir] with a Content-Length (the viewer sizes its buffer from it). */
    private class PlyHandler(private val dir: File) : WebViewAssetLoader.PathHandler {
        override fun handle(path: String): WebResourceResponse? {
            val f = File(dir, path)
            val inside = runCatching { f.canonicalPath.startsWith(dir.canonicalPath + File.separator) }.getOrDefault(false)
            if (!inside || !f.isFile) {
                return WebResourceResponse("text/plain", "utf-8", 404, "Not Found", emptyMap(), null)
            }
            val headers = mapOf("Content-Length" to f.length().toString(), "Cache-Control" to "no-store")
            return WebResourceResponse("application/octet-stream", null, 200, "OK", headers, FileInputStream(f))
        }
    }
}

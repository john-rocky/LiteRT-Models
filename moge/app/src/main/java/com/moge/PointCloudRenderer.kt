package com.moge

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix as GLMatrix
import android.view.MotionEvent
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Minimal OpenGL ES 2.0 point cloud renderer with touch rotation.
 * Displays MoGe affine point map as colored points.
 */
class PointCloudView(context: Context) : GLSurfaceView(context) {

    private val renderer = PointCloudRendererImpl()
    private var prevX = 0f
    private var prevY = 0f

    init {
        setEGLContextClientVersion(2)
        setRenderer(renderer)
        renderMode = RENDERMODE_WHEN_DIRTY
    }

    fun setPointCloud(result: MoGeResult) {
        renderer.updatePoints(result)
        requestRender()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                prevX = event.x
                prevY = event.y
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.x - prevX
                val dy = event.y - prevY
                renderer.rotateY += dx * 0.3f
                renderer.rotateX += dy * 0.3f
                renderer.rotateX = renderer.rotateX.coerceIn(-90f, 90f)
                prevX = event.x
                prevY = event.y
                requestRender()
            }
        }
        return true
    }
}

private class PointCloudRendererImpl : GLSurfaceView.Renderer {

    var rotateX = 15f
    var rotateY = 0f

    private var program = 0
    private var vertexBuffer: Int = 0
    private var colorBuffer: Int = 0
    private var numPoints = 0
    private var centerX = 0f
    private var centerY = 0f
    private var centerZ = 0f
    private var scale = 1f

    private val mvpMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val projMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val tempMatrix = FloatArray(16)

    @Volatile
    private var pendingPositions: FloatArray? = null
    @Volatile
    private var pendingColors: FloatArray? = null

    fun updatePoints(result: MoGeResult) {
        val w = result.width
        val h = result.height
        val step = 2  // subsample for performance
        val count = (h / step) * (w / step)

        val positions = FloatArray(count * 3)
        val colors = FloatArray(count * 4)

        // Compute center and scale from valid points
        var cx = 0f; var cy = 0f; var cz = 0f; var valid = 0
        for (y in 0 until h step step) {
            for (x in 0 until w step step) {
                val i = y * w + x
                if (result.mask[i] > 0.5f) {
                    cx += result.points[i * 3]
                    cy += result.points[i * 3 + 1]
                    cz += result.points[i * 3 + 2]
                    valid++
                }
            }
        }
        if (valid > 0) { cx /= valid; cy /= valid; cz /= valid }

        var maxDist = 1f
        var idx = 0
        for (y in 0 until h step step) {
            for (x in 0 until w step step) {
                val i = y * w + x
                val px = result.points[i * 3] - cx
                val py = result.points[i * 3 + 1] - cy
                val pz = result.points[i * 3 + 2] - cz

                positions[idx * 3] = px
                positions[idx * 3 + 1] = -py  // flip Y for OpenGL
                positions[idx * 3 + 2] = -pz  // flip Z so depth goes into screen

                // Color from normal map
                val nr = (result.normal[i * 3] + 1f) * 0.5f
                val ng = (result.normal[i * 3 + 1] + 1f) * 0.5f
                val nb = (result.normal[i * 3 + 2] + 1f) * 0.5f
                val alpha = if (result.mask[i] > 0.5f) 1f else 0f

                colors[idx * 4] = nr
                colors[idx * 4 + 1] = ng
                colors[idx * 4 + 2] = nb
                colors[idx * 4 + 3] = alpha

                val d = Math.sqrt((px * px + py * py + pz * pz).toDouble()).toFloat()
                if (d > maxDist && result.mask[i] > 0.5f) maxDist = d

                idx++
            }
        }

        centerX = 0f; centerY = 0f; centerZ = 0f
        scale = 2f / maxDist
        pendingPositions = positions
        pendingColors = colors
        numPoints = count
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)

        val vs = compileShader(GLES20.GL_VERTEX_SHADER, VERT_SHADER)
        val fs = compileShader(GLES20.GL_FRAGMENT_SHADER, FRAG_SHADER)
        program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vs)
        GLES20.glAttachShader(program, fs)
        GLES20.glLinkProgram(program)

        val bufs = IntArray(2)
        GLES20.glGenBuffers(2, bufs, 0)
        vertexBuffer = bufs[0]
        colorBuffer = bufs[1]
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val aspect = width.toFloat() / height
        GLMatrix.perspectiveM(projMatrix, 0, 45f, aspect, 0.01f, 100f)
    }

    override fun onDrawFrame(gl: GL10?) {
        // Upload pending data
        pendingPositions?.let { pos ->
            val pb = ByteBuffer.allocateDirect(pos.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            pb.put(pos).position(0)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer)
            GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, pos.size * 4, pb, GLES20.GL_STATIC_DRAW)
            pendingPositions = null
        }
        pendingColors?.let { col ->
            val cb = ByteBuffer.allocateDirect(col.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            cb.put(col).position(0)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, colorBuffer)
            GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, col.size * 4, cb, GLES20.GL_STATIC_DRAW)
            pendingColors = null
        }

        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        if (numPoints == 0) return

        GLES20.glUseProgram(program)

        // Model matrix: center + scale + rotate
        GLMatrix.setIdentityM(modelMatrix, 0)
        GLMatrix.scaleM(modelMatrix, 0, scale, scale, scale)
        GLMatrix.setIdentityM(tempMatrix, 0)
        GLMatrix.rotateM(tempMatrix, 0, rotateX, 1f, 0f, 0f)
        GLMatrix.multiplyMM(mvpMatrix, 0, tempMatrix, 0, modelMatrix, 0)
        System.arraycopy(mvpMatrix, 0, modelMatrix, 0, 16)
        GLMatrix.setIdentityM(tempMatrix, 0)
        GLMatrix.rotateM(tempMatrix, 0, rotateY, 0f, 1f, 0f)
        GLMatrix.multiplyMM(mvpMatrix, 0, tempMatrix, 0, modelMatrix, 0)
        System.arraycopy(mvpMatrix, 0, modelMatrix, 0, 16)

        // View matrix
        GLMatrix.setLookAtM(viewMatrix, 0, 0f, 0f, 3f, 0f, 0f, 0f, 0f, 1f, 0f)

        // MVP
        GLMatrix.multiplyMM(tempMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        GLMatrix.multiplyMM(mvpMatrix, 0, projMatrix, 0, tempMatrix, 0)

        val mvpLoc = GLES20.glGetUniformLocation(program, "uMVP")
        GLES20.glUniformMatrix4fv(mvpLoc, 1, false, mvpMatrix, 0)

        // Vertices
        val posLoc = GLES20.glGetAttribLocation(program, "aPosition")
        GLES20.glEnableVertexAttribArray(posLoc)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer)
        GLES20.glVertexAttribPointer(posLoc, 3, GLES20.GL_FLOAT, false, 0, 0)

        // Colors
        val colLoc = GLES20.glGetAttribLocation(program, "aColor")
        GLES20.glEnableVertexAttribArray(colLoc)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, colorBuffer)
        GLES20.glVertexAttribPointer(colLoc, 4, GLES20.GL_FLOAT, false, 0, 0)

        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, numPoints)

        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(colLoc)
    }

    private fun compileShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        return shader
    }

    companion object {
        private const val VERT_SHADER = """
            uniform mat4 uMVP;
            attribute vec3 aPosition;
            attribute vec4 aColor;
            varying vec4 vColor;
            void main() {
                gl_Position = uMVP * vec4(aPosition, 1.0);
                gl_PointSize = 3.0;
                vColor = aColor;
            }
        """
        private const val FRAG_SHADER = """
            precision mediump float;
            varying vec4 vColor;
            void main() {
                if (vColor.a < 0.5) discard;
                gl_FragColor = vec4(vColor.rgb, 1.0);
            }
        """
    }
}

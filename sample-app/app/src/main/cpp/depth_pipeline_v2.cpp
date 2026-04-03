// LiteRT C++ SDK zero-copy depth pipeline.
// Based on official litert-samples/c++_segmentation/main_gpu.cc pattern.
// FP32 via GpuOptions + GL buffer zero-copy + RunAsync.

#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <GLES3/gl31.h>
#include <EGL/egl.h>

#include <cstring>
#include <vector>
#include <string>
#include <chrono>

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/options/litert_gpu_options.h"

#define TAG "DepthPipelineV2"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Inferno LUT
static const uint8_t INF_R[] = {0,0,1,1,2,2,3,4,5,6,7,8,9,10,12,13,15,16,18,19,21,23,24,26,28,30,32,34,36,38,40,42,44,47,49,51,53,56,58,60,62,65,67,69,72,74,76,78,81,83,85,88,90,92,94,96,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,128,130,132,134,135,137,139,140,142,144,145,147,148,150,151,153,154,156,157,158,160,161,162,163,165,166,167,168,169,170,172,173,174,175,176,177,178,179,179,180,181,182,183,184,185,186,187,187,188,189,190,191,191,192,193,194,195,195,196,197,198,198,199,200,200,201,202,202,203,204,204,205,205,206,207,207,208,208,209,209,210,210,211,211,211,212,212,213,213,213,214,214,214,215,215,215,215,216,216,216,216,217,217,217,217,218,218,218,218,218,219,219,219,219,219,220,220,220,220,220,221,221,221,221,221,222,222,222,222,223,223,223,223,224,224,224,225,225,225,226,226,227,227,228,228,229,229,230,230,231,231,232,233,233,234,235,236,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,252,253,254,254,255,255,255,255,255,255,252};
static const uint8_t INF_G[] = {0,0,1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,10,10,10,11,11,11,12,12,13,13,14,14,14,15,16,16,17,17,18,18,19,20,20,21,22,22,23,24,24,25,26,27,28,29,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,51,52,53,54,56,57,58,60,61,62,64,65,67,68,70,71,72,74,75,77,79,80,82,83,85,86,88,90,91,93,94,96,98,99,101,103,104,106,108,109,111,113,114,116,118,120,121,123,125,126,128,130,132,133,135,137,138,140,142,144,145,147,149,150,152,154,155,157,159,161,162,164,166,167,169,171,172,174,176,177,179,181,183,184,186,188,189,191,193,194,196,198,199,201,203,204,206,208,209,211,213,214,216,218,219,221,223,224,226,228,229,231,232,234,236,237,239,240,242,244,245,247,248,250,251,253,254,254,254,254,254,254,254,254,254,254,254,254,254,253,253,253,253,252,252,252,251,251,250,250,249,249,248,247,246,245,244,243,242,240,236};
static const uint8_t INF_B[] = {4,5,7,9,11,14,16,19,22,24,27,30,33,35,38,41,44,47,49,52,55,57,60,62,65,67,69,71,73,75,77,79,80,82,83,84,85,86,87,87,88,88,89,89,89,89,89,89,88,88,88,87,87,86,86,85,84,84,83,82,82,81,80,79,78,77,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,53,52,51,50,49,49,48,47,46,46,45,45,44,43,43,42,42,41,41,40,40,39,39,39,38,38,38,38,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,39,39,39,40,40,41,41,42,42,43,43,44,45,45,46,47,48,49,49,50,52,53,54,55,56,58,59,61,62,64,65,67,69,71,73,75,77,79,81,83,85,88,90,93,95,98,100,103,106,109,112,115,118,121,124,127,131,134,137,141,144,148,152,155,159,163,167,170,174,178,182,186,190,194,198,202,206,210,214,218,222,226,230,234,237,238,238,239,240,241,242,242,243,244,245,246,246,247,248,248,249,250,251,251,252,252,253,254,254,255,255,255,255,255,255,255,255,252};

struct Pipeline {
    std::unique_ptr<litert::Environment> env;
    std::unique_ptr<litert::CompiledModel> model;
    std::vector<litert::TensorBuffer> inputBuffers;
    std::vector<litert::TensorBuffer> outputBuffers;
    bool useGlBuffers = false;

    int inputW = 518, inputH = 518;
    int outputW = 518, outputH = 518;

    std::vector<float> inputFloats;
    std::vector<float> outputFloats;
    std::vector<uint8_t> rgbaPixels;

    // GL resources for rendering
    GLuint depthTexture = 0;
    GLuint quadProgram = 0;
    GLuint quadVao = 0, quadVbo = 0;

    bool initialized = false;
    std::vector<uint8_t> modelData;

    static constexpr float MEAN[] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[] = {0.229f, 0.224f, 0.225f};
};
constexpr float Pipeline::MEAN[];
constexpr float Pipeline::STD[];

static Pipeline g;

// Vertex/fragment shaders for fullscreen quad
static const char* kVertSrc = R"(#version 310 es
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = vec2(aPos.x, 1.0 - aPos.y);
    gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);
})";

static const char* kFragSrc = R"(#version 310 es
precision mediump float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() { fragColor = texture(uTex, vUV); }
)";

static GLuint compileProgram(const char* vs, const char* fs) {
    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[256]; glGetShaderInfoLog(s, 256, nullptr, log); LOGE("Shader: %s", log); }
        return s;
    };
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static void initGl() {
    // Texture
    glGenTextures(1, &g.depthTexture);
    glBindTexture(GL_TEXTURE_2D, g.depthTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, g.outputW, g.outputH);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Quad
    g.quadProgram = compileProgram(kVertSrc, kFragSrc);
    float verts[] = {0,0, 1,0, 0,1, 1,1};
    glGenBuffers(1, &g.quadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, g.quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glGenVertexArrays(1, &g.quadVao);
    glBindVertexArray(g.quadVao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    g.inputFloats.resize(g.inputW * g.inputH * 3);
    g.outputFloats.resize(g.outputW * g.outputH);
    g.rgbaPixels.resize(g.outputW * g.outputH * 4);

    LOGI("GL init done: tex=%u prog=%u", g.depthTexture, g.quadProgram);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeInitGl(
    JNIEnv* env, jobject, jint inW, jint inH, jint outW, jint outH) {
    g.inputW = inW; g.inputH = inH; g.outputW = outW; g.outputH = outH;
    initGl();
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeInitLiteRT(
    JNIEnv* jenv, jobject, jobject assetMgr, jstring modelPath,
    jint inW, jint inH, jint outW, jint outH, jlong) {

    // Load model data
    AAssetManager* mgr = AAssetManager_fromJava(jenv, assetMgr);
    const char* path = jenv->GetStringUTFChars(modelPath, nullptr);
    AAsset* asset = AAssetManager_open(mgr, path, AASSET_MODE_BUFFER);
    jenv->ReleaseStringUTFChars(modelPath, path);
    if (!asset) { LOGE("Failed to open model"); return JNI_FALSE; }

    size_t sz = AAsset_getLength(asset);
    g.modelData.resize(sz);
    memcpy(g.modelData.data(), AAsset_getBuffer(asset), sz);
    AAsset_close(asset);

    // 1. Create environment
    auto envResult = litert::Environment::Create({});
    if (!envResult) { LOGE("Environment::Create failed"); return JNI_FALSE; }
    g.env = std::make_unique<litert::Environment>(std::move(*envResult));
    LOGI("Environment created");

    // 2. Create options with FP32 GPU
    auto optResult = litert::Options::Create();
    if (!optResult) { LOGE("Options::Create failed"); return JNI_FALSE; }
    auto options = std::move(*optResult);

    auto gpuResult = options.GetGpuOptions();
    if (gpuResult) {
        auto& gpuOpts = *gpuResult;
        gpuOpts.SetPrecision(litert::GpuOptions::Precision::kFp32);
        LOGI("GPU options: FP32");
    } else {
        LOGE("GetGpuOptions failed");
    }
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);

    // 3. Compile model from buffer
    auto modelResult = litert::CompiledModel::Create(
        *g.env,
        litert::BufferRef<uint8_t>(g.modelData.data(), g.modelData.size()),
        options);
    if (!modelResult) { LOGE("CompiledModel::Create failed"); return JNI_FALSE; }
    g.model = std::make_unique<litert::CompiledModel>(std::move(*modelResult));
    LOGI("CompiledModel created with FP32 GPU");

    // 4. Try GL buffer zero-copy, fall back to managed buffers
    bool glOk = false;

    // Try creating GL buffer tensors (official pattern from main_gpu.cc)
    {
        // Build fixed-shape tensor types (model may report -1 for batch dim)
        LiteRtRankedTensorType inRtt{};
        inRtt.element_type = kLiteRtElementTypeFloat32;
        inRtt.layout.rank = 4;
        inRtt.layout.dimensions[0] = 1;
        inRtt.layout.dimensions[1] = g.inputH;
        inRtt.layout.dimensions[2] = g.inputW;
        inRtt.layout.dimensions[3] = 3;
        litert::RankedTensorType inTT(inRtt);

        LiteRtRankedTensorType outRtt{};
        outRtt.element_type = kLiteRtElementTypeFloat32;
        outRtt.layout.rank = 4;
        outRtt.layout.dimensions[0] = 1;
        outRtt.layout.dimensions[1] = g.outputH;
        outRtt.layout.dimensions[2] = g.outputW;
        outRtt.layout.dimensions[3] = 1;
        litert::RankedTensorType outTT(outRtt);

        size_t inSize = g.inputH * g.inputW * 3 * sizeof(float);
        size_t outSize = g.outputH * g.outputW * sizeof(float);
        LOGI("Buffer sizes: in=%zu out=%zu", inSize, outSize);

        auto outBuf = litert::TensorBuffer::CreateManaged(
            *g.env, litert::TensorBufferType::kGlBuffer, outTT, outSize);
        if (outBuf) {
            auto inBuf = litert::TensorBuffer::CreateManaged(
                *g.env, litert::TensorBufferType::kGlBuffer, inTT, inSize);
            if (inBuf) {
                g.inputBuffers.push_back(std::move(*inBuf));
                g.outputBuffers.push_back(std::move(*outBuf));
                g.useGlBuffers = true;
                glOk = true;
                LOGI("GL buffer zero-copy created!");
            } else {
                LOGI("GL input buffer failed");
            }
        } else {
            LOGI("GL output buffer failed");
        }
    }

    // Fallback: default managed buffers
    if (!glOk) {
        auto inResult = g.model->CreateInputBuffers();
        auto outResult = g.model->CreateOutputBuffers();
        if (inResult && outResult) {
            g.inputBuffers = std::move(*inResult);
            g.outputBuffers = std::move(*outResult);
            auto btResult = g.outputBuffers[0].BufferType();
            if (btResult) LOGI("Output buffer type: %d", static_cast<int>(*btResult));
        }
        LOGI("Using managed buffers (fallback)");
    }

    g.initialized = true;
    LOGI("Pipeline ready: %dx%d → %dx%d (glBuffers=%d)", inW, inH, outW, outH, g.useGlBuffers);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeProcessFrame(
    JNIEnv* env, jobject, jintArray pixels, jint w, jint h, jint rot) {

    if (!g.initialized) return;

    // Preprocess
    jint* data = env->GetIntArrayElements(pixels, nullptr);
    float* out = g.inputFloats.data();
    for (int y = 0; y < g.inputH; y++) {
        for (int x = 0; x < g.inputW; x++) {
            int sx, sy;
            switch (rot) {
                case 90:  sx = y*w/g.inputH; sy = (g.inputW-1-x)*h/g.inputW; break;
                case 180: sx = (g.inputW-1-x)*w/g.inputW; sy = (g.inputH-1-y)*h/g.inputH; break;
                case 270: sx = (g.inputH-1-y)*w/g.inputH; sy = x*h/g.inputW; break;
                default:  sx = x*w/g.inputW; sy = y*h/g.inputH; break;
            }
            if (sx >= w) sx = w-1; if (sy >= h) sy = h-1;
            uint32_t argb = ((uint32_t*)data)[sy * w + sx];
            float r = ((argb >> 16) & 0xFF) / 255.0f;
            float g_ = ((argb >> 8) & 0xFF) / 255.0f;
            float b = (argb & 0xFF) / 255.0f;
            int idx = (y * g.inputW + x) * 3;
            out[idx+0] = (r - Pipeline::MEAN[0]) / Pipeline::STD[0];
            out[idx+1] = (g_ - Pipeline::MEAN[1]) / Pipeline::STD[1];
            out[idx+2] = (b - Pipeline::MEAN[2]) / Pipeline::STD[2];
        }
    }
    env->ReleaseIntArrayElements(pixels, data, JNI_ABORT);

    // Write input — GL buffer: use glBufferSubData, managed: use Write<float>
    auto t0 = std::chrono::high_resolution_clock::now();
    if (g.useGlBuffers) {
        auto glBuf = g.inputBuffers[0].GetGlBuffer();
        if (!glBuf) { LOGE("GetGlBuffer failed for input"); return; }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, glBuf->id);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        g.inputFloats.size() * sizeof(float), g.inputFloats.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    } else {
        auto writeResult = g.inputBuffers[0].Write<float>(
            absl::MakeConstSpan(g.inputFloats));
        if (!writeResult) { LOGE("Write input failed"); return; }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // Run inference
    auto runResult = g.model->Run(
        absl::MakeSpan(g.inputBuffers), absl::MakeSpan(g.outputBuffers));
    if (!runResult) {
        static int errLog = 0;
        if (errLog++ < 5) LOGE("Run failed");
        return;
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    // Read output — GL buffer: get SSBO ID, managed: Read to CPU
    auto t3 = t2;
    if (g.useGlBuffers) {
        // GL zero-copy: output is already in a GL SSBO
        // Just read for logging on first frames
        auto glBuf = g.outputBuffers[0].GetGlBuffer();
        if (glBuf) {
            static int glLog = 0;
            if (glLog++ < 5) LOGI("GL output SSBO id=%u size=%zu", glBuf->id, glBuf->size_bytes);
            // TODO: use glBuf->id for compute shader colormap (skip CPU read)
            // For now, also read to CPU for colormap
            g.outputBuffers[0].Read<float>(absl::MakeSpan(g.outputFloats));
        }
        t3 = std::chrono::high_resolution_clock::now();
    } else {
        g.outputBuffers[0].Read<float>(absl::MakeSpan(g.outputFloats));
        t3 = std::chrono::high_resolution_clock::now();
    }

    long writeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    long runMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    long readMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();

    static int logCount = 0;
    if (logCount++ < 20) {
        LOGI("write=%ldms run=%ldms read=%ldms out[0..2]=%f %f %f",
             writeMs, runMs, readMs, g.outputFloats[0], g.outputFloats[1], g.outputFloats[2]);
    }

    // CPU colormap → RGBA
    int n = g.outputW * g.outputH;
    float dmin = g.outputFloats[0], dmax = g.outputFloats[0];
    for (int i = 0; i < n; i += 16) {
        float v = g.outputFloats[i];
        if (v < dmin) dmin = v; if (v > dmax) dmax = v;
    }
    float range = dmax - dmin;
    if (range < 1e-6f) range = 1.0f;
    float scale = 254.0f / range;
    for (int i = 0; i < n; i++) {
        int idx = (int)((g.outputFloats[i] - dmin) * scale);
        if (idx < 0) idx = 0; if (idx > 254) idx = 254;
        g.rgbaPixels[i*4+0] = INF_R[idx];
        g.rgbaPixels[i*4+1] = INF_G[idx];
        g.rgbaPixels[i*4+2] = INF_B[idx];
        g.rgbaPixels[i*4+3] = 255;
    }

    // Upload to texture
    glBindTexture(GL_TEXTURE_2D, g.depthTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g.outputW, g.outputH,
                    GL_RGBA, GL_UNSIGNED_BYTE, g.rgbaPixels.data());
}

JNIEXPORT void JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeRender(
    JNIEnv*, jobject, jint viewW, jint viewH) {
    if (!g.initialized) return;
    glViewport(0, 0, viewW, viewH);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g.quadProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g.depthTexture);
    glUniform1i(glGetUniformLocation(g.quadProgram, "uTex"), 0);
    glBindVertexArray(g.quadVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

JNIEXPORT void JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeDestroy(JNIEnv*, jobject) {
    g.model.reset(); g.env.reset();
    g.inputBuffers.clear(); g.outputBuffers.clear();
    g.initialized = false;
}

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeIsInitialized(JNIEnv*, jobject) {
    return g.initialized ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeIsZeroCopy(JNIEnv*, jobject) {
    return g.useGlBuffers ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeLockAndColormap(
    JNIEnv*, jobject, jlong, jintArray, jint, jint) { return -1; }

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeSetHandles(
    JNIEnv*, jobject, jlong, jlong) { return JNI_FALSE; }

} // extern "C"

// C++ NDK zero-copy depth pipeline.
// Camera → CPU preprocess → input SSBO → LiteRT GPU inference → output SSBO
//   → compute shader (Inferno colormap) → texture → fullscreen quad render.
// Eliminates the 321ms readFloat() bottleneck by keeping data on GPU.

#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>

#include "litert_c_api.h"

#define TAG "DepthPipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// GLSL shader sources
// ---------------------------------------------------------------------------

// Min/max reduction compute shader (pass 1)
// Uses atomicMin/atomicMax on uint — works for positive floats (IEEE 754 ordering)
static const char* kMinMaxComputeShader = R"glsl(#version 310 es
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer DepthBuf {
    float depth[];
};

layout(std430, binding = 1) buffer MinMaxBuf {
    uint globalMin;
    uint globalMax;
};

uniform int uCount;

shared uint localMin;
shared uint localMax;

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;

    // Initialize shared memory
    if (tid == 0u) {
        localMin = 0x7F7FFFFFu; // float max as uint
        localMax = 0u;
    }
    memoryBarrierShared();
    barrier();

    // Each thread processes one element
    if (gid < uint(uCount)) {
        uint val = floatBitsToUint(depth[gid]);
        atomicMin(localMin, val);
        atomicMax(localMax, val);
    }

    memoryBarrierShared();
    barrier();

    // First thread writes to global
    if (tid == 0u) {
        atomicMin(globalMin, localMin);
        atomicMax(globalMax, localMax);
    }
}
)glsl";

// Compute shader: read depth SSBO → Inferno colormap → write RGBA texture (pass 2)
static const char* kColormapComputeShader = R"glsl(#version 310 es
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer DepthBuf {
    float depth[];
};

layout(std430, binding = 1) readonly buffer MinMaxBuf {
    uint globalMin;
    uint globalMax;
};

layout(rgba8, binding = 0) writeonly uniform highp image2D outImage;

uniform int uWidth;
uniform int uHeight;

// Inferno colormap LUT (256 entries packed as vec3)
// Stored as uniform to avoid SSBO overhead
const vec3 infernoLut[256] = vec3[256](
    vec3(0.000, 0.000, 0.016), vec3(0.000, 0.000, 0.020), vec3(0.004, 0.004, 0.027),
    vec3(0.004, 0.004, 0.035), vec3(0.008, 0.004, 0.043), vec3(0.008, 0.004, 0.055),
    vec3(0.012, 0.004, 0.063), vec3(0.016, 0.008, 0.075), vec3(0.020, 0.008, 0.086),
    vec3(0.024, 0.008, 0.094), vec3(0.027, 0.008, 0.106), vec3(0.031, 0.012, 0.118),
    vec3(0.035, 0.012, 0.129), vec3(0.039, 0.012, 0.137), vec3(0.047, 0.012, 0.149),
    vec3(0.051, 0.012, 0.161), vec3(0.059, 0.016, 0.173), vec3(0.063, 0.016, 0.184),
    vec3(0.071, 0.016, 0.192), vec3(0.075, 0.016, 0.204), vec3(0.082, 0.016, 0.216),
    vec3(0.090, 0.020, 0.224), vec3(0.094, 0.020, 0.235), vec3(0.102, 0.020, 0.243),
    vec3(0.110, 0.020, 0.255), vec3(0.118, 0.020, 0.263), vec3(0.125, 0.024, 0.271),
    vec3(0.133, 0.024, 0.278), vec3(0.141, 0.024, 0.286), vec3(0.149, 0.024, 0.294),
    vec3(0.157, 0.027, 0.302), vec3(0.165, 0.027, 0.310), vec3(0.173, 0.027, 0.314),
    vec3(0.184, 0.027, 0.322), vec3(0.192, 0.031, 0.325), vec3(0.200, 0.031, 0.329),
    vec3(0.208, 0.031, 0.333), vec3(0.220, 0.031, 0.337), vec3(0.227, 0.035, 0.341),
    vec3(0.235, 0.035, 0.341), vec3(0.243, 0.035, 0.345), vec3(0.255, 0.039, 0.345),
    vec3(0.263, 0.039, 0.349), vec3(0.271, 0.039, 0.349), vec3(0.282, 0.043, 0.349),
    vec3(0.290, 0.043, 0.349), vec3(0.298, 0.043, 0.349), vec3(0.306, 0.047, 0.349),
    vec3(0.318, 0.047, 0.345), vec3(0.325, 0.051, 0.345), vec3(0.333, 0.051, 0.345),
    vec3(0.345, 0.055, 0.341), vec3(0.353, 0.055, 0.341), vec3(0.361, 0.055, 0.337),
    vec3(0.369, 0.059, 0.337), vec3(0.376, 0.063, 0.333), vec3(0.388, 0.063, 0.329),
    vec3(0.396, 0.067, 0.329), vec3(0.404, 0.067, 0.325), vec3(0.412, 0.071, 0.322),
    vec3(0.420, 0.071, 0.322), vec3(0.427, 0.075, 0.318), vec3(0.435, 0.078, 0.314),
    vec3(0.443, 0.078, 0.310), vec3(0.451, 0.082, 0.306), vec3(0.459, 0.086, 0.302),
    vec3(0.467, 0.086, 0.302), vec3(0.475, 0.090, 0.298), vec3(0.482, 0.094, 0.294),
    vec3(0.490, 0.094, 0.290), vec3(0.498, 0.098, 0.286), vec3(0.502, 0.102, 0.282),
    vec3(0.510, 0.106, 0.278), vec3(0.518, 0.110, 0.275), vec3(0.525, 0.114, 0.271),
    vec3(0.529, 0.114, 0.267), vec3(0.537, 0.118, 0.263), vec3(0.545, 0.122, 0.259),
    vec3(0.549, 0.125, 0.255), vec3(0.557, 0.129, 0.251), vec3(0.565, 0.133, 0.247),
    vec3(0.569, 0.137, 0.243), vec3(0.576, 0.141, 0.239), vec3(0.580, 0.145, 0.235),
    vec3(0.588, 0.149, 0.231), vec3(0.592, 0.153, 0.227), vec3(0.600, 0.157, 0.224),
    vec3(0.604, 0.161, 0.220), vec3(0.612, 0.165, 0.216), vec3(0.616, 0.169, 0.212),
    vec3(0.620, 0.176, 0.208), vec3(0.627, 0.180, 0.208), vec3(0.631, 0.184, 0.204),
    vec3(0.635, 0.188, 0.200), vec3(0.639, 0.192, 0.196), vec3(0.647, 0.200, 0.192),
    vec3(0.651, 0.204, 0.192), vec3(0.655, 0.208, 0.188), vec3(0.659, 0.212, 0.184),
    vec3(0.663, 0.220, 0.180), vec3(0.667, 0.224, 0.180), vec3(0.675, 0.227, 0.176),
    vec3(0.678, 0.235, 0.176), vec3(0.682, 0.239, 0.173), vec3(0.686, 0.243, 0.169),
    vec3(0.690, 0.251, 0.169), vec3(0.694, 0.255, 0.165), vec3(0.698, 0.263, 0.165),
    vec3(0.702, 0.267, 0.161), vec3(0.702, 0.275, 0.161), vec3(0.706, 0.278, 0.157),
    vec3(0.710, 0.282, 0.157), vec3(0.714, 0.290, 0.153), vec3(0.718, 0.294, 0.153),
    vec3(0.722, 0.302, 0.153), vec3(0.725, 0.310, 0.149), vec3(0.729, 0.314, 0.149),
    vec3(0.733, 0.322, 0.149), vec3(0.733, 0.325, 0.149), vec3(0.737, 0.333, 0.145),
    vec3(0.741, 0.337, 0.145), vec3(0.745, 0.345, 0.145), vec3(0.749, 0.353, 0.145),
    vec3(0.749, 0.357, 0.145), vec3(0.753, 0.365, 0.145), vec3(0.757, 0.369, 0.145),
    vec3(0.761, 0.376, 0.145), vec3(0.765, 0.384, 0.145), vec3(0.765, 0.388, 0.145),
    vec3(0.769, 0.396, 0.145), vec3(0.773, 0.404, 0.145), vec3(0.776, 0.408, 0.145),
    vec3(0.776, 0.416, 0.145), vec3(0.780, 0.424, 0.145), vec3(0.784, 0.427, 0.149),
    vec3(0.784, 0.435, 0.149), vec3(0.788, 0.443, 0.149), vec3(0.792, 0.447, 0.149),
    vec3(0.792, 0.455, 0.153), vec3(0.796, 0.463, 0.153), vec3(0.800, 0.471, 0.153),
    vec3(0.800, 0.475, 0.157), vec3(0.804, 0.482, 0.157), vec3(0.804, 0.490, 0.161),
    vec3(0.808, 0.494, 0.161), vec3(0.812, 0.502, 0.165), vec3(0.812, 0.510, 0.165),
    vec3(0.816, 0.518, 0.169), vec3(0.816, 0.522, 0.169), vec3(0.820, 0.529, 0.173),
    vec3(0.820, 0.537, 0.176), vec3(0.824, 0.541, 0.176), vec3(0.824, 0.549, 0.180),
    vec3(0.827, 0.557, 0.184), vec3(0.827, 0.565, 0.188), vec3(0.827, 0.569, 0.192),
    vec3(0.831, 0.576, 0.192), vec3(0.831, 0.584, 0.196), vec3(0.835, 0.588, 0.204),
    vec3(0.835, 0.596, 0.208), vec3(0.835, 0.604, 0.212), vec3(0.839, 0.608, 0.216),
    vec3(0.839, 0.616, 0.220), vec3(0.839, 0.624, 0.227), vec3(0.843, 0.631, 0.231),
    vec3(0.843, 0.635, 0.239), vec3(0.843, 0.643, 0.243), vec3(0.843, 0.651, 0.251),
    vec3(0.847, 0.655, 0.255), vec3(0.847, 0.663, 0.263), vec3(0.847, 0.671, 0.271),
    vec3(0.847, 0.675, 0.278), vec3(0.851, 0.682, 0.286), vec3(0.851, 0.690, 0.294),
    vec3(0.851, 0.694, 0.302), vec3(0.851, 0.702, 0.310), vec3(0.855, 0.710, 0.318),
    vec3(0.855, 0.718, 0.325), vec3(0.855, 0.722, 0.333), vec3(0.855, 0.729, 0.345),
    vec3(0.855, 0.737, 0.353), vec3(0.859, 0.741, 0.365), vec3(0.859, 0.749, 0.373),
    vec3(0.859, 0.757, 0.384), vec3(0.859, 0.761, 0.392), vec3(0.859, 0.769, 0.404),
    vec3(0.863, 0.776, 0.416), vec3(0.863, 0.780, 0.427), vec3(0.863, 0.788, 0.439),
    vec3(0.863, 0.796, 0.451), vec3(0.863, 0.800, 0.463), vec3(0.867, 0.808, 0.475),
    vec3(0.867, 0.816, 0.486), vec3(0.867, 0.820, 0.498), vec3(0.867, 0.827, 0.514),
    vec3(0.867, 0.835, 0.525), vec3(0.871, 0.839, 0.537), vec3(0.871, 0.847, 0.553),
    vec3(0.871, 0.855, 0.565), vec3(0.871, 0.859, 0.580), vec3(0.875, 0.867, 0.596),
    vec3(0.875, 0.875, 0.608), vec3(0.875, 0.878, 0.624), vec3(0.875, 0.886, 0.639),
    vec3(0.878, 0.894, 0.655), vec3(0.878, 0.898, 0.667), vec3(0.878, 0.906, 0.682),
    vec3(0.882, 0.910, 0.698), vec3(0.882, 0.918, 0.714), vec3(0.882, 0.925, 0.729),
    vec3(0.886, 0.929, 0.745), vec3(0.886, 0.937, 0.761), vec3(0.890, 0.941, 0.776),
    vec3(0.890, 0.949, 0.792), vec3(0.894, 0.957, 0.808), vec3(0.894, 0.961, 0.824),
    vec3(0.898, 0.969, 0.839), vec3(0.898, 0.973, 0.855), vec3(0.902, 0.980, 0.871),
    vec3(0.902, 0.984, 0.886), vec3(0.906, 0.992, 0.902), vec3(0.906, 0.996, 0.918),
    vec3(0.910, 0.996, 0.929), vec3(0.914, 0.996, 0.933), vec3(0.914, 0.996, 0.933),
    vec3(0.918, 0.996, 0.937), vec3(0.922, 0.996, 0.941), vec3(0.925, 0.996, 0.945),
    vec3(0.925, 0.996, 0.949), vec3(0.929, 0.996, 0.949), vec3(0.933, 0.996, 0.953),
    vec3(0.937, 0.996, 0.957), vec3(0.941, 0.996, 0.961), vec3(0.945, 0.996, 0.965),
    vec3(0.949, 0.992, 0.965), vec3(0.953, 0.992, 0.969), vec3(0.957, 0.992, 0.973),
    vec3(0.961, 0.992, 0.973), vec3(0.965, 0.988, 0.976), vec3(0.969, 0.988, 0.980),
    vec3(0.973, 0.988, 0.984), vec3(0.976, 0.984, 0.984), vec3(0.980, 0.984, 0.988),
    vec3(0.984, 0.980, 0.988), vec3(0.988, 0.980, 0.992), vec3(0.988, 0.976, 0.996),
    vec3(0.992, 0.976, 0.996), vec3(0.996, 0.973, 1.000), vec3(0.996, 0.969, 1.000),
    vec3(1.000, 0.965, 1.000), vec3(1.000, 0.961, 1.000), vec3(1.000, 0.957, 1.000),
    vec3(1.000, 0.953, 1.000), vec3(1.000, 0.949, 1.000), vec3(1.000, 0.941, 1.000),
    vec3(0.988, 0.925, 0.988)
);

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= uWidth || pos.y >= uHeight) return;

    float dmin = uintBitsToFloat(globalMin);
    float dmax = uintBitsToFloat(globalMax);
    float range = dmax - dmin;
    if (range < 1.0e-6) range = 1.0;

    float d = depth[pos.y * uWidth + pos.x];
    float norm = clamp((d - dmin) / range, 0.0, 1.0);
    int idx = int(norm * 254.0);  // close=dark(0), far=bright(254)

    vec3 color = infernoLut[idx];
    imageStore(outImage, pos, vec4(color, 1.0));
}
)glsl";

// Fullscreen quad vertex shader
static const char* kQuadVertShader = R"glsl(#version 310 es
layout(location = 0) in vec2 aPos;
out vec2 vTexCoord;
void main() {
    vTexCoord = vec2(aPos.x, 1.0 - aPos.y);  // flip Y
    gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);
}
)glsl";

// Fullscreen quad fragment shader
static const char* kQuadFragShader = R"glsl(#version 310 es
precision mediump float;
in vec2 vTexCoord;
out vec4 fragColor;
uniform sampler2D uTexture;
void main() {
    fragColor = texture(uTexture, vTexCoord);
}
)glsl";

// ---------------------------------------------------------------------------
// GL helper functions
// ---------------------------------------------------------------------------

static void checkGlError(const char* op) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        LOGE("GL error after %s: 0x%x", op, err);
    }
}

static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        LOGE("Shader compile error: %s", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!vs || !fs) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        LOGE("Program link error: %s", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

static GLuint createComputeProgram(const char* src) {
    GLuint cs = compileShader(GL_COMPUTE_SHADER, src);
    if (!cs) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glDeleteShader(cs);

    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        LOGE("Compute link error: %s", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// ---------------------------------------------------------------------------
// Pipeline state
// ---------------------------------------------------------------------------
struct DepthPipeline {
    // LiteRT
    LiteRtApi api;
    LiteRtEnvironment env = nullptr;
    LiteRtModel model = nullptr;
    LiteRtCompiledModel compiledModel = nullptr;
    LiteRtTensorBuffer inputTensorBuffer = nullptr;
    LiteRtTensorBuffer outputTensorBuffer = nullptr;
    bool useGlBuffers = false;  // true if SSBO zero-copy succeeded

    // Model dimensions
    int inputW = 518, inputH = 518;
    int outputW = 518, outputH = 518;

    // GL resources
    GLuint inputSsbo = 0;     // input floats [H*W*3]
    GLuint outputSsbo = 0;    // output floats [H*W]
    GLuint minMaxSsbo = 0;    // 2 uints: [min, max] for normalization
    GLuint depthTexture = 0;  // colormapped output RGBA
    GLuint minMaxProgram = 0;
    GLuint colormapProgram = 0;
    GLuint quadProgram = 0;
    GLuint quadVao = 0;
    GLuint quadVbo = 0;

    // Uniforms
    GLint mmCount = -1;  // minmax shader
    GLint cmWidth = -1, cmHeight = -1;  // colormap shader

    // EGL
    EGLDisplay eglDisplay = EGL_NO_DISPLAY;
    EGLContext eglContextLiteRt = EGL_NO_CONTEXT;

    // CPU buffers
    std::vector<float> inputFloats;   // preprocessed input (written by GL thread)
    std::vector<float> outputFloats;  // inference output (written by inference thread)
    std::vector<float> renderFloats;  // copy for GL thread to upload
    std::vector<uint8_t> modelData;

    // Thread sync
    std::mutex mtx;
    bool hasInputReady = false;    // new preprocessed input available
    bool hasOutputReady = false;   // new inference output available
    bool inferenceRunning = false; // inference thread is busy

    // ImageNet normalization constants
    static constexpr float MEAN[] = {0.485f, 0.456f, 0.406f};
    static constexpr float STD[]  = {0.229f, 0.224f, 0.225f};

    bool initialized = false;
    volatile bool inferenceReady = false;
};
constexpr float DepthPipeline::MEAN[];
constexpr float DepthPipeline::STD[];

static DepthPipeline g_pipeline;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

static bool initLiteRT(const uint8_t* modelData, size_t modelSize) {
    auto& p = g_pipeline;
    auto& api = p.api;

    if (!api.load()) return false;

    // Pass OUR EGL display+context so LiteRT shares it (no separate context)
    EGLDisplay curDisplay = eglGetCurrentDisplay();
    EGLContext curContext = eglGetCurrentContext();
    LOGI("Passing EGL context to LiteRT: display=%p ctx=%p", curDisplay, curContext);

    LiteRtEnvOption envOpts[2];
    envOpts[0].tag = kLiteRtEnvOptionTagEglDisplay;
    envOpts[0].value.type = kLiteRtAnyTypeVoidPtr;
    envOpts[0].value.ptr_value = (const void*)curDisplay;
    envOpts[1].tag = kLiteRtEnvOptionTagEglContext;
    envOpts[1].value.type = kLiteRtAnyTypeVoidPtr;
    envOpts[1].value.ptr_value = (const void*)curContext;

    int numOpts = (curDisplay != EGL_NO_DISPLAY && curContext != EGL_NO_CONTEXT) ? 2 : 0;

    LiteRtStatus status = api.CreateEnvironment(numOpts, numOpts ? envOpts : nullptr, &p.env);
    if (status != kLiteRtStatusOk) {
        LOGE("CreateEnvironment failed: %d", status);
        return false;
    }
    LOGI("CreateEnvironment: status=%d (with %d EGL options)", status, numOpts);

    // GPU environment — LiteRT uses our shared context, no new context created
    if (api.GpuEnvironmentCreate) {
        status = api.GpuEnvironmentCreate(p.env, 0, nullptr);
        LOGI("GpuEnvironmentCreate: status=%d", status);
    }

    // Load model from buffer
    status = api.CreateModelFromBuffer(modelData, modelSize, &p.model);
    if (status != kLiteRtStatusOk) {
        LOGE("CreateModelFromBuffer failed: %d", status);
        return false;
    }

    // Create compilation options — GPU accelerator on GL thread (FP32 via GLSurfaceView context)
    LiteRtOptions options = nullptr;
    if (api.CreateOptions) {
        api.CreateOptions(&options);
        if (options && api.SetOptionsHwAccelerators) {
            api.SetOptionsHwAccelerators(options, kLiteRtHwAcceleratorGpu);
        }
    }

    // Create compiled model
    status = api.CreateCompiledModel(p.env, p.model, options, &p.compiledModel);
    if (options && api.DestroyOptions) api.DestroyOptions(options);
    if (status != kLiteRtStatusOk) {
        LOGE("CreateCompiledModel failed: %d", status);
        return false;
    }

    LOGI("LiteRT model compiled successfully on GPU");

    // Create tensor buffers from compiled model's requirements
    LiteRtTensorBufferRequirements inReqs = nullptr, outReqs = nullptr;
    status = api.GetInputBufferRequirements(p.compiledModel, 0, 0, &inReqs);
    LOGI("GetInputBufferRequirements: status=%d", status);
    status = api.GetOutputBufferRequirements(p.compiledModel, 0, 0, &outReqs);
    LOGI("GetOutputBufferRequirements: status=%d", status);

    LiteRtRankedTensorType inputType{};
    inputType.element_type = kLiteRtElementTypeFloat32;
    inputType.layout.rank = 4;
    inputType.layout.dimensions[0] = 1;
    inputType.layout.dimensions[1] = p.inputH;
    inputType.layout.dimensions[2] = p.inputW;
    inputType.layout.dimensions[3] = 3;

    LiteRtRankedTensorType outputType{};
    outputType.element_type = kLiteRtElementTypeFloat32;
    outputType.layout.rank = 4;
    outputType.layout.dimensions[0] = 1;
    outputType.layout.dimensions[1] = p.outputH;
    outputType.layout.dimensions[2] = p.outputW;
    outputType.layout.dimensions[3] = 1;

    if (inReqs && outReqs && api.CreateManagedTensorBufferFromRequirements) {
        status = api.CreateManagedTensorBufferFromRequirements(
            p.env, &inputType, inReqs, &p.inputTensorBuffer);
        LOGI("CreateInputBuffer: status=%d", status);
        status = api.CreateManagedTensorBufferFromRequirements(
            p.env, &outputType, outReqs, &p.outputTensorBuffer);
        LOGI("CreateOutputBuffer: status=%d", status);
    } else {
        LOGE("Cannot create tensor buffers");
        return false;
    }

    return true;
}

static bool initGlResources() {
    auto& p = g_pipeline;

    size_t inputSize = p.inputW * p.inputH * 3 * sizeof(float);
    size_t outputSize = p.outputW * p.outputH * sizeof(float);

    // Create SSBOs
    glGenBuffers(1, &p.inputSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, p.inputSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, inputSize, nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &p.outputSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, p.outputSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, outputSize, nullptr, GL_DYNAMIC_DRAW);

    // Min/max SSBO: 2 uints [globalMin, globalMax]
    glGenBuffers(1, &p.minMaxSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, p.minMaxSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(uint32_t), nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    checkGlError("SSBO creation");

    // Tensor buffers are created in initLiteRT (on bg thread)
    p.useGlBuffers = false;

    // Pre-allocate CPU buffers
    p.inputFloats.resize(p.inputW * p.inputH * 3);
    p.outputFloats.resize(p.outputW * p.outputH);
    p.renderFloats.resize(p.outputW * p.outputH);

    // Create depth output texture
    glGenTextures(1, &p.depthTexture);
    glBindTexture(GL_TEXTURE_2D, p.depthTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, p.outputW, p.outputH);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    checkGlError("depth texture");

    // Compile shaders
    p.minMaxProgram = createComputeProgram(kMinMaxComputeShader);
    if (p.minMaxProgram) {
        p.mmCount = glGetUniformLocation(p.minMaxProgram, "uCount");
    }

    p.colormapProgram = createComputeProgram(kColormapComputeShader);
    if (p.colormapProgram) {
        p.cmWidth  = glGetUniformLocation(p.colormapProgram, "uWidth");
        p.cmHeight = glGetUniformLocation(p.colormapProgram, "uHeight");
    }

    p.quadProgram = createProgram(kQuadVertShader, kQuadFragShader);

    // Fullscreen quad VBO + VAO
    float quadVerts[] = {
        0.0f, 0.0f,  // bottom-left
        1.0f, 0.0f,  // bottom-right
        0.0f, 1.0f,  // top-left
        1.0f, 1.0f,  // top-right
    };
    glGenBuffers(1, &p.quadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, p.quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

    glGenVertexArrays(1, &p.quadVao);
    glBindVertexArray(p.quadVao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkGlError("GL resources init");
    LOGI("GL resources initialized (SSBO zero-copy: %s)",
         p.useGlBuffers ? "yes" : "no");

    return p.minMaxProgram && p.colormapProgram && p.quadProgram;
}

// ---------------------------------------------------------------------------
// Frame processing
// ---------------------------------------------------------------------------

// Preprocess camera RGBA pixels → normalized float array (NHWC)
static void preprocessFrame(const uint32_t* pixels, int srcW, int srcH,
                            int rotation) {
    auto& p = g_pipeline;
    float* out = p.inputFloats.data();

    // Simple nearest-neighbor resize with rotation
    // For production, bilinear would be better but NN is faster for preprocess
    for (int y = 0; y < p.inputH; y++) {
        for (int x = 0; x < p.inputW; x++) {
            int sx, sy;
            switch (rotation) {
                case 90:
                    sx = y * srcW / p.inputH;
                    sy = (p.inputW - 1 - x) * srcH / p.inputW;
                    break;
                case 180:
                    sx = (p.inputW - 1 - x) * srcW / p.inputW;
                    sy = (p.inputH - 1 - y) * srcH / p.inputH;
                    break;
                case 270:
                    sx = (p.inputH - 1 - y) * srcW / p.inputH;
                    sy = x * srcH / p.inputW;
                    break;
                default: // 0
                    sx = x * srcW / p.inputW;
                    sy = y * srcH / p.inputH;
                    break;
            }
            sx = sx < srcW ? sx : srcW - 1;
            sy = sy < srcH ? sy : srcH - 1;

            uint32_t argb = pixels[sy * srcW + sx];
            // getPixels() returns ARGB_8888: 0xAARRGGBB
            float r = ((argb >> 16) & 0xFF) / 255.0f;
            float g = ((argb >> 8) & 0xFF) / 255.0f;
            float b = ((argb >> 0) & 0xFF) / 255.0f;

            int idx = (y * p.inputW + x) * 3;
            out[idx + 0] = (r - DepthPipeline::MEAN[0]) / DepthPipeline::STD[0];
            out[idx + 1] = (g - DepthPipeline::MEAN[1]) / DepthPipeline::STD[1];
            out[idx + 2] = (b - DepthPipeline::MEAN[2]) / DepthPipeline::STD[2];
        }
    }
}

static int g_frameCount = 0;

// Inference loop — runs on SEPARATE thread, NEVER touches GL/EGL
static void inferenceLoop() {
    auto& p = g_pipeline;
    LOGI("Inference thread started");

    while (p.initialized) {
        bool hasInput = false;
        {
            std::lock_guard<std::mutex> lock(p.mtx);
            hasInput = p.hasInputReady;
        }
        if (!hasInput) {
        static int pollLog = 0;
        if (pollLog++ % 1000 == 0 && pollLog < 5000)
            LOGI("inferenceLoop: polling (no input), count=%d", pollLog);
        usleep(1000);
        continue;
    }

        if (!p.inputTensorBuffer || !p.outputTensorBuffer ||
            !p.api.LockTensorBuffer) { usleep(10000); continue; }

        // Copy input under lock
        void* inPtr = nullptr;
        LiteRtStatus status = p.api.LockTensorBuffer(p.inputTensorBuffer, &inPtr,
                kLiteRtTensorBufferLockModeWriteReplace);
        if (status != kLiteRtStatusOk || !inPtr) continue;
        {
            std::lock_guard<std::mutex> lock(p.mtx);
            memcpy(inPtr, p.inputFloats.data(), p.inputFloats.size() * sizeof(float));
            p.hasInputReady = false;
        }
        p.api.UnlockTensorBuffer(p.inputTensorBuffer);

        // Inference (no GL context on this thread — safe)
        status = p.api.RunCompiledModel(
            p.compiledModel, 0,
            1, &p.inputTensorBuffer, 1, &p.outputTensorBuffer);
        if (g_frameCount < 5) LOGI("Inference status=%d", status);
        if (status != kLiteRtStatusOk) continue;

        // Copy output under lock
        void* outPtr = nullptr;
        status = p.api.LockTensorBuffer(p.outputTensorBuffer, &outPtr,
                kLiteRtTensorBufferLockModeRead);
        if (status == kLiteRtStatusOk && outPtr) {
            std::lock_guard<std::mutex> lock(p.mtx);
            memcpy(p.outputFloats.data(), outPtr,
                   p.outputW * p.outputH * sizeof(float));
            p.hasOutputReady = true;
            p.api.UnlockTensorBuffer(p.outputTensorBuffer);
            if (g_frameCount < 5) {
                float* f = (float*)outPtr;
                LOGI("Output[0..2]=%f %f %f", f[0], f[1], f[2]);
            }
        }
        g_frameCount++;
    }
    LOGI("Inference thread stopped");
}

// Pre-baked Inferno LUT (same as Kotlin Colormap.kt)
static const uint8_t INFERNO_R[] = {0,0,1,1,2,2,3,4,5,6,7,8,9,10,12,13,15,16,18,19,21,23,24,26,28,30,32,34,36,38,40,42,44,47,49,51,53,56,58,60,62,65,67,69,72,74,76,78,81,83,85,88,90,92,94,96,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,128,130,132,134,135,137,139,140,142,144,145,147,148,150,151,153,154,156,157,158,160,161,162,163,165,166,167,168,169,170,172,173,174,175,176,177,178,179,179,180,181,182,183,184,185,186,187,187,188,189,190,191,191,192,193,194,195,195,196,197,198,198,199,200,200,201,202,202,203,204,204,205,205,206,207,207,208,208,209,209,210,210,211,211,211,212,212,213,213,213,214,214,214,215,215,215,215,216,216,216,216,217,217,217,217,218,218,218,218,218,219,219,219,219,219,220,220,220,220,220,221,221,221,221,221,222,222,222,222,223,223,223,223,224,224,224,225,225,225,226,226,227,227,228,228,229,229,230,230,231,231,232,233,233,234,235,236,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,252,253,254,254,255,255,255,255,255,255,252};
static const uint8_t INFERNO_G[] = {0,0,1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,10,10,10,11,11,11,12,12,13,13,14,14,14,15,16,16,17,17,18,18,19,20,20,21,22,22,23,24,24,25,26,27,28,29,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,51,52,53,54,56,57,58,60,61,62,64,65,67,68,70,71,72,74,75,77,79,80,82,83,85,86,88,90,91,93,94,96,98,99,101,103,104,106,108,109,111,113,114,116,118,120,121,123,125,126,128,130,132,133,135,137,138,140,142,144,145,147,149,150,152,154,155,157,159,161,162,164,166,167,169,171,172,174,176,177,179,181,183,184,186,188,189,191,193,194,196,198,199,201,203,204,206,208,209,211,213,214,216,218,219,221,223,224,226,228,229,231,232,234,236,237,239,240,242,244,245,247,248,250,251,253,254,254,254,254,254,254,254,254,254,254,254,254,254,253,253,253,253,252,252,252,251,251,250,250,249,249,248,247,246,245,244,243,242,240,236};
static const uint8_t INFERNO_B[] = {4,5,7,9,11,14,16,19,22,24,27,30,33,35,38,41,44,47,49,52,55,57,60,62,65,67,69,71,73,75,77,79,80,82,83,84,85,86,87,87,88,88,89,89,89,89,89,89,88,88,88,87,87,86,86,85,84,84,83,82,82,81,80,79,78,77,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,53,52,51,50,49,49,48,47,46,46,45,45,44,43,43,42,42,41,41,40,40,39,39,39,38,38,38,38,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,39,39,39,40,40,41,41,42,42,43,43,44,45,45,46,47,48,49,49,50,52,53,54,55,56,58,59,61,62,64,65,67,69,71,73,75,77,79,81,83,85,88,90,93,95,98,100,103,106,109,112,115,118,121,124,127,131,134,137,141,144,148,152,155,159,163,167,170,174,178,182,186,190,194,198,202,206,210,214,218,222,226,230,234,237,238,238,239,240,241,242,242,243,244,245,246,246,247,248,248,249,250,251,251,252,252,253,254,254,255,255,255,255,255,255,255,255,252};

static float g_smoothMin = 0.0f, g_smoothMax = 1.0f;
static bool g_hasValidFrame = false;

// CPU colormap: depth floats → RGBA texture (no compute shader needed)
static void updateDepthTexture() {
    auto& p = g_pipeline;

    bool newData = false;
    {
        std::lock_guard<std::mutex> lock(p.mtx);
        if (p.hasOutputReady) {
            p.renderFloats.swap(p.outputFloats);
            p.hasOutputReady = false;
            newData = true;
        }
    }
    if (!newData) return;

    int n = p.outputW * p.outputH;
    float* depth = p.renderFloats.data();

    // Compute 2nd/98th percentile for robust normalization
    float rawMin = depth[0], rawMax = depth[0];
    constexpr int STRIDE = 16;
    int sc = 0;
    float samples[16384];
    for (int i = 0; i < n; i += STRIDE) {
        float v = depth[i];
        if (v < rawMin) rawMin = v;
        if (v > rawMax) rawMax = v;
        if (sc < 16384) samples[sc++] = v;
    }
    if (sc > 100) {
        int lo = sc * 2 / 100, hi = sc * 98 / 100;
        std::nth_element(samples, samples + lo, samples + sc);
        float pMin = samples[lo];
        std::nth_element(samples, samples + hi, samples + sc);
        float pMax = samples[hi];
        if (!g_hasValidFrame) {
            g_smoothMin = pMin; g_smoothMax = pMax; g_hasValidFrame = true;
        } else {
            g_smoothMin = g_smoothMin * 0.7f + pMin * 0.3f;
            g_smoothMax = g_smoothMax * 0.7f + pMax * 0.3f;
        }
    }

    float dmin = g_smoothMin, range = g_smoothMax - g_smoothMin;
    if (range < 1e-6f) range = 1.0f;
    float scale = 254.0f / range;

    // Apply Inferno colormap → RGBA8 pixels
    static std::vector<uint8_t> rgba;
    rgba.resize(n * 4);
    for (int i = 0; i < n; i++) {
        int idx = (int)((depth[i] - dmin) * scale);
        if (idx < 0) idx = 0;
        if (idx > 254) idx = 254;
        rgba[i*4+0] = INFERNO_R[idx];
        rgba[i*4+1] = INFERNO_G[idx];
        rgba[i*4+2] = INFERNO_B[idx];
        rgba[i*4+3] = 255;
    }

    // Upload to GL texture directly
    glBindTexture(GL_TEXTURE_2D, p.depthTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, p.outputW, p.outputH,
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    static int normLog = 0;
    if (normLog++ < 10) {
        LOGI("Norm: raw=[%f,%f] smooth=[%f,%f] range=%f samples=%d",
             rawMin, rawMax, g_smoothMin, g_smoothMax, g_smoothMax - g_smoothMin, sc);
    }
}

static void applyColormapAndRender(int viewW, int viewH) {
    auto& p = g_pipeline;
    int count = p.outputW * p.outputH;

    // Min/max already computed on CPU in uploadOutputToSsbo() — skip GPU reduction

    // Colormap compute shader
    glUseProgram(p.colormapProgram);
    glUniform1i(p.cmWidth, p.outputW);
    glUniform1i(p.cmHeight, p.outputH);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, p.outputSsbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, p.minMaxSsbo);
    glBindImageTexture(0, p.depthTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    glDispatchCompute(
        (p.outputW + 15) / 16,
        (p.outputH + 15) / 16,
        1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
    checkGlError("colormap compute");

    // Render fullscreen quad
    glViewport(0, 0, viewW, viewH);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // opaque black clear
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p.quadProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, p.depthTexture);
    glUniform1i(glGetUniformLocation(p.quadProgram, "uTexture"), 0);
    glBindVertexArray(p.quadVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    checkGlError("quad render");

    // DEBUG: on first few frames, bypass compute shaders and write red to texture
    // to verify the quad rendering pipeline works
    if (g_frameCount < 3) {
        GLenum err = glGetError(); // clear errors
        (void)err;
        LOGI("Render: view=%dx%d tex=%u quad=%u colormap=%u minmax=%u",
             viewW, viewH, p.depthTexture, p.quadProgram,
             p.colormapProgram, p.minMaxProgram);
    }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

static void destroyPipeline() {
    auto& p = g_pipeline;

    if (p.inputTensorBuffer && p.api.DestroyTensorBuffer)
        p.api.DestroyTensorBuffer(p.inputTensorBuffer);
    if (p.outputTensorBuffer && p.api.DestroyTensorBuffer)
        p.api.DestroyTensorBuffer(p.outputTensorBuffer);
    if (p.compiledModel && p.api.DestroyCompiledModel)
        p.api.DestroyCompiledModel(p.compiledModel);
    if (p.model && p.api.DestroyModel)
        p.api.DestroyModel(p.model);
    if (p.env && p.api.DestroyEnvironment)
        p.api.DestroyEnvironment(p.env);

    if (p.eglContextLiteRt != EGL_NO_CONTEXT && p.eglDisplay != EGL_NO_DISPLAY) {
        eglDestroyContext(p.eglDisplay, p.eglContextLiteRt);
    }

    if (p.inputSsbo) glDeleteBuffers(1, &p.inputSsbo);
    if (p.outputSsbo) glDeleteBuffers(1, &p.outputSsbo);
    if (p.minMaxSsbo) glDeleteBuffers(1, &p.minMaxSsbo);
    if (p.depthTexture) glDeleteTextures(1, &p.depthTexture);
    if (p.minMaxProgram) glDeleteProgram(p.minMaxProgram);
    if (p.colormapProgram) glDeleteProgram(p.colormapProgram);
    if (p.quadProgram) glDeleteProgram(p.quadProgram);
    if (p.quadVao) glDeleteVertexArrays(1, &p.quadVao);
    if (p.quadVbo) glDeleteBuffers(1, &p.quadVbo);

    p.api.unload();
    p.initialized = false;
    p.inferenceReady = false;
}

// ---------------------------------------------------------------------------
// JNI exports
// ---------------------------------------------------------------------------

extern "C" {

// Init GL resources only (called from onSurfaceCreated on GL thread)
JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeInitGl(
    JNIEnv* env, jobject thiz,
    jint inputW, jint inputH, jint outputW, jint outputH) {

    auto& p = g_pipeline;
    p.inputW = inputW;
    p.inputH = inputH;
    p.outputW = outputW;
    p.outputH = outputH;

    if (!p.api.load()) {
        LOGE("Failed to load libLiteRt.so");
        return JNI_FALSE;
    }

    if (!initGlResources()) {
        LOGE("Failed to init GL resources");
        return JNI_FALSE;
    }

    LOGI("GL resources ready: %dx%d", inputW, inputH);
    return JNI_TRUE;
}

// Receive native handles from Kotlin CompiledModel (FP32 GPU)
JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeSetHandles(
    JNIEnv* env, jobject thiz,
    jlong envHandle, jlong compiledModelHandle) {

    auto& p = g_pipeline;

    // Cast Kotlin native handles to LiteRT C API types
    p.env = (LiteRtEnvironment)(intptr_t)envHandle;
    p.compiledModel = (LiteRtCompiledModel)(intptr_t)compiledModelHandle;
    LOGI("Using Kotlin handles: env=%p model=%p", p.env, p.compiledModel);

    // Create SSBO tensor buffers using the compiled model's requirements
    LiteRtTensorBufferRequirements inReqs = nullptr, outReqs = nullptr;
    LiteRtStatus s;

    s = p.api.GetInputBufferRequirements(p.compiledModel, 0, 0, &inReqs);
    LOGI("GetInputBufferRequirements: status=%d", s);
    s = p.api.GetOutputBufferRequirements(p.compiledModel, 0, 0, &outReqs);
    LOGI("GetOutputBufferRequirements: status=%d", s);

    // Create managed buffers from requirements (FP32, correct format)
    LiteRtRankedTensorType inputType{};
    inputType.element_type = kLiteRtElementTypeFloat32;
    inputType.layout.rank = 4;
    inputType.layout.dimensions[0] = 1;
    inputType.layout.dimensions[1] = p.inputH;
    inputType.layout.dimensions[2] = p.inputW;
    inputType.layout.dimensions[3] = 3;

    LiteRtRankedTensorType outputType{};
    outputType.element_type = kLiteRtElementTypeFloat32;
    outputType.layout.rank = 4;
    outputType.layout.dimensions[0] = 1;
    outputType.layout.dimensions[1] = p.outputH;
    outputType.layout.dimensions[2] = p.outputW;
    outputType.layout.dimensions[3] = 1;

    if (inReqs && outReqs && p.api.CreateManagedTensorBufferFromRequirements) {
        s = p.api.CreateManagedTensorBufferFromRequirements(
            p.env, &inputType, inReqs, &p.inputTensorBuffer);
        LOGI("CreateInputBuffer: status=%d", s);
        s = p.api.CreateManagedTensorBufferFromRequirements(
            p.env, &outputType, outReqs, &p.outputTensorBuffer);
        LOGI("CreateOutputBuffer: status=%d", s);
    }

    if (!p.inputTensorBuffer || !p.outputTensorBuffer) {
        LOGE("Failed to create tensor buffers");
        return JNI_FALSE;
    }

    p.initialized = true;
    p.inferenceReady = true;
    p.useGlBuffers = false;
    LOGI("Pipeline ready with Kotlin FP32 CompiledModel");
    return JNI_TRUE;
}


JNIEXPORT void JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeProcessFrame(
    JNIEnv* env, jobject thiz,
    jintArray pixels, jint width, jint height, jint rotation) {

    auto& p = g_pipeline;
    if (!p.initialized || !p.inferenceReady) return;

    // ALL on GL thread: preprocess → inference → output ready for render
    jint* data = env->GetIntArrayElements(pixels, nullptr);
    preprocessFrame((const uint32_t*)data, width, height, rotation);
    env->ReleaseIntArrayElements(pixels, data, JNI_ABORT);

    if (!p.inputTensorBuffer || !p.outputTensorBuffer ||
        !p.api.LockTensorBuffer) return;

    LiteRtStatus status;
    void* inPtr = nullptr;
    status = p.api.LockTensorBuffer(p.inputTensorBuffer, &inPtr,
            kLiteRtTensorBufferLockModeWriteReplace);
    if (status == kLiteRtStatusOk && inPtr) {
        memcpy(inPtr, p.inputFloats.data(), p.inputFloats.size() * sizeof(float));
        p.api.UnlockTensorBuffer(p.inputTensorBuffer);

        status = p.api.RunCompiledModel(p.compiledModel, 0,
            1, &p.inputTensorBuffer, 1, &p.outputTensorBuffer);

        if (status == kLiteRtStatusOk) {
            void* outPtr = nullptr;
            status = p.api.LockTensorBuffer(p.outputTensorBuffer, &outPtr,
                    kLiteRtTensorBufferLockModeRead);
            if (status == kLiteRtStatusOk && outPtr) {
                memcpy(p.outputFloats.data(), outPtr,
                       p.outputW * p.outputH * sizeof(float));
                p.api.UnlockTensorBuffer(p.outputTensorBuffer);
                p.hasOutputReady = true;
            }
        }
        static int infLog = 0;
        if (infLog++ < 5) LOGI("Inference status=%d", status);
    }
}

JNIEXPORT void JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeRender(
    JNIEnv* env, jobject thiz, jint viewWidth, jint viewHeight) {

    auto& p = g_pipeline;
    if (!p.initialized) return;

    if (!p.inferenceReady) {
        glViewport(0, 0, viewWidth, viewHeight);
        glClearColor(0.0f, 0.0f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        return;
    }

    // CPU colormap → texture upload
    updateDepthTexture();


    // Render fullscreen quad
    glViewport(0, 0, viewWidth, viewHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(g_pipeline.quadProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_pipeline.depthTexture);
    glUniform1i(glGetUniformLocation(g_pipeline.quadProgram, "uTexture"), 0);
    glBindVertexArray(g_pipeline.quadVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

JNIEXPORT void JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeDestroy(
    JNIEnv* env, jobject thiz) {

    destroyPipeline();
}

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeIsInitialized(
    JNIEnv* env, jobject thiz) {

    return g_pipeline.initialized ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NativeDepthPipeline_nativeIsZeroCopy(
    JNIEnv* env, jobject thiz) {

    return g_pipeline.useGlBuffers ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"

// NCNN + Vulkan depth pipeline. No LiteRT, no OpenCV.
#include <jni.h>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <net.h>
#include <gpu.h>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

#define TAG "DepthNCNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static const uint8_t IR[] = {0,0,1,1,2,2,3,4,5,6,7,8,9,10,12,13,15,16,18,19,21,23,24,26,28,30,32,34,36,38,40,42,44,47,49,51,53,56,58,60,62,65,67,69,72,74,76,78,81,83,85,88,90,92,94,96,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,128,130,132,134,135,137,139,140,142,144,145,147,148,150,151,153,154,156,157,158,160,161,162,163,165,166,167,168,169,170,172,173,174,175,176,177,178,179,179,180,181,182,183,184,185,186,187,187,188,189,190,191,191,192,193,194,195,195,196,197,198,198,199,200,200,201,202,202,203,204,204,205,205,206,207,207,208,208,209,209,210,210,211,211,211,212,212,213,213,213,214,214,214,215,215,215,215,216,216,216,216,217,217,217,217,218,218,218,218,218,219,219,219,219,219,220,220,220,220,220,221,221,221,221,221,222,222,222,222,223,223,223,223,224,224,224,225,225,225,226,226,227,227,228,228,229,229,230,230,231,231,232,233,233,234,235,236,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,252,253,254,254,255,255,255,255,255,255,252};
static const uint8_t IG[] = {0,0,1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,10,10,10,11,11,11,12,12,13,13,14,14,14,15,16,16,17,17,18,18,19,20,20,21,22,22,23,24,24,25,26,27,28,29,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,51,52,53,54,56,57,58,60,61,62,64,65,67,68,70,71,72,74,75,77,79,80,82,83,85,86,88,90,91,93,94,96,98,99,101,103,104,106,108,109,111,113,114,116,118,120,121,123,125,126,128,130,132,133,135,137,138,140,142,144,145,147,149,150,152,154,155,157,159,161,162,164,166,167,169,171,172,174,176,177,179,181,183,184,186,188,189,191,193,194,196,198,199,201,203,204,206,208,209,211,213,214,216,218,219,221,223,224,226,228,229,231,232,234,236,237,239,240,242,244,245,247,248,250,251,253,254,254,254,254,254,254,254,254,254,254,254,254,254,253,253,253,253,252,252,252,251,251,250,250,249,249,248,247,246,245,244,243,242,240,236};
static const uint8_t IB[] = {4,5,7,9,11,14,16,19,22,24,27,30,33,35,38,41,44,47,49,52,55,57,60,62,65,67,69,71,73,75,77,79,80,82,83,84,85,86,87,87,88,88,89,89,89,89,89,89,88,88,88,87,87,86,86,85,84,84,83,82,82,81,80,79,78,77,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,53,52,51,50,49,49,48,47,46,46,45,45,44,43,43,42,42,41,41,40,40,39,39,39,38,38,38,38,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,38,38,38,38,39,39,39,40,40,41,41,42,42,43,43,44,45,45,46,47,48,49,49,50,52,53,54,55,56,58,59,61,62,64,65,67,69,71,73,75,77,79,81,83,85,88,90,93,95,98,100,103,106,109,112,115,118,121,124,127,131,134,137,141,144,148,152,155,159,163,167,170,174,178,182,186,190,194,198,202,206,210,214,218,222,226,230,234,237,238,238,239,240,241,242,242,243,244,245,246,246,247,248,248,249,250,251,251,252,252,253,254,254,255,255,255,255,255,255,255,255,252};

static ncnn::Net g_net;
static int g_targetSize = 518;
static bool g_initialized = false;
static bool g_vulkan = false;

static const float MEAN[] = {123.675f, 116.28f, 103.53f};
static const float NORM[] = {1.f/58.395f, 1.f/57.12f, 1.f/57.375f};

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NcnnDepthPipeline_nativeInit(
    JNIEnv* env, jobject, jobject assetMgr,
    jstring paramPath, jstring binPath,
    jint targetSize, jboolean useGpu) {

    g_targetSize = targetSize;
    g_vulkan = useGpu && ncnn::get_gpu_count() > 0;

    g_net.opt.use_vulkan_compute = g_vulkan;
    g_net.opt.use_fp16_packed = true;
    g_net.opt.use_fp16_storage = true;
    g_net.opt.use_fp16_arithmetic = false;  // FP32 arithmetic for ViT
    g_net.opt.num_threads = ncnn::get_big_cpu_count();

    AAssetManager* mgr = AAssetManager_fromJava(env, assetMgr);
    const char* p = env->GetStringUTFChars(paramPath, nullptr);
    const char* b = env->GetStringUTFChars(binPath, nullptr);

    int r1 = g_net.load_param(mgr, p);
    int r2 = g_net.load_model(mgr, b);
    env->ReleaseStringUTFChars(paramPath, p);
    env->ReleaseStringUTFChars(binPath, b);

    if (r1 || r2) { LOGE("Model load failed: param=%d bin=%d", r1, r2); return JNI_FALSE; }

    g_initialized = true;
    LOGI("NCNN init: size=%d vulkan=%d gpus=%d threads=%d",
         g_targetSize, g_vulkan, ncnn::get_gpu_count(), g_net.opt.num_threads);
    return JNI_TRUE;
}

JNIEXPORT jintArray JNICALL
Java_com_depthanything_sample_NcnnDepthPipeline_nativeInfer(
    JNIEnv* env, jobject, jintArray pixels, jint w, jint h, jint rotation) {

    if (!g_initialized) return nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Preprocess: ARGB int[] → ncnn::Mat (RGB, CHW, 0-255)
    jint* data = env->GetIntArrayElements(pixels, nullptr);
    int ts = g_targetSize;

    ncnn::Mat in(ts, ts, 3);
    float* rCh = (float*)in.channel(0);
    float* gCh = (float*)in.channel(1);
    float* bCh = (float*)in.channel(2);

    for (int y = 0; y < ts; y++) {
        for (int x = 0; x < ts; x++) {
            int sx, sy;
            switch (rotation) {
                case 90:  sx = y*w/ts; sy = (ts-1-x)*h/ts; break;
                case 180: sx = (ts-1-x)*w/ts; sy = (ts-1-y)*h/ts; break;
                case 270: sx = (ts-1-y)*w/ts; sy = x*h/ts; break;
                default:  sx = x*w/ts; sy = y*h/ts; break;
            }
            if (sx >= w) sx = w-1; if (sy >= h) sy = h-1;
            uint32_t argb = ((uint32_t*)data)[sy * w + sx];
            int idx = y * ts + x;
            rCh[idx] = (float)((argb >> 16) & 0xFF);
            gCh[idx] = (float)((argb >> 8) & 0xFF);
            bCh[idx] = (float)(argb & 0xFF);
        }
    }
    env->ReleaseIntArrayElements(pixels, data, JNI_ABORT);

    in.substract_mean_normalize(MEAN, NORM);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Inference
    ncnn::Extractor ex = g_net.create_extractor();
    ex.input("image", in);
    ncnn::Mat out;
    ex.extract("depth", out);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Colormap → ARGB int[]
    int n = out.w * out.h;
    const float* depth = (const float*)out.data;
    float dmin = depth[0], dmax = depth[0];
    for (int i = 0; i < n; i++) {
        if (depth[i] < dmin) dmin = depth[i];
        if (depth[i] > dmax) dmax = depth[i];
    }
    float scale = 254.0f / std::max(dmax - dmin, 1e-6f);

    jintArray result = env->NewIntArray(n);
    jint* out_px = env->GetIntArrayElements(result, nullptr);
    for (int i = 0; i < n; i++) {
        int ci = (int)((depth[i] - dmin) * scale);
        if (ci < 0) ci = 0; if (ci > 254) ci = 254;
        out_px[i] = (jint)((0xFF << 24) | (IR[ci] << 16) | (IG[ci] << 8) | IB[ci]);
    }
    env->ReleaseIntArrayElements(result, out_px, 0);
    auto t3 = std::chrono::high_resolution_clock::now();

    long preMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
    long infMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    long postMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
    static int lc = 0;
    if (lc++ < 30) LOGI("pre=%ldms inf=%ldms post=%ldms total=%ldms",
                        preMs, infMs, postMs, preMs+infMs+postMs);

    return result;
}

JNIEXPORT void JNICALL
Java_com_depthanything_sample_NcnnDepthPipeline_nativeDestroy(JNIEnv*, jobject) {
    g_net.clear(); g_initialized = false;
}

JNIEXPORT jboolean JNICALL
Java_com_depthanything_sample_NcnnDepthPipeline_nativeIsVulkan(JNIEnv*, jobject) {
    return g_vulkan ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"

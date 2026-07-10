plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.rfdetr"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.rfdetr"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += setOf("arm64-v8a") }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging {
        jniLibs {
            pickFirsts += setOf("**/libc++_shared.so","**/libtensorflowlite_jni.so","**/libtensorflowlite_gpu_jni.so")
        }
    }
    androidResources { noCompress += listOf("tflite","txt") }
}
dependencies {
    implementation("com.google.ai.edge.litert:litert:2.1.5")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.camera:camera-core:1.4.1")
    implementation("androidx.camera:camera-camera2:1.4.1")
    implementation("androidx.camera:camera-lifecycle:1.4.1")
}

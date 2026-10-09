// Audio8-TTS-Preview-0.6b on Android: type a sentence, pick a voice, tap Speak. The model repository's Python host loop
// (audio8_tts_litert.py) ported to Kotlin on the LiteRT CompiledModel API: slow AR + fast AR on the CPU (XNNPACK,
// 4 threads), codec decoder on the GPU (OpenCL) when its output matches the CPU int8 decoder, else the CPU int8
// decoder, codec encoder created for a voice recording and closed right after. Model files are pushed to the app's
// external files dir; the APK carries no model.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.audio8tts"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.audio8tts"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += setOf("arm64-v8a") }
        buildConfigField("String", "LITERT_VERSION", "\"${libs.versions.litert.get()}\"")
        // Device check (app/src/androidTest): Audio8DeviceCheck, see README.
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures { buildConfig = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs { useLegacyPackaging = true }   // extract the .so files: the GPU accelerator is dlopen'ed by path
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    // Native libLiteRt.so + libLiteRtClGlAccelerator.so; brings litert-api (CompiledModel, TensorBuffer).
    implementation(libs.litert)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.ext.junit)
}

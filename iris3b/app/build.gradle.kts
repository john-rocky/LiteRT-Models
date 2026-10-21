plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    // The Compose compiler of Kotlin 2.x; keep it at the `kotlin` version of gradle/libs.versions.toml.
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.21"
}

android {
    namespace = "com.iris3b"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.iris3b"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }
    buildTypes {
        release { isMinifyEnabled = false }
    }
    packaging {
        // Extracted .so files: the GPU accelerator (libLiteRtClGlAccelerator.so) is dlopen()ed by name.
        jniLibs { useLegacyPackaging = true }
    }
    androidResources {
        noCompress += "tflite"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
    implementation(libs.litert)
    // The sample UI, at the versions of the zoo's other Compose modules (gliclass, laya, ...).
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.7")
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material")
    debugImplementation("androidx.compose.ui:ui-tooling")
}

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val litertVersion = "2.2.0"

android {
    namespace = "com.reuse"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.reuse"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += setOf("arm64-v8a") }
        buildConfigField("String", "LITERT_VERSION", "\"$litertVersion\"")
    }

    buildFeatures { buildConfig = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // The model graphs in assets/ are memory-mapped by LiteRT: keep them uncompressed.
    androidResources { noCompress += listOf("tflite") }

    packaging {
        // -PnoGpuAccelerator builds an APK without LiteRT's GPU accelerator library, to check on a phone that the
        // app reports a missing GPU accelerator instead of crashing.
        if (project.hasProperty("noGpuAccelerator")) {
            jniLibs { excludes += "**/libLiteRtClGlAccelerator.so" }
        }
    }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:$litertVersion")
}

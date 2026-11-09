plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.idol"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.idol"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        buildConfigField("String", "LITERT_VERSION", "\"${libs.versions.litert.get()}\"")
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // sample convenience: installable without a release keystore
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    androidResources {
        // graphs are never assets here (they are installed in files/models/), but keep any .tflite mmappable
        noCompress += "tflite"
    }

    packaging {
        // The GPU accelerator library (litert-gpu) is loaded from the extracted native library directory, as in the
        // zoo's other GPU samples on the same phones.
        jniLibs { useLegacyPackaging = true }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        buildConfig = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        // LiteRT 2.3.0's Kotlin API carries Kotlin 2.4 metadata; Google's own 2.3.0 samples on Kotlin 2.2.21 pass this
        freeCompilerArgs.add("-Xskip-metadata-version-check")
    }
}

dependencies {
    // LiteRT CompiledModel (CPU / XNNPACK for BiRefNet) and its GPU accelerator (ML Drift, OpenCL) for the rest
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.ktx)
    // WebViewAssetLoader: the splat viewer and the .ply served to the WebView from the app itself
    implementation(libs.androidx.webkit)
}

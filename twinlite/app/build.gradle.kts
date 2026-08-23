plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.twinlite"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.twinlite"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += setOf("arm64-v8a")
        }
    }

    buildFeatures { buildConfig = true }

    // Two APKs, not two modes in one: cold start is the thing being filmed, and a shared
    // LiteRT Environment would let whichever ran first fix the dispatch options.
    flavorDimensions += "accel"
    productFlavors {
        create("gpu") {
            dimension = "accel"
            buildConfigField("boolean", "USE_NPU", "false")
        }
        create("npu") {
            dimension = "accel"
            applicationIdSuffix = ".npu"
            buildConfigField("boolean", "USE_NPU", "true")
            // The Hexagon skel is opened by the DSP through a real path, so the libs must
            // be extracted rather than left inside the APK.
            packaging { jniLibs { useLegacyPackaging = true } }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    packaging {
        jniLibs {
            pickFirsts += setOf(
                "**/libc++_shared.so",
                "**/libtensorflowlite_jni.so",
                "**/libtensorflowlite_gpu_jni.so"
            )
        }
    }

    aaptOptions {
        noCompress += listOf("tflite")
    }
}

dependencies {
    // LiteRT (CompiledModel API)
    implementation("com.google.ai.edge.litert:litert:2.2.0")

    // CameraX
    val cameraVersion = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraVersion")
    implementation("androidx.camera:camera-camera2:$cameraVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraVersion")
    implementation("androidx.camera:camera-view:$cameraVersion")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
}

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.litertzoo.npubench"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.litertzoo.npubench"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += setOf("arm64-v8a")
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
            // The Hexagon skel is opened by the DSP through ADSP_LIBRARY_PATH, which
            // needs real files on disk. Without legacy packaging nativeLibraryDir
            // points inside the APK and the DSP finds nothing.
            useLegacyPackaging = true
        }
    }

    aaptOptions {
        noCompress += listOf("tflite")
    }
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:2.2.0")
    implementation("androidx.core:core-ktx:1.15.0")

    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

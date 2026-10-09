plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// One source for the dependency and the version the app shows on screen.
val litertVersion: String = providers.gradleProperty("litertVersion").get()

android {
    namespace = "com.bonsai.imagegen"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.bonsai.imagegen"
        minSdk = 26
        targetSdk = 35
        versionCode = 4
        versionName = "1.3"
        buildConfigField("String", "LITERT_VERSION", "\"$litertVersion\"")
        // BonsaiDeviceCheck: ./gradlew :app:connectedDebugAndroidTest, or am instrument (README)
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += setOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
            // sample convenience: installable without a release keystore
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    lint {
        // lintVitalAnalyzeRelease crashes in AGP 8.7.3 on this project (lint
        // -internal NPE); unit tests + on-device runs are the real gate here
        checkReleaseBuilds = false
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
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
}

dependencies {
    // LiteRT CompiledModel on the CPU (XNNPACK, thread count from CpuOptions)
    implementation("com.google.ai.edge.litert:litert:$litertVersion")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")

    // Host-JVM tests for the tokenizer / math ports (golden + fixture parity)
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")

    // On-device check of the screen and the pipeline (BonsaiDeviceCheck)
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

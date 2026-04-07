plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.kokoro"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.kokoro"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

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

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        noCompress += listOf("onnx", "bin", "json", "gz", "txt")
    }

    packaging {
        resources {
            pickFirsts += setOf(
                "META-INF/CONTRIBUTORS.md",
                "META-INF/LICENSE.md",
                "META-INF/NOTICE.md",
            )
        }
    }
}

dependencies {
    // ONNX Runtime — Kokoro StyleTTS2 (single graph)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")

    // Japanese morphological analyzer for runtime phonemization
    implementation("com.atilika.kuromoji:kuromoji-ipadic:0.9.0")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
}

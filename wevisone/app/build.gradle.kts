plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// Directory of the conversion run's reference data for the full parity tests (scripts/TEST_DATA.md); the tests that
// need it are skipped without it, the ones on the committed test resources always run.
val wevisoneGolden = providers.gradleProperty("wevisone.golden")
    .orElse(providers.systemProperty("wevisone.golden"))

android {
    namespace = "com.wevisone"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.wevisone"
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
    testOptions {
        unitTests.all {
            wevisoneGolden.orNull?.takeIf { it.isNotBlank() }?.let { directory ->
                it.systemProperty("wevisone.golden", rootProject.file(directory).absolutePath)
            }
            it.systemProperty("wevisone.moduleRoot", rootProject.projectDir.absolutePath)
            it.maxHeapSize = "3g"
            it.testLogging {
                events("passed", "skipped", "failed")
                showStandardStreams = true
            }
        }
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
    // LiteRT CompiledModel (CPU / XNNPACK) and its GPU accelerator (ML Drift, OpenCL)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.ktx)
    // Streaming reader for the 11.6 MB tokenizer.json, the same class on the phone and in the JVM tests
    implementation(libs.gson)
    testImplementation(libs.junit)
    // the real org.json on the JVM (the Android jar's copy is a stub in unit tests)
    testImplementation(libs.org.json)
}

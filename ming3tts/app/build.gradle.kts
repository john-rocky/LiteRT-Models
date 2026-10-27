plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// JVM unit tests read the generated host assets and expected vectors from here
// (the conversion's verification scripts write them).
val vectorsOut = providers.environmentVariable("MING3TTS_OUT")
    .orElse(rootProject.file("out").path)

// -Pming3tts.noNpuLibs builds the APK without the Qualcomm vendor libraries (the app then runs the
// NPU graphs on the GPU and says so: one of the device failure paths).
val noNpuLibs = providers.gradleProperty("ming3tts.noNpuLibs").isPresent

android {
    namespace = "com.ming3tts"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.ming3tts"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += setOf("arm64-v8a") }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            // Same key as debug, so a release build installs over the debug build
            // that received the model files (run-as needs a debuggable app).
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    packaging {
        jniLibs {
            // The Qualcomm vendor libraries in src/main/jniLibs/arm64-v8a (not in git: copy them in before
            // the build; Galaxy S26 = SM8850 = Hexagon v81): libLiteRtDispatch_Qualcomm.so and
            // libLiteRtCompilerPlugin_Qualcomm.so (LiteRT's NPU runtime libraries for JIT), and from QAIRT
            // 2.47: libQnnHtp.so, libQnnSystem.so, libQnnHtpPrepare.so, libQnnIr.so, libQnnSaver.so,
            // libQnnHtpV81Stub.so, libQnnHtpV81CalculatorStub.so (lib/aarch64-android) and
            // libQnnHtpV81Skel.so (lib/hexagon-v81/unsigned). They must be real files in
            // nativeLibraryDir: the Hexagon DSP opens its skel library from there.
            useLegacyPackaging = true
            if (noNpuLibs) {
                excludes += "**/libQnn*.so"
                excludes += "**/libLiteRt*_Qualcomm.so"
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    testOptions {
        // android.util.Log in the host code returns defaults on the JVM
        unitTests.isReturnDefaultValues = true
        unitTests.all {
            it.systemProperty("ming3tts.out", vectorsOut.get())
            it.systemProperty("ming3tts.contract", rootProject.file("contract.json").path)
            providers.environmentVariable("MING3TTS_RECORD").orNull?.let { r -> it.systemProperty("ming3tts.record", r) }
            it.maxHeapSize = "3g"
            it.testLogging {
                events("passed", "skipped", "failed")
                showStandardStreams = true
            }
        }
    }
}

dependencies {
    // 2.2.0: the version of the Mac checks and the NPU runs (the Mali
    // KV-step FULLY_CONNECTED needs 2.1.5 or later).
    implementation("com.google.ai.edge.litert:litert:2.2.0")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
}

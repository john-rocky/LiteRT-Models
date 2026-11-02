plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.compose.compiler)
}

val debugKeystore = rootProject.file(".local/debug.keystore")
// The model directory (provider/, tables/) and the host check fixtures for the JVM tests; see
// README.md. Without them the parity tests skip.
val modelDir = providers.gradleProperty("d1.model").orElse("")
val fixturesDir = providers.gradleProperty("d1.fixtures").orElse("")
// -Pd1.noNpuLibs builds the APK without Qualcomm's libraries even when they are in jniLibs: the app then
// runs the decision graph on the GPU and says so.
val noNpuLibs = providers.gradleProperty("d1.noNpuLibs").isPresent
val prepareDebugKeystore by tasks.registering(Exec::class) {
  description = "Creates this isolated sample's debug signing key."
  outputs.file(debugKeystore)
  onlyIf { !debugKeystore.exists() }
  doFirst { debugKeystore.parentFile.mkdirs() }
  commandLine(
    File(System.getProperty("java.home"), "bin/keytool").absolutePath,
    "-genkeypair", "-keystore", debugKeystore.absolutePath,
    "-storepass", "android", "-keypass", "android",
    "-alias", "androiddebugkey", "-dname", "CN=Android Debug,O=Android,C=US",
    "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000",
  )
}

tasks.configureEach {
  if (name == "validateSigningDebug") {
    dependsOn(prepareDebugKeystore)
  }
}

android {
  namespace = "com.d1_omni_2b"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.d1_omni_2b"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"
    ndk { abiFilters += setOf("arm64-v8a") }
  }

  buildTypes {
    debug {
      signingConfig = signingConfigs.getByName("debug").apply { storeFile = debugKeystore }
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }

  packaging {
    jniLibs {
      // The GPU accelerator library and Qualcomm's libraries are opened as files from the app's
      // library folder, so they are stored uncompressed and extracted at install.
      // For the Galaxy S26's NPU, copy into src/main/jniLibs/arm64-v8a (not in git; see README.md):
      // libLiteRtDispatch_Qualcomm.so (LiteRT's NPU runtime libraries, qualcomm_runtime_v81) and, from
      // QAIRT 2.47.0, libQnnHtp.so, libQnnSystem.so, libQnnHtpV81Stub.so, libQnnHtpV81CalculatorStub.so
      // (lib/aarch64-android) and libQnnHtpV81Skel.so (lib/hexagon-v81/unsigned). They must be real files,
      // not links: the Hexagon DSP opens its skel library from the app's library folder.
      useLegacyPackaging = true
      pickFirsts +=
        setOf(
          "**/libc++_shared.so",
          "**/libtensorflowlite_jni.so",
          "**/libtensorflowlite_gpu_jni.so",
        )
      if (noNpuLibs) {
        excludes += "**/libQnn*.so"
        excludes += "**/libLiteRt*_Qualcomm.so"
      }
    }
  }

  androidResources { noCompress += listOf("json") }

  testOptions {
    unitTests.all {
      it.systemProperty("d1.model", modelDir.get().let { path -> if (path.isEmpty()) "" else rootProject.file(path).absolutePath })
      it.systemProperty("d1.fixtures", fixturesDir.get().let { path -> if (path.isEmpty()) "" else rootProject.file(path).absolutePath })
      it.maxHeapSize = "4g"
    }
  }
}

kotlin {
  compilerOptions {
    jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
  }
}

dependencies {
  // 2.2.0: the version the graphs were checked with on the phones (GPU and the S26's NPU); its AAR
  // carries the GPU accelerator.
  implementation(libs.litert)
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.ktx)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material2)
  implementation(libs.kotlinx.coroutines.android)
  debugImplementation(libs.androidx.ui.tooling)

  testImplementation(libs.junit)
}

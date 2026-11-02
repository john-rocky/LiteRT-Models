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
  namespace = "com.d1_2b"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.d1_2b"
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
      // The GPU accelerator library is opened as a file.
      useLegacyPackaging = true
      pickFirsts +=
        setOf(
          "**/libc++_shared.so",
          "**/libtensorflowlite_jni.so",
          "**/libtensorflowlite_gpu_jni.so",
        )
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
    // LiteRT 2.3.0's Kotlin metadata is newer than this Kotlin compiler reads by default.
    freeCompilerArgs.add("-Xskip-metadata-version-check")
  }
}

dependencies {
  implementation(libs.litert)
  implementation(libs.litert.gpu)
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

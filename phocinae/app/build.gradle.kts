// SPDX-License-Identifier: Apache-2.0
plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.compose)
}

val debugKeystore = rootProject.file(".local/debug.keystore")
val prepareDebugKeystore by
  tasks.registering(Exec::class) {
    description = "Creates this isolated sample's debug signing key."
    outputs.file(debugKeystore)
    onlyIf { !debugKeystore.exists() }
    doFirst { debugKeystore.parentFile.mkdirs() }
    commandLine(
      File(System.getProperty("java.home"), "bin/keytool").absolutePath,
      "-genkeypair",
      "-keystore",
      debugKeystore.absolutePath,
      "-storepass",
      "android",
      "-keypass",
      "android",
      "-alias",
      "androiddebugkey",
      "-dname",
      "CN=Android Debug,O=Android,C=US",
      "-keyalg",
      "RSA",
      "-keysize",
      "2048",
      "-validity",
      "10000",
    )
  }

tasks.configureEach {
  if (name == "validateSigningDebug" || name == "validateSigningRelease")
    dependsOn(prepareDebugKeystore)
}

android {
  namespace = "com.phocinae"
  compileSdk = 35
  buildToolsVersion = "35.0.0"

  defaultConfig {
    applicationId = "com.phocinae"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "0.1"
    ndk { abiFilters += "arm64-v8a" }
  }

  buildTypes {
    debug { signingConfig = signingConfigs.getByName("debug").apply { storeFile = debugKeystore } }
    release {
      isDebuggable = false
      isMinifyEnabled = false
      signingConfig = signingConfigs.getByName("debug")
    }
  }

  packaging {
    // The Hexagon skel is opened by the DSP through ADSP_LIBRARY_PATH, which needs real files.
    jniLibs { useLegacyPackaging = true }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
  implementation(libs.litert)
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.ktx)
  implementation(platform(libs.compose.bom))
  implementation(libs.compose.ui)
  implementation(libs.compose.ui.tooling.preview)
  implementation(libs.compose.material)
  debugImplementation(libs.compose.ui.tooling)
}

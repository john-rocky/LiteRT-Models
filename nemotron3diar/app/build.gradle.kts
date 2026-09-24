plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

// LiteRT runtime version. 2.2.0 is the version this harness is gated on; pass -PlitertVersion=2.1.6 to
// build the same code against another runtime (the value is recorded in timing.json).
val litertVersion = providers.gradleProperty("litertVersion").getOrElse("2.2.0")

android {
  namespace = "com.nemotron3diar"
  compileSdk = 35
  buildToolsVersion = "35.0.0"

  defaultConfig {
    applicationId = "com.nemotron3diar"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "0.1"
    ndk { abiFilters += "arm64-v8a" }
    buildConfigField("String", "LITERT_VERSION", "\"$litertVersion\"")
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      signingConfig = signingConfigs.getByName("debug")
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  buildFeatures { buildConfig = true }

  androidResources { noCompress += "tflite" }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
  implementation("com.google.ai.edge.litert:litert:$litertVersion")
  implementation("androidx.core:core-ktx:1.15.0")
  // litert-api pulls play-services-basement -> fragment 1.1.0 -> activity 1.0.0; pin the AndroidX versions
  // that sibling LiteRT 2.2.0 apps resolve, so the build works from an offline Gradle cache.
  constraints {
    implementation("androidx.activity:activity:1.10.1")
    implementation("androidx.annotation:annotation:1.9.1")
    implementation("androidx.annotation:annotation-experimental:1.4.1")
    implementation("androidx.collection:collection:1.5.0")
    implementation("androidx.arch.core:core-common:2.2.0")
    implementation("androidx.arch.core:core-runtime:2.2.0")
    implementation("androidx.savedstate:savedstate:1.4.0")
    implementation("androidx.profileinstaller:profileinstaller:1.4.0")
    implementation("androidx.startup:startup-runtime:1.1.1")
    implementation("androidx.tracing:tracing:1.2.0")
    implementation("androidx.concurrent:concurrent-futures:1.1.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
  }
}

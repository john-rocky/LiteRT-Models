// On-device speech recognition with three LiteRT-LM bundles (Qwen3-ASR-1.7B, Fun-ASR-Nano-2512, Confucius4-R2T2):
// one Engine at a time, a new Conversation per clip, the language model and the audio encoder on the CPU.
plugins { id("com.android.application") }

val litertlmVersion: String = providers.gradleProperty("litertlmVersion").get()

android {
  namespace = "com.asrlitertlm"
  compileSdk = 36

  defaultConfig {
    applicationId = "com.asrlitertlm"
    minSdk = 31
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"
    ndk { abiFilters += setOf("arm64-v8a") }
    buildConfigField("String", "LITERTLM_VERSION", "\"$litertlmVersion\"")
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  buildFeatures { buildConfig = true }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
}

dependencies {
  // Brings gson, kotlin-reflect and kotlinx-coroutines-android (its POM).
  implementation("com.google.ai.edge.litertlm:litertlm-android:$litertlmVersion")

  testImplementation("junit:junit:4.13.2")

  // Device check (app/src/androidTest): README "Device check" runs it with am instrument.
  androidTestImplementation("androidx.test:runner:1.6.2")
  androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

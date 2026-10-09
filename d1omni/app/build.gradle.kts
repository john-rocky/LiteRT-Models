plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

val debugKeystore = rootProject.file(".local/debug.keystore")
// The model repository directory (contract.json, tokenizer.json, fixtures/) and the conversion
// run's test data directory; see scripts/TEST_DATA.md. Without them the parity tests skip.
val d1omniRepo = providers.gradleProperty("d1omni.repo")
    .orElse(providers.systemProperty("d1omni.repo"))
val d1omniDemo = providers.gradleProperty("d1omni.demo")
    .orElse(providers.systemProperty("d1omni.demo"))
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
    namespace = "com.d1omni"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.d1omni"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("debug").apply {
                storeFile = debugKeystore
            }
        }
        release {
            isMinifyEnabled = false
        }
    }

    packaging {
        // LiteRT's GPU accelerator library is opened as a file, as in the zoo's other LiteRT samples
        // measured on the Galaxy S26.
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
    testOptions {
        unitTests.all {
            d1omniRepo.orNull?.takeIf { it.isNotBlank() }?.let { directory ->
                it.systemProperty("d1omni.repo", rootProject.file(directory).absolutePath)
            }
            d1omniDemo.orNull?.takeIf { it.isNotBlank() }?.let { directory ->
                it.systemProperty("d1omni.demo", rootProject.file(directory).absolutePath)
            }
            it.systemProperty("d1omni.buildDir", layout.buildDirectory.get().asFile.absolutePath)
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
    }
}

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
    testImplementation(libs.junit)
}

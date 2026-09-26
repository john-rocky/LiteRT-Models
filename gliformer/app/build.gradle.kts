plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

val debugKeystore = rootProject.file(".local/debug.keystore")
val fixtureRoot = providers.gradleProperty("gliformer.fixtures")
    .orElse(providers.systemProperty("gliformer.fixtures"))
val prepareDebugKeystore by tasks.registering(Exec::class) {
    description = "Creates the sample's local debug signing key."
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
    if (name == "validateSigningDebug" || name == "validateSigningBenchmark") dependsOn(prepareDebugKeystore)
}

android {
    namespace = "com.gliformer"
    compileSdk = 35
    buildToolsVersion = "35.0.0"

    defaultConfig {
        applicationId = "com.gliformer"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("debug").apply {
                storeFile = debugKeystore
            }
        }
        release { isMinifyEnabled = false }
        create("benchmark") {
            initWith(getByName("release"))
            isDebuggable = false
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
            matchingFallbacks += "release"
        }
    }

    listOf("debug", "benchmark").forEach { name ->
        sourceSets.getByName(name).java.srcDir("src/measurement/java")
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
            fixtureRoot.orNull?.takeIf { value -> value.isNotBlank() }?.let { directory ->
                it.systemProperty("gliformer.fixtures", rootProject.file(directory).absolutePath)
            }
            it.systemProperty("gliformer.moduleRoot", rootProject.projectDir.absolutePath)
            it.systemProperty("gliformer.buildDir", layout.buildDirectory.get().asFile.absolutePath)
            it.systemProperty("java.io.tmpdir", rootProject.file(".local/tmp").absolutePath)
            it.maxHeapSize = "3g"
            it.testLogging {
                events("passed", "skipped", "failed")
                showStandardStreams = true
            }
        }
    }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
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
    testImplementation(libs.json)
}

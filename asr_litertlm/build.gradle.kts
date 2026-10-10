// AGP 9 has Kotlin built in; do not apply org.jetbrains.kotlin.android.
// litertlm-android 0.17.1 ships Kotlin 2.4 metadata. The Kotlin Gradle plugin that AGP 9.3.1 brings (2.2.10) reads
// metadata up to 2.3 only, so the build classpath gets KGP 2.4.0, which the built-in Kotlin then uses.
buildscript {
  repositories { mavenCentral() }
  dependencies { classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.0") }
}

plugins { id("com.android.application") version "9.3.1" apply false }

pluginManagement {
  repositories {
    google()
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  // The LiteRT-LM AARs live on Google Maven.
  repositories {
    google()
    mavenCentral()
  }
}

rootProject.name = "asr-litertlm"

include(":app")

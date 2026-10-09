pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()   // LiteRT AARs live on Google Maven
        mavenCentral()
    }
}

rootProject.name = "audio8_tts"
include(":app")

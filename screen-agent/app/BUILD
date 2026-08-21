load("@rules_android//rules:rules.bzl", "android_binary")
load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")

kt_android_library(
    name = "app_lib",
    srcs = glob(["src/main/java/**/*.kt"]),
    custom_package = "com.edgeagent.lab",
    manifest = "src/main/AndroidManifest.xml",
    resource_files = glob(["src/main/res/**"]),
    deps = [
        "//sdk:screen_agent",
        "@maven//:androidx_core_core_ktx",
        "@maven//:com_google_ai_edge_litertlm_litertlm_android",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android",
    ],
)

android_binary(
    name = "screen_grounding",
    custom_package = "com.edgeagent.lab",
    manifest = "src/main/AndroidManifest.xml",
    deps = [":app_lib"],
)

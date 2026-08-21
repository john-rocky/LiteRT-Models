load("@rules_kotlin//kotlin:android.bzl", "kt_android_library")

# The screen-agent middle layer: framing, grounding, and the loop.
#
# Deliberately free of MediaProjection and AccessibilityService. Anything that
# can produce a Bitmap satisfies ScreenSource; anything that can synthesise a
# tap satisfies ActionExecutor. That keeps the library testable off-device and
# lets a caller swap either side.
kt_android_library(
    name = "screen_agent",
    srcs = glob(["src/main/java/**/*.kt"]),
    custom_package = "com.edgeagent.sdk",
    manifest = "src/main/AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        "@maven//:com_google_ai_edge_litertlm_litertlm_android",
        "@maven//:org_jetbrains_kotlinx_kotlinx_coroutines_android",
    ],
)

cmake_minimum_required(VERSION 3.22)
project(depth_pipeline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

# LiteRT C++ SDK
add_subdirectory(litert_sdk/litert_cc_sdk litert_cc_sdk_build)

# Our depth pipeline library
add_library(depth_pipeline SHARED depth_pipeline_v2.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_include_directories(depth_pipeline PRIVATE
    litert_sdk/litert_cc_sdk
)

find_library(nativewindow-lib nativewindow)

target_link_libraries(depth_pipeline
    litert_cc_api
    absl::log_internal_check_op
    absl::raw_hash_set
    ${log-lib}
    ${android-lib}
    ${nativewindow-lib}
    EGL
    GLESv3
    dl
    jnigraphics
)

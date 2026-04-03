cmake_minimum_required(VERSION 3.22)
project(depth_pipeline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# NCNN with Vulkan
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/../jni/ncnn/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(depth_pipeline SHARED depth_ncnn.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(depth_pipeline
    ncnn
    ${log-lib}
    ${android-lib}
    jnigraphics
)

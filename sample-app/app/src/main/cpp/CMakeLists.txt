cmake_minimum_required(VERSION 3.22)
project(depth_pipeline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(depth_pipeline SHARED depth_pipeline.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(depth_pipeline
    ${log-lib}
    ${android-lib}
    EGL
    GLESv3
    dl
    jnigraphics
)

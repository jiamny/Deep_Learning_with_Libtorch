
project(26_Yolov3_deepsort)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For profiler on Ubuntu
if (CMAKE_BUILD_TYPE STREQUAL Debug AND NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif ()

# Output directory structure
set(OUTPUT_DIR "result")
set(TARGETS_DIR_NAME "targets")
set(TRAJ_TXT_NAME "trajectories.txt")
set(SNAPSHOTS_DIR_NAME "snapshots")
set(VIDEO_NAME "compressed.flv")

configure_file(
        "${PROJECT_SOURCE_DIR}/config.h.in"
        "${PROJECT_BINARY_DIR}/config.h"
)

# GCC need to link against stdc++fs
if(MSVC)
set(STDCXXFS "")
else()
set(STDCXXFS "stdc++fs")
endif()

add_subdirectory(detection)
add_subdirectory(tracking)
add_subdirectory(processing)

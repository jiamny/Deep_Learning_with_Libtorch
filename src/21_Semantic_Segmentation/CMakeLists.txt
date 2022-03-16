cmake_minimum_required(VERSION 3.20)
project(LibtorchSegmentation)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
)
find_package(OpenCV REQUIRED)
include_directories(include)


include_directories(
   ${PROJECT_SOURCE_DIR}/include
)


FILE(GLOB ALL_SOURCES "*.cpp" "./src/*.cpp"  "./src/architectures/*.cpp"  "./src/backbones/*.cpp"  "./src/utils/*.cpp" )
add_executable(21_LibtorchSegmentation src/main.cpp ${ALL_SOURCES})

target_link_libraries(21_LibtorchSegmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES})
#target_link_libraries(21_LibtorchSegmentation ${CUDA_LIBRARIES} ${OpenCV_LIBS})


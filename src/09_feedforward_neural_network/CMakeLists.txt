cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.6m")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

add_executable(09_feedforward)
target_sources(09_feedforward PRIVATE main.cpp neural_net.h neural_net.cpp)

target_link_libraries(09_feedforward ${TORCH_LIBRARIES})

set_target_properties(09_feedforward PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


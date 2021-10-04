cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(BasicModels)

include_directories(${OpenCV_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

add_executable(05_BasicModels)
target_sources(05_BasicModels PRIVATE main.cpp
                                   BaseModule.h
                                   BaseModule.cpp
                                   CNN.h
                                   CNN.cpp
                                   LSTM.h
                                   LSTM.cpp
                                   MLP.h
                                   MLP.cpp
)

target_link_libraries(05_BasicModels ${OpenCV_LIBS})
target_link_libraries(05_BasicModels "${TORCH_LIBRARIES}")

set_target_properties(05_BasicModels PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


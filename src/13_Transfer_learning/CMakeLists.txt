cmake_minimum_required(VERSION 3.16)

find_package (Eigen3 REQUIRED NO_MODULE)

add_definitions     ( ${EIGEN3_DEFINITIONS} )
#include_directories ( ${EIGEN3_INCLUDE_DIRS} )

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

# tell CMake to search for the static libraries ONLY
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(LIBS ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${requiredlibs} Python3::Python Python3::NumPy) 

set(INCLUDE_DIRS
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# -------------------- add executable ------------------------
include_directories(${INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

add_executable(13_Tf_learning)
target_sources(13_Tf_learning PRIVATE Transfer_learning/main.cpp
                                          Transfer_learning/main.h
)

target_link_libraries(13_Tf_learning ${LIBS})

set_target_properties(13_Tf_learning PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(13_Tf_classify)
target_sources(13_Tf_classify PRIVATE Transfer_learning/classify.cpp)

target_link_libraries(13_Tf_classify ${LIBS})

set_target_properties(13_Tf_classify PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)



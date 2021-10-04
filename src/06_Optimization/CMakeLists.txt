cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.6m")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")


set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)

add_executable(06_TrainPPO TrainPPO.cpp)

target_sources(06_TrainPPO PRIVATE TrainPPO.cpp
                                ProximalPolicyOptimization.h
                                Models.h
                                TestEnvironment.h
)

target_include_directories(06_TrainPPO PRIVATE ${EIGEN3_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(06_TrainPPO ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(06_TrainPPO PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


add_executable(06_TestPPO TestPPO.cpp) 
target_sources(06_TestPPO PRIVATE TestPPO.cpp
                               ProximalPolicyOptimization.h
                               Models.h
                               TestEnvironment.h
)
target_include_directories(06_TestPPO PRIVATE ${EIGEN3_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(06_TestPPO ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(06_TestPPO PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

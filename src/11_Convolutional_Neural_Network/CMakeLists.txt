cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

set(EXECUTABLE_NAME 11_cifar10_classifier)


find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

add_executable(${EXECUTABLE_NAME})

target_sources(${EXECUTABLE_NAME} PRIVATE cifar10_clf/cifar10_classifier.cpp
                                		  cifar10_clf/cifar10.cpp
                                          cifar10_clf/nnet.cpp
                                          cifar10_clf/cifar10.h
                                          cifar10_clf/nnet.h
)


target_include_directories(${EXECUTABLE_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(11_Conv_mnist Conv_mnist.cpp)
target_compile_features(11_Conv_mnist PUBLIC cxx_range_for)
target_link_libraries(11_Conv_mnist ${TORCH_LIBRARIES})

set_target_properties(11_Conv_mnist PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

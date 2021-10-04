cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.6m")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

add_executable(04_Softmax_regression Softmax_regression.cpp)
target_sources(04_Softmax_regression PRIVATE Softmax_regression.cpp)

target_include_directories(04_Softmax_regression PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_Softmax_regression ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_Softmax_regression PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(04_Logistic_reg_mnist Logistic_reg_mnist.cpp)
target_sources(04_Logistic_reg_mnist PRIVATE Logistic_reg_mnist.cpp)

target_include_directories(04_Logistic_reg_mnist PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_Logistic_reg_mnist ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_Logistic_reg_mnist PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

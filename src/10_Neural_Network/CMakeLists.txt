cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.9")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# -----------------------------------------------------------------------------------------
add_executable(10_FC_fashion_mnist)
target_sources(10_FC_fashion_mnist PRIVATE FC_fashion_mnist.cpp)

target_include_directories(10_FC_fashion_mnist PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_FC_fashion_mnist ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_FC_fashion_mnist PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------------------------------
add_executable(10_pytorch_custom_backward)
target_sources(10_pytorch_custom_backward PRIVATE pytorch_custom_backward.cpp)

target_include_directories(10_pytorch_custom_backward PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_pytorch_custom_backward ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_pytorch_custom_backward PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------------------------------
add_executable(10_nnet)
target_sources(10_nnet PRIVATE nnet.cpp)

target_include_directories(10_nnet PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_nnet ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_nnet PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------------------------------
add_executable(10_feedForword)
target_sources(10_feedForword PRIVATE feedForword.cpp)

target_include_directories(10_feedForword PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_feedForword ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_feedForword PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------------------------------
add_executable(10_function_approx)
target_sources(10_function_approx PRIVATE function_approx.cpp)

target_include_directories(10_function_approx PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_function_approx ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_function_approx PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------------------------------
add_executable(10_DefineModel)
target_sources(10_DefineModel PRIVATE DefineModel.cpp)

target_include_directories(10_DefineModel PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_DefineModel ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_DefineModel PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


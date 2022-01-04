cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

add_executable(03_Linear_regression Linear_regression.cpp)
target_sources(03_Linear_regression PRIVATE Linear_regression.cpp)

target_include_directories(03_Linear_regression PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_Linear_regression ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_Linear_regression PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(03_LR_fashion LR_fashion.cpp)

target_sources(03_LR_fashion PRIVATE LR_fashion.cpp
                                fashion.cpp
                                transform.cpp
                                fashion.h
                                transform.h
)

target_include_directories(03_LR_fashion PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_LR_fashion ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_LR_fashion PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(03_LR_reg_BostonHousing LR_reg_BostonHousing.cpp)

target_sources(03_LR_reg_BostonHousing PRIVATE LR_reg_BostonHousing.cpp
                                csvloader.h
)

target_include_directories(03_LR_reg_BostonHousing PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_LR_reg_BostonHousing ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_LR_reg_BostonHousing PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(03_Regression_and_gradient Regression_and_gradient.cpp)
target_sources(03_Regression_and_gradient PRIVATE Regression_and_gradient.cpp)

target_include_directories(03_Regression_and_gradient PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_Regression_and_gradient ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_Regression_and_gradient PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(03_Linear_reg_exp2 Linear_reg_exp2.cpp)
target_sources(03_Linear_reg_exp2 PRIVATE Linear_reg_exp2.cpp)

target_include_directories(03_Linear_reg_exp2 PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_Linear_reg_exp2 ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_Linear_reg_exp2 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


add_executable(03_regression regression.cpp)
target_sources(03_regression PRIVATE regression.cpp)

target_include_directories(03_regression PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_regression ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_regression PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

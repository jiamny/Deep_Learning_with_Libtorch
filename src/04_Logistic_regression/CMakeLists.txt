
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# -----------------------------------------------------------------------------------------
add_executable(04_Softmax_regression)
target_sources(04_Softmax_regression PRIVATE 
../LRdataset.h
../LRdataset.cpp
Softmax_regression.cpp
)

target_link_libraries(04_Softmax_regression ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_Softmax_regression PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(04_Logistic_reg_mnist)
target_sources(04_Logistic_reg_mnist PRIVATE Logistic_reg_mnist.cpp)

target_link_libraries(04_Logistic_reg_mnist ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_Logistic_reg_mnist PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


# -----------------------------------------------------------------------------------------
add_executable(04_Low_level_api_dnn_binary_classifier)
target_sources(04_Low_level_api_dnn_binary_classifier PRIVATE Low_level_api_dnn_binary_classifier.cpp 
															  ../LRdataset.h
															  ../LRdataset.cpp)

target_link_libraries(04_Low_level_api_dnn_binary_classifier ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_Low_level_api_dnn_binary_classifier PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


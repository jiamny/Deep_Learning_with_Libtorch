
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# -----------------------------------------------------------------------------------------
add_executable(10_FC_fashion_mnist)
target_sources(10_FC_fashion_mnist PRIVATE FC_fashion_mnist.cpp)

target_link_libraries(10_FC_fashion_mnist ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(10_FC_fashion_mnist PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(10_pytorch_custom_backward)
target_sources(10_pytorch_custom_backward PRIVATE pytorch_custom_backward.cpp)

target_link_libraries(10_pytorch_custom_backward ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(10_pytorch_custom_backward PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(10_nnet)
target_sources(10_nnet PRIVATE nnet.cpp)

target_link_libraries(10_nnet ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(10_nnet PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(10_feedForword)
target_sources(10_feedForword PRIVATE feedForword.cpp)

target_link_libraries(10_feedForword ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(10_feedForword PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(10_function_approx)
target_sources(10_function_approx PRIVATE function_approx.cpp)

target_link_libraries(10_function_approx ${TORCH_LIBRARIES} ${requiredlibs})

set_target_properties(10_function_approx PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(10_DefineModel)
target_sources(10_DefineModel PRIVATE DefineModel.cpp)

target_link_libraries(10_DefineModel ${TORCH_LIBRARIES} ${requiredlibs} )

set_target_properties(10_DefineModel PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(10_Fashion_template_demo)
target_sources(10_Fashion_template_demo PRIVATE Fashion_template_demo.cpp)

target_link_libraries(10_Fashion_template_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(10_Fashion_template_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


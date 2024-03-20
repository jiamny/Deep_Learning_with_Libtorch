
# -----------------------------------------------------------
add_executable(02_Tensor_Ops)
target_sources(02_Tensor_Ops PRIVATE TensorOps.cpp)

target_link_libraries(02_Tensor_Ops ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(02_Tensor_Ops PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------
add_executable(02_Tensor)
target_sources(02_Tensor PRIVATE Tensor.cpp)

target_link_libraries(02_Tensor ${TORCH_LIBRARIES})

set_target_properties(02_Tensor PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------
add_executable(02_pytorch_tensor)
target_sources(02_pytorch_tensor PRIVATE pytorch_tensor.cpp)

target_link_libraries(02_pytorch_tensor ${TORCH_LIBRARIES})

set_target_properties(02_pytorch_tensor PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------
add_executable(02_pytorch_tensor_2)
target_sources(02_pytorch_tensor_2 PRIVATE pytorch_tensor_2.cpp)

target_link_libraries(02_pytorch_tensor_2 ${TORCH_LIBRARIES})

set_target_properties(02_pytorch_tensor_2 PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------
add_executable(02_Tensor_structure_ops)
target_sources(02_Tensor_structure_ops PRIVATE Tensor_structure_ops.cpp)

target_link_libraries(02_Tensor_structure_ops ${TORCH_LIBRARIES})

set_target_properties(02_Tensor_structure_ops PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------
add_executable(02_Tensor_math_ops)
target_sources(02_Tensor_math_ops PRIVATE Tensor_math_ops.cpp)

target_link_libraries(02_Tensor_math_ops ${TORCH_LIBRARIES})

set_target_properties(02_Tensor_math_ops PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)








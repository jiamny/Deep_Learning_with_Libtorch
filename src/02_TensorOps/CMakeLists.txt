cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

add_executable(02_Tensor_Ops)

target_sources(02_Tensor_Ops PRIVATE TensorOps.cpp)

target_link_libraries(02_Tensor_Ops ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(02_Tensor_Ops PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(02_Tensor Tensor.cpp)
target_sources(02_Tensor PRIVATE Tensor.cpp)

target_link_libraries(02_Tensor ${TORCH_LIBRARIES})

set_target_properties(02_Tensor PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# --- tensor
add_executable(02_pytorch_tensor pytorch_tensor.cpp)
target_sources(02_pytorch_tensor PRIVATE pytorch_tensor.cpp)

target_link_libraries(02_pytorch_tensor ${TORCH_LIBRARIES})

set_target_properties(02_pytorch_tensor PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

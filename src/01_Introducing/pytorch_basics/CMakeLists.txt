cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(Deep_Learning_with_PyTorch_CPP VERSION 1.0.0 LANGUAGES CXX)

# -------------------------------------------------------------
add_executable(01_pytorch_basics)
target_sources(01_pytorch_basics PRIVATE pytorch_basics.cpp)

target_link_libraries(01_pytorch_basics ${TORCH_LIBRARIES})

set_target_properties(01_pytorch_basics PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(01_Autograd)
target_sources(01_Autograd PRIVATE Autograd.cpp)

target_link_libraries(01_Autograd ${TORCH_LIBRARIES})

set_target_properties(01_Autograd PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(01_Autograd2)
target_sources(01_Autograd2 PRIVATE Autograd2.cpp)

target_link_libraries(01_Autograd2 ${TORCH_LIBRARIES})

set_target_properties(01_Autograd2 PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(01_autograd3)
target_sources(01_autograd3 PRIVATE autograd3.cpp)

target_link_libraries(01_autograd3 ${TORCH_LIBRARIES})

set_target_properties(01_autograd3 PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(01_Autograd_mechanics)
target_sources(01_Autograd_mechanics PRIVATE Autograd_mechanics.cpp)

target_link_libraries(01_Autograd_mechanics ${TORCH_LIBRARIES})

set_target_properties(01_Autograd_mechanics PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(01_Dynamic_computation_graphs)
target_sources(01_Dynamic_computation_graphs PRIVATE  Dynamic_computation_graphs.cpp)

target_link_libraries(01_Dynamic_computation_graphs ${TORCH_LIBRARIES})

set_target_properties(01_Dynamic_computation_graphs PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)



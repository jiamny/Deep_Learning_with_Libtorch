cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(Deep_Learning_with_PyTorch_CPP VERSION 1.0.0 LANGUAGES CXX)


add_executable(01_pytorch_overview)
target_sources(01_pytorch_overview PRIVATE pytorch_overview.cpp)

target_link_libraries(01_pytorch_overview ${TORCH_LIBRARIES})

set_target_properties(01_pytorch_overview PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


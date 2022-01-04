cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

add_executable(16_VAE)
target_sources(16_VAE PRIVATE VAE/main.cpp
                              VAE/variational_autoencoder.cpp
                              VAE/variational_autoencoder.h
)

target_link_libraries(16_VAE ${TORCH_LIBRARIES} image-io)

set_target_properties(16_VAE PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

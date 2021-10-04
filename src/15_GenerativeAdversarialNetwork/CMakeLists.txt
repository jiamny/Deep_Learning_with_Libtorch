cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


find_package(Torch REQUIRED)

add_executable(15_dcgan DCGAN/dcgan.cpp)

target_link_libraries(15_dcgan "${TORCH_LIBRARIES}")

set_target_properties(15_dcgan PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

add_executable(15_gan GAN/main.cpp)

target_link_libraries(15_gan ${TORCH_LIBRARIES} image-io)

set_target_properties(15_gan PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/DCGAN-CelebA/include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

add_executable(15_DCGAN_celebA DCGAN-CelebA/src/main.cpp DCGAN-CelebA/src/dataset.cpp)

target_link_libraries(15_DCGAN_celebA ${OpenCV_LIBS})
target_link_libraries(15_DCGAN_celebA "${TORCH_LIBRARIES}")

set_target_properties(15_DCGAN_celebA PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


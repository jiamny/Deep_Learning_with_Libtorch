cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIRS})

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

list(APPEND requiredlibs ${PNG_LIBRARIES})
list(APPEND requiredlibs ${OpenMP_CXX_LIBRARIES})

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

# -------------------------------------------------------------------------------------
add_executable(15_dcgan DCGAN/dcgan.cpp)

target_link_libraries(15_dcgan "${TORCH_LIBRARIES}")

set_target_properties(15_dcgan PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
add_executable(15_gan GAN/main.cpp)

target_link_libraries(15_gan ${TORCH_LIBRARIES} image-io)

set_target_properties(15_gan PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/DCGAN-CelebA/include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

add_executable(15_DCGAN_celebA DCGAN-CelebA/src/main.cpp DCGAN-CelebA/src/dataset.cpp)

target_link_libraries(15_DCGAN_celebA ${OpenCV_LIBS})
target_link_libraries(15_DCGAN_celebA "${TORCH_LIBRARIES}")

set_target_properties(15_DCGAN_celebA PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------
# ----WAE2d_GAN_demo
add_executable(15_WAE2d_GAN_demo)
target_sources(15_WAE2d_GAN_demo PRIVATE  WAE2d_GAN/WAE2d_GAN_demo.cpp
    WAE2d_GAN/networks.cpp
    WAE2d_GAN/networks.hpp
    WAE2d_GAN/loss.hpp
    WAE2d_GAN/loss.cpp
    WAE2d_GAN/sample.cpp
    WAE2d_GAN/synth.cpp
    ../image_tools/losses.hpp
    ../image_tools/losses.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_include_directories(15_WAE2d_GAN_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(15_WAE2d_GAN_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs}  Python3::Python Python3::NumPy)

set_target_properties(15_WAE2d_GAN_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


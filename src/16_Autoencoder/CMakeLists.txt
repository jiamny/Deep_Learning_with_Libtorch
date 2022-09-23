cmake_minimum_required(VERSION 3.18 FATAL_ERROR)


find_package(Python3 COMPONENTS Development NumPy)

find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIRS})

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")

list(APPEND requiredlibs ${PNG_LIBRARIES})
list(APPEND requiredlibs ${OpenMP_CXX_LIBRARIES})

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

# -------------------------------------------------------------------------
add_executable(16_VAE)
target_sources(16_VAE PRIVATE VAE/main.cpp
                              VAE/variational_autoencoder.cpp
                              VAE/variational_autoencoder.h
)

target_link_libraries(16_VAE ${TORCH_LIBRARIES} image-io)

set_target_properties(16_VAE PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# --------------------------------------------------------------------------
# ----AE2d_demo
add_executable(16_AE2d_demo)
target_sources(16_AE2d_demo PRIVATE  AE2d/AE2d_demo.cpp
    AE2d/networks.cpp
    AE2d/networks.hpp
    AE2d/loss.hpp
    AE2d/loss.cpp
    ../image_tools/losses.hpp
    ../image_tools/losses.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_include_directories(16_AE2d_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(16_AE2d_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs}  Python3::Python Python3::NumPy)

set_target_properties(16_AE2d_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------------------
# ----DAE2d_demo
add_executable(16_DAE2d_demo)
target_sources(16_DAE2d_demo PRIVATE  DAE2d/DAE2d_demo.cpp
    DAE2d/networks.cpp
    DAE2d/networks.hpp
    DAE2d/loss.hpp
    DAE2d/loss.cpp
    ../image_tools/losses.hpp
    ../image_tools/losses.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_include_directories(16_DAE2d_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(16_DAE2d_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs}  Python3::Python Python3::NumPy)

set_target_properties(16_DAE2d_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

find_package(PNG REQUIRED)

list(APPEND requiredlibs ${PNG_LIBRARIES})

#----------------------------------------------------------------------------------

add_executable(20_pix2pix_demo)

target_sources(20_pix2pix_demo PRIVATE 
	pix2pix/pix2pix_demo.cpp 
	pix2pix/networks.hpp
	pix2pix/networks.cpp
	pix2pix/loss.hpp
    pix2pix/loss.cpp
    ../image_tools/losses.hpp
    ../image_tools/losses.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)
													
target_include_directories(20_pix2pix_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
target_link_libraries(20_pix2pix_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(20_pix2pix_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(20_U-Net_Regression_demo)

target_sources(20_U-Net_Regression_demo PRIVATE 
	U-Net_Regression/U-Net_Regression_demo.cpp 
	U-Net_Regression/networks.hpp
	U-Net_Regression/networks.cpp
	U-Net_Regression/loss.hpp
    U-Net_Regression/loss.cpp
    ../image_tools/losses.hpp
    ../image_tools/losses.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_include_directories(20_U-Net_Regression_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
target_link_libraries(20_U-Net_Regression_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(20_U-Net_Regression_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

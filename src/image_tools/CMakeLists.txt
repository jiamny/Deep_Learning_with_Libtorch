cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

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

# -------------------------------------------------------------
add_executable(Image_tools_check_tool)

target_sources(Image_tools_check_tool PRIVATE check_tools/check_img_tool.cpp )	

target_include_directories(Image_tools_check_tool PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(Image_tools_check_tool ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(Image_tools_check_tool PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(Image_tools_img_loder_demo)

target_sources(Image_tools_img_loder_demo PRIVATE check_tools/img_loder_demo.cpp 
						dataloader.hpp
						dataloader.cpp
						datasets.hpp
						datasets.cpp
						transforms.hpp
						transforms.cpp
						)
													
target_include_directories(Image_tools_img_loder_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(Image_tools_img_loder_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(Image_tools_img_loder_demo PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------------
add_executable(Image_tools_Discriminator)

target_sources(Image_tools_Discriminator PRIVATE Discriminator/main.cpp
												 Discriminator/options.hpp
												 Discriminator/loss.hpp
												 Discriminator/loss.cpp
												 Discriminator/networks.hpp
												 Discriminator/networks.cpp
												 Discriminator/train.cpp
												 Discriminator/test.cpp
												 Discriminator/valid.cpp
												 dataloader.hpp
												 dataloader.cpp
												 datasets.hpp
												 datasets.cpp
												 losses.hpp
												 losses.cpp
												 transforms.hpp
												 transforms.cpp
)

target_include_directories(Image_tools_Discriminator PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(Image_tools_Discriminator ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(Image_tools_Discriminator PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)



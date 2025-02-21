cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Deep_Learning_with_PyTorch_CPP VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIRS})

set(requiredlibs  "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")

list(APPEND requiredlibs ${PNG_LIBRARIES})
list(APPEND requiredlibs ${OpenMP_CXX_LIBRARIES})

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

# ---------------------------------------------------------------------------------------------
add_executable(01_pytorch_overview)
target_sources(01_pytorch_overview PRIVATE 1_pytorch_overview.cpp)

target_link_libraries(01_pytorch_overview ${TORCH_LIBRARIES})

set_target_properties(01_pytorch_overview PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------------------------
add_executable(01_structure_data_example)
target_sources(01_structure_data_example PRIVATE 2_structure_data_example.cpp
                                                 ../../csvloader.h
                                                 ../../LRdataset.h
                                                 ../../LRdataset.cpp
)

target_link_libraries(01_structure_data_example ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(01_structure_data_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------------------------
add_executable(01_image_data_example)
target_sources(01_image_data_example PRIVATE 3_image_data_example.cpp
    ../../image_tools/dataloader.hpp
	../../image_tools/dataloader.cpp
	../../image_tools/datasets.hpp
	../../image_tools/datasets.cpp
	../../image_tools/transforms.hpp
	../../image_tools/transforms.cpp
)

target_link_libraries( 01_image_data_example ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot )
set_target_properties( 01_image_data_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



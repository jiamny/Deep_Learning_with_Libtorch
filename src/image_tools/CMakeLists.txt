
find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIRS})

set(requiredlibs "pthread")
#list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")

list(APPEND requiredlibs ${PNG_LIBRARIES})
list(APPEND requiredlibs ${OpenMP_CXX_LIBRARIES})

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

# -------------------------------------------------------------
# add executable
add_executable( Image_tools_check_matplotplusplus check_tools/Check_matplotplusplus.cpp)

# link libraries
target_link_libraries( Image_tools_check_matplotplusplus ${TORCH_LIBRARIES} matplot)
set_target_properties( Image_tools_check_matplotplusplus  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
# add executable
add_executable( LoadDisplaySave check_tools/loadDisplaySave.cpp)

# link libraries
target_link_libraries( LoadDisplaySave ${OpenCV_LIBS})
set_target_properties( LoadDisplaySave  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(Image_tools_check_tool)

target_sources(Image_tools_check_tool PRIVATE check_tools/check_img_tool.cpp )	

target_link_libraries(Image_tools_check_tool ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties( Image_tools_check_tool PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES )

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
													
target_link_libraries(Image_tools_img_loder_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties( Image_tools_img_loder_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

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

target_link_libraries(Image_tools_Discriminator ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${requiredlibs})
set_target_properties( Image_tools_Discriminator PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )



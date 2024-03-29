
find_package (Eigen3 REQUIRED NO_MODULE)

add_definitions     ( ${EIGEN3_DEFINITIONS} )
#include_directories ( ${EIGEN3_INCLUDE_DIRS} )

# tell CMake to search for the static libraries ONLY
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(LIBS ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} ) 

set(INCLUDE_DIRS
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
)

# -----------------------------------------------------------------------------------------
add_executable(07_dataset_example)
target_sources(07_dataset_example PRIVATE dataset_example.cpp)

target_include_directories(07_dataset_example PRIVATE ${INCLUDE_DIRS} )
target_link_libraries(07_dataset_example ${LIBS})

set_target_properties(07_dataset_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
include_directories(${OpenCV_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lm -ldl")

add_executable(07_Load_img)
target_sources(07_Load_img PRIVATE Load_img.cpp)

target_link_libraries(07_Load_img ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(07_Load_img PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(07_custom-dataset)
target_sources(07_custom-dataset PRIVATE custom-dataset.cpp)

target_link_libraries(07_custom-dataset ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(07_custom-dataset PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------- cifar10
add_executable(07_cifar10_demo)
target_sources(07_cifar10_demo PRIVATE cifar10_demo.cpp cifar10.h cifar10.cpp)

target_link_libraries(07_cifar10_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(07_cifar10_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------- fashion
add_executable(07_fashion_demo)
target_sources(07_fashion_demo PRIVATE fashion_demo.cpp fashion.h fashion.cpp)

target_link_libraries(07_fashion_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(07_fashion_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------- mnist
add_executable(07_mnist_demo)
target_sources(07_mnist_demo PRIVATE mnist_demo.cpp mnist.h mnist.cpp transform.h transform.cpp)

target_link_libraries(07_mnist_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(07_mnist_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


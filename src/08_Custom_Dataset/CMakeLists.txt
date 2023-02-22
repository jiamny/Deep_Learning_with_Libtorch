find_package (Eigen3 REQUIRED NO_MODULE)

add_definitions     ( ${EIGEN3_DEFINITIONS} )
#include_directories ( ${EIGEN3_INCLUDE_DIRS} )

# tell CMake to search for the static libraries ONLY
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

set(LIBS ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs}) 

set(INCLUDE_DIRS
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(${OpenCV_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lm -ldl")

# -----------------------------------------------------------------------------------------
add_executable(08_LoadByInfoFile)
target_sources(08_LoadByInfoFile PRIVATE LoadByInfoFile.cpp dataSet.h dataSet.cpp)

target_link_libraries(08_LoadByInfoFile ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(08_LoadByInfoFile PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(08_LoadWithSplitRatio)
target_sources(08_LoadWithSplitRatio PRIVATE LoadWithSplitRatio.cpp dataSet.h dataSet.cpp)

target_link_libraries(08_LoadWithSplitRatio ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(08_LoadWithSplitRatio PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# data already splitted in train/test/valid
# -----------------------------------------------------------------------------------------
add_executable(08_LoadInSplitFolder)
target_sources(08_LoadInSplitFolder PRIVATE LoadInSplitFolder.cpp dataSet.h dataSet.cpp)

target_link_libraries(08_LoadInSplitFolder ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(08_LoadInSplitFolder PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# data already in folder
# -----------------------------------------------------------------------------------------
add_executable(08_LoadFromFolder)
target_sources(08_LoadFromFolder PRIVATE LoadFromFolder.cpp dataSet.h dataSet.cpp)

target_link_libraries(08_LoadFromFolder ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(08_LoadFromFolder PROPERTIES  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


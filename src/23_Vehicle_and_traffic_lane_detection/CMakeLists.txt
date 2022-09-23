

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# --------------------------------------------------------------------------------
add_executable(23_Lane_detection_demo
Lane_detection_demo.cpp 
LaneDetector.hpp
LaneDetector.cpp
)

target_include_directories(23_Lane_detection_demo  PRIVATE ${ALE_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(23_Lane_detection_demo  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(23_Lane_detection_demo  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


# --------------------------------------------------------------------------------
add_executable(23_Vehicle_and_traffic_lane_detection 
Vehicle_and_traffic_lane_detection.cpp 
LaneDetector.hpp
LaneDetector.cpp
yolo5.h
yolo5.cpp
)

target_include_directories(23_Vehicle_and_traffic_lane_detection  PRIVATE ${ALE_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(23_Vehicle_and_traffic_lane_detection  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(23_Vehicle_and_traffic_lane_detection  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

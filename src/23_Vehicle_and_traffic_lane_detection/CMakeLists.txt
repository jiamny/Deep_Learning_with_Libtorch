

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# --------------------------------------------------------------------------------
add_executable(23_Lane_detection_demo
Lane_detection_demo.cpp 
LaneDetector.hpp
LaneDetector.cpp
)

target_link_libraries(23_Lane_detection_demo  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(23_Lane_detection_demo  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------------------------
add_executable(23_Vehicle_and_traffic_lane_detection 
Vehicle_and_traffic_lane_detection.cpp 
LaneDetector.hpp
LaneDetector.cpp
yolo5.h
yolo5.cpp
)

target_link_libraries(23_Vehicle_and_traffic_lane_detection  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(23_Vehicle_and_traffic_lane_detection  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


# -------------------------------------------------------------------------------------
add_executable(23_AdvanceLaneDetection 
AdvanceLaneDetection/main.cpp
AdvanceLaneDetection/calibrator.cpp
AdvanceLaneDetection/calibrator.h
AdvanceLaneDetection/utils.h
AdvanceLaneDetection/windowbox.cpp
AdvanceLaneDetection/windowbox.h
)

target_link_libraries(23_AdvanceLaneDetection  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(23_AdvanceLaneDetection  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)







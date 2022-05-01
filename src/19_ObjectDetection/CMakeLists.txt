
# --------------------------------------------------------------
add_executable(19_YOLOV3)

target_sources(19_YOLOV3 PRIVATE YOLOv3/main.cpp 
YOLOv3/Darknet.h
YOLOv3/Darknet.cpp
)

target_link_libraries(19_YOLOV3 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(19_YOLOV3 PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------
FILE(GLOB ALL_SOURCES "YOLOv4/*.cpp" 
					  "YOLOv4/src/*.cpp" 
					  "YOLOv4/src/models/*.cpp" 
					  "YOLOv4/src/utils/*.cpp")


add_executable(19_YOLOV4 YOLOv4/src/main.cpp ${ALL_SOURCES})
target_link_libraries(19_YOLOV4 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(19_YOLOV4 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# --------------------------------------------------------------
add_executable(19_YOLOV4_2)
target_sources(19_YOLOV4_2 PRIVATE YOLOv4_2/yolo4.cpp)

target_link_libraries(19_YOLOV4_2 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(19_YOLOV4_2 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------
add_executable(19_YOLOV5)
target_sources(19_YOLOV5 PRIVATE YOLOv5/yolo.cpp)

target_link_libraries(19_YOLOV5 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(19_YOLOV5 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)



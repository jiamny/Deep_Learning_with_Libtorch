cmake_minimum_required(VERSION 3.18)

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

# -------------------------------------------------------------

add_executable(19_YOLOV5)
target_sources(19_YOLOV5 PRIVATE YOLOv5/src/YOLOv5LibTorch.cpp)

target_link_libraries(19_YOLOV5 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(19_YOLOV5 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

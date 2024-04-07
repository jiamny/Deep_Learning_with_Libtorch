
# --------------------------------------------------------------
add_executable(19_YOLOV3)

target_sources(19_YOLOV3 PRIVATE YOLOv3/main.cpp 
YOLOv3/Darknet.h
YOLOv3/Darknet.cpp
)

target_link_libraries(19_YOLOV3 ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_YOLOV3 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------
add_executable(19_YOLOv3_DNN)

target_sources(19_YOLOv3_DNN PRIVATE YOLOv3_DNN/dnn_yolov3.cpp)

target_link_libraries(19_YOLOv3_DNN ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_YOLOv3_DNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------
FILE(GLOB ALL_SOURCES "YOLOv4/*.cpp" 
					  "YOLOv4/src/*.cpp" 
					  "YOLOv4/src/models/*.cpp" 
					  "YOLOv4/src/utils/*.cpp")


add_executable(19_YOLOV4 YOLOv4/src/main.cpp ${ALL_SOURCES})
target_link_libraries(19_YOLOV4 ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_YOLOV4 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------
add_executable(19_YOLOV4_2)
target_sources(19_YOLOV4_2 PRIVATE YOLOv4_2/yolo4.cpp)

target_link_libraries(19_YOLOV4_2 ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_YOLOV4_2 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
add_executable(19_YOLOV5)
target_sources(19_YOLOV5 PRIVATE YOLOv5/yolo.cpp)

target_link_libraries(19_YOLOV5 ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_YOLOV5 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
add_executable(19_YOLOV5_video)
target_sources(19_YOLOV5_video PRIVATE YOLOv5/yolo_video.cpp)

target_link_libraries(19_YOLOV5_video ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_YOLOV5_video PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


# -------------------------------------------------------------
add_executable(19_DNN_SSD_demo)
target_sources(19_DNN_SSD_demo PRIVATE OpenCV_DNN/dnn_ssd_demo.cpp)

target_link_libraries(19_DNN_SSD_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_DNN_SSD_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
add_executable(19_DetectBaseTrack_demo)
target_sources(19_DetectBaseTrack_demo PRIVATE OpenCV_DNN/DetectBaseTrack_Demo.cpp)

target_link_libraries(19_DetectBaseTrack_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_DetectBaseTrack_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
add_executable(19_Yolo_libtorch_demo)
target_sources(19_Yolo_libtorch_demo PRIVATE 
Yolo_libtorch/main.cpp
Yolo_libtorch/Parameter.h
Yolo_libtorch/Parameter.cpp
Yolo_libtorch/Yolo.h
Yolo_libtorch/Yolo.cpp
)

target_link_libraries(19_Yolo_libtorch_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(19_Yolo_libtorch_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)






aux_source_directory(. PROCESSING_SRCS)

add_executable(26_Yolov3_deepsort_processing ${PROCESSING_SRCS})
target_link_libraries(26_Yolov3_deepsort_processing ${OpenCV_LIBS} detection tracking ${STDCXXFS})
target_include_directories(26_Yolov3_deepsort_processing PRIVATE "${PROJECT_BINARY_DIR}")
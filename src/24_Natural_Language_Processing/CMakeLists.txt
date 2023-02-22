
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

add_executable(24_Text_classification
				Text_classification/nlp_predicator.cpp 
)

target_link_libraries(24_Text_classification  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(24_Text_classification  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

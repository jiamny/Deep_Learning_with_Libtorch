
find_package (Eigen3 REQUIRED NO_MODULE)

add_definitions     ( ${EIGEN3_DEFINITIONS} )

# tell CMake to search for the static libraries ONLY
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

set(LIBS ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${requiredlibs}) 

set(INCLUDE_DIRS
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
)

# -------------------- add executable ------------------------
include_directories(${INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lm -ldl")

add_executable(13_Tf_learning)
target_sources(13_Tf_learning PRIVATE Transfer_learning/main.cpp
                                          Transfer_learning/main.h
)

target_link_libraries(13_Tf_learning ${LIBS})
set_target_properties(13_Tf_learning PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(13_Tf_classify)
target_sources(13_Tf_classify PRIVATE Transfer_learning/classify.cpp)

target_link_libraries(13_Tf_classify ${LIBS})
set_target_properties(13_Tf_classify PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(13_Image_classification_pretrained)
target_sources(13_Image_classification_pretrained PRIVATE Image_classification_pretrained/main.cpp)

target_link_libraries(13_Image_classification_pretrained ${LIBS})
set_target_properties(13_Image_classification_pretrained PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(13_Mask_face_detection_classify)
target_sources(13_Mask_face_detection_classify PRIVATE 
	Mask_face_detection/classify.cpp
	Mask_face_detection/utils.hpp
	Mask_face_detection/utils.cpp
)

target_link_libraries(13_Mask_face_detection_classify ${LIBS})
set_target_properties(13_Mask_face_detection_classify PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(13_Mask_face_detection_train)
target_sources(13_Mask_face_detection_train PRIVATE 
	Mask_face_detection/train.cpp
	Mask_face_detection/utils.hpp
	Mask_face_detection/utils.cpp
	Mask_face_detection/RMFD.hpp
	Mask_face_detection/RMFD.cpp
)

target_link_libraries(13_Mask_face_detection_train ${LIBS})
set_target_properties(13_Mask_face_detection_train PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


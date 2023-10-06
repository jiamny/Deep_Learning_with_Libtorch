

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB ALL_SOURCES "*.hpp" "./src/*.cpp" "./src/*/*.cpp" "./src/*/*.h" "./src/*.h")
add_executable(25_Diffusion_train train_demo.cpp ${ALL_SOURCES})
target_compile_features(25_Diffusion_train PUBLIC cxx_range_for)
target_link_libraries(25_Diffusion_train ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(25_Diffusion_train PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------------------------------------
add_executable(25_Diffusion_test test_demo.cpp ${ALL_SOURCES})
target_compile_features(25_Diffusion_test PUBLIC cxx_range_for)
target_link_libraries(25_Diffusion_test ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(25_Diffusion_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
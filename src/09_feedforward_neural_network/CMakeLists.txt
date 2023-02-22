

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# -----------------------------------------------------------------------------------------
add_executable(09_feedforward)
target_sources(09_feedforward PRIVATE main.cpp neural_net.h neural_net.cpp)

target_link_libraries(09_feedforward ${TORCH_LIBRARIES})
set_target_properties(09_feedforward PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



set(EXECUTABLE_NAME 11_cifar10_classifier)

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

# -----------------------------------------------------------------------------------------
add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE cifar10_clf/cifar10_classifier.cpp
                                		  cifar10_clf/cifar10.cpp
                                          cifar10_clf/nnet.cpp
                                          cifar10_clf/cifar10.h
                                          cifar10_clf/nnet.h
)


target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} ${requiredlibs} )

set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(11_Conv_mnist Conv_mnist.cpp)
target_compile_features(11_Conv_mnist PUBLIC cxx_range_for)
target_link_libraries(11_Conv_mnist ${TORCH_LIBRARIES})

set_target_properties(11_Conv_mnist PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

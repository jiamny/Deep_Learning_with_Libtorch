
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)

message(STATUS "    Eigen3 include: ${EIGEN3_INCLUDE_DIRS}")

# -----------------------------------------------------------------------------------------
add_executable(06_TrainPPO)
target_sources(06_TrainPPO PRIVATE TrainPPO.cpp
                                ProximalPolicyOptimization.h
                                Models.h
                                TestEnvironment.h
)

target_include_directories(06_TrainPPO PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(06_TrainPPO ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(06_TrainPPO PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(06_TestPPO TestPPO.cpp) 
target_sources(06_TestPPO PRIVATE TestPPO.cpp
                               ProximalPolicyOptimization.h
                               Models.h
                               TestEnvironment.h
)

target_include_directories(06_TestPPO PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(06_TestPPO ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(06_TestPPO PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------------
add_executable(06_simple_optimization_example) 
target_sources(06_simple_optimization_example PRIVATE simple_optimization_example.cpp)

target_link_libraries(06_simple_optimization_example ${TORCH_LIBRARIES})
set_target_properties(06_simple_optimization_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------------------
add_executable(06_Loss_function) 
target_sources(06_Loss_function PRIVATE Loss_function.cpp
										../LRdataset.h
										../LRdataset.cpp)

target_link_libraries(06_Loss_function ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(06_Loss_function PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



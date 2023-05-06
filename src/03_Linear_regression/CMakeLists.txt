
set( requiredlibs "pthread" )
list( APPEND requiredlibs "dl" )
list( APPEND requiredlibs "m" )

# -----------------------------------------------------------------------------------------
add_executable(03_Linear_regression)
target_sources(03_Linear_regression PRIVATE Linear_regression.cpp)

target_link_libraries( 03_Linear_regression ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties( 03_Linear_regression PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -----------------------------------------------------------------------------------------
add_executable( 03_LR_fashion )
target_sources( 03_LR_fashion PRIVATE LR_fashion.cpp
                                fashion.cpp
                                transform.cpp
                                fashion.h
                                transform.h
)

target_link_libraries( 03_LR_fashion ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_LR_fashion PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(03_LR_reg_BostonHousing)
target_sources(03_LR_reg_BostonHousing PRIVATE LR_reg_BostonHousing.cpp
                                ../csvloader.h
)

target_link_libraries(03_LR_reg_BostonHousing ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(03_LR_reg_BostonHousing PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(03_Regression_and_gradient)
target_sources(03_Regression_and_gradient PRIVATE Regression_and_gradient.cpp)

target_link_libraries(03_Regression_and_gradient ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_Regression_and_gradient PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(03_Linear_reg_exp2)
target_sources(03_Linear_reg_exp2 PRIVATE Linear_reg_exp2.cpp)

target_link_libraries(03_Linear_reg_exp2 ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(03_Linear_reg_exp2 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------------------------
add_executable(03_regression)
target_sources(03_regression PRIVATE regression.cpp)

target_link_libraries( 03_regression ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties( 03_regression PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -----------------------------------------------------------------------------------------
add_executable(03_Low_level_api_linear_reg)
target_sources(03_Low_level_api_linear_reg PRIVATE Low_level_api_linear_reg.cpp 
													../LRdataset.h
													../LRdataset.cpp)

target_link_libraries(03_Low_level_api_linear_reg ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_Low_level_api_linear_reg PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


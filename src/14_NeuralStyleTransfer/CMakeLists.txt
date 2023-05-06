project(neural-style-transfer VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME 14_style_transfer)

# -----------------------------------------------------------------------------------------
add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE style_transfer/main.cpp
                                          style_transfer/vggnet.cpp
                                          style_transfer/vggnet.h
)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} image-io)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


if(CREATE_SCRIPTMODULES AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/style_transfer/model/vgg19_layers.pt")
    execute_process(COMMAND ${Python3_EXECUTABLE} "create_vgg19_layers_scriptmodule.py" 
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/model")
endif()


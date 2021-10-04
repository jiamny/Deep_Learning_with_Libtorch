cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(Deep_learning_with_pytorch VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." OFF)
option(CREATE_SCRIPTMODULES "Automatically create all required scriptmodule files at build-time (requires python3)." OFF)

set(PYTORCH_VERSION "1.9.1")

set(CMAKE_CXX_STANDARD 14)
#--------------------------------------------------------------------------------------------
# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
# -------------------------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")

# add torch package
find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

message(STATUS "Torch library status:")
message(STATUS "    version: ${Torch_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}") 

# add opencv package to the project
find_package( OpenCV 4.3.0 REQUIRED PATHS "/usr/local/")
#find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ) 
SET(OpenCV_LIBRARIES ${OpenCV_LIBS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(fetch_libtorch)
endif()

if(CREATE_SCRIPTMODULES)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

INCLUDE_DIRECTORIES( ${TORCH_INCLUDE_DIRS} )

# Dataset fetching
if(DOWNLOAD_DATASETS)
    set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
    file(MAKE_DIRECTORY ${DATA_DIR})

    add_custom_target(mnist COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_mnist.cmake)
    add_custom_target(cifar10 COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_cifar10.cmake)
    add_custom_target(penntreebank COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_penntreebank.cmake)
    add_custom_target(neural_style_transfer_images COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_neural_style_transfer_images.cmake)
    add_custom_target(flickr8k COMMAND ${CMAKE_COMMAND}
        -D DATA_DIR=${DATA_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_flickr8k.cmake)
endif()


# External dependencies
add_subdirectory("extern")

# Utils
add_subdirectory("utils/image_io")
    
# Add  sub-projects:
add_subdirectory("src/01_Introducing/pytorch_basics")
add_subdirectory("src/01_Introducing/pytorch_overview")

add_custom_target(01_Introducing)
add_dependencies(01_Introducing  
	01_pytorch_basics 
	01_pytorch_overview)

add_subdirectory("src/02_TensorOps")	
add_subdirectory("src/03_Linear_regression")
add_subdirectory("src/04_Logistic_regression")
add_subdirectory("src/05_BasicModels")
add_subdirectory("src/06_Optimization")
add_subdirectory("src/07_Dataset_and_dataloader")
add_subdirectory("src/08_Custom_Dataset")
add_subdirectory("src/09_feedforward_neural_network")
add_subdirectory("src/10_Neural_Network")
add_subdirectory("src/11_Convolutional_Neural_Network")
add_subdirectory("src/12_classcal_CNN_models")
add_subdirectory("src/13_Transfer_learning")
add_subdirectory("src/14_NeuralStyleTransfer")
add_subdirectory("src/15_GenerativeAdversarialNetwork")
add_subdirectory("src/16_Autoencoder")
add_subdirectory("src/17_ImageCaptioning")
add_subdirectory("src/18_RecurrentNeuralNetwork")




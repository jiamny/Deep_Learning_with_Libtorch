cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(Deep_learning_with_pytorch VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." OFF)
option(CREATE_SCRIPTMODULES "Automatically create all required scriptmodule files at build-time (requires python3)." OFF)

set(PYTORCH_VERSION "1.13.1")

set(USE_CPU_ONLY NO)


if( NOT USE_CPU_ONLY )

	find_package(PythonInterp REQUIRED)
	
	if(NOT DEFINED CUDA_ARCHITECTURES)
		# change based on your GPU. check at: https://developer.nvidia.com/cuda-gpus
   		set(CUDA_ARCHITECTURES 86)
	endif()
	
	# change libtorch path
	set( LIBTORCH_PATH "/home/stree/libtorch_gpu" )
	
	
	set( CUDA_HOME /usr/local/cuda)
	set( CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda )
	set( CUDA_CUDART_LIBRARY /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcudart.so )
	set( CUDA_nvrtc_LIBRARY /usr/local/cuda-11.7/targets/x86_64-linux/lib/libnvrtc.so )
	set( CUDA_curand_LIBRARY /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcurand.so )
	set( CUDA_cufft_LIBRARY /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcufft.so )
	set( CUDA_cublas_LIBRARY /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublas.so )
	set( CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc )
	# set( CUDA_cublas_device_LIBRARY /usr/local/cuda/lib64 )
	set( CUDA_INCLUDE_DIRS /usr/local/cuda/include )
	# SET(CMAKE_CUDA_COMPILER_ENV_VAR /usr/local/cuda/bin/nvcc)

	set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch" )
	#--------------------------------------------------------------------------------------------
	# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
	#-------------------------------------------------------------------------------------------- 

	list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
             
	set(CMAKE_MODULE_PATH "${LIBTORCH_PATH}/share/cmake/Torch")  
	# add torch package
	find_package(Torch REQUIRED PATHS "${LIBTORCH_PATH}") 
else()
	# change libtorch path
	set( LIBTORCH_PATH "/home/stree/libtorch_cpu" )
	
	set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch" )
	#--------------------------------------------------------------------------------------------
	# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
	#-------------------------------------------------------------------------------------------- 

	list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}/share/cmake")
             
	set(CMAKE_MODULE_PATH "${LIBTORCH_PATH}/share/cmake/Torch")    
	# add torch package
	find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${LIBTORCH_PATH}") 
endif()

set(CMAKE_CXX_STANDARD 14)

message(STATUS "Torch library status:")
message(STATUS "    version: ${Torch_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}") 

# add opencv package to the project
FIND_PACKAGE( OpenCV 4.5.4 REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ) 
SET(OpenCV_LIBRARIES ${OpenCV_LIBS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ) 

file( GLOB_RECURSE Python310_LIB "/usr/lib/python3.10/config-3.10-x86_64-linux-gnu/*.so" )
SET( Python310_INCLUDE_DIRS "/usr/include/python3.10" )
INCLUDE_DIRECTORIES( include ${Python310_INCLUDE_DIRS} ) 

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(fetch_libtorch)
endif()

if(CREATE_SCRIPTMODULES)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

INCLUDE_DIRECTORIES( ${TORCH_INCLUDE_DIRS} )

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

message(STATUS "    include: ${DMATPLOTLIBCPP_PYTHON_HEADER}")
message(STATUS "    python3: ${Python3_INCLUDE_DIRS}")
message(STATUS "    numpy: ${Python3_NumPy_INCLUDE_DIRS}")


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
add_subdirectory("src/image_tools")
    
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
add_subdirectory("src/06_Optimization_and_loss_function")
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
add_subdirectory("src/19_ObjectDetection")
add_subdirectory("src/20_Img-to-Img_Translation")
add_subdirectory("src/21_Semantic_Segmentation")
add_subdirectory("src/22_Reinforcement_Learning")
add_subdirectory("src/23_Vehicle_and_traffic_lane_detection")
add_subdirectory("src/24_Natural_Language_Processing")









cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17) # To use C++ <filesystem>

# use openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
 

set(requiredlibs ${OpenCV_LIBS})
list(APPEND requiredlibs ${TORCH_LIBRARIES})
list(APPEND requiredlibs "png16")

# ------------------------------------------------------------------
# ----AlexNet_custom
add_executable(12_AlexNet_custom)
target_sources(12_AlexNet_custom PRIVATE  AlexNet/custom/alexnet_demo.cpp
    AlexNet/custom/alexnet.cpp
    AlexNet/custom/alexnet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_AlexNet_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_AlexNet_custom PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ------------------------------------------------------------------
# ----AlexNet_torchvision
add_executable(12_AlexNet_torchvision)
target_sources(12_AlexNet_torchvision PRIVATE  AlexNet/torchvision/main.cpp
	modelsimpl.h
    AlexNet/torchvision/alexnet.cpp
    AlexNet/torchvision/alexnet.h
    dataSet.cpp
    dataSet.h
    modelsimpl.h
)

target_link_libraries(12_AlexNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_AlexNet_torchvision PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ------------------------------------------------------------------
# --- DenseNet torchvision
add_executable(12_DenseNet_torchvision)
target_sources(12_DenseNet_torchvision PRIVATE  DenseNet/torchvision/main.cpp
	modelsimpl.h
    DenseNet/torchvision/densenet.cpp
    DenseNet/torchvision/densenet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_DenseNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_DenseNet_torchvision PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ----------------------------------------------------------------------
# --- DeepResidualNetwork
add_executable(12_DeepResidualNetwork)
target_sources(12_DeepResidualNetwork PRIVATE  ResNet/DeepResidualNetwork/main.cpp
    cifar10.cpp
    cifar10.h
    ResNet/DeepResidualNetwork/residual_block.cpp
    ResNet/DeepResidualNetwork/residual_block.h
    ResNet/DeepResidualNetwork/resnet.h
    transform.h
    transform.cpp
)

target_link_libraries(12_DeepResidualNetwork ${requiredlibs})

set_target_properties(12_DeepResidualNetwork PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


# --- DPN ---------------------------------------------------
add_executable(12_DPN_32)
target_sources(12_DPN_32 PRIVATE  DPN/DPN_32/main.cpp
    DPN/DPN_32/dpn.cpp
    DPN/DPN_32/dpn.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_DPN_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_DPN_32 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# --- EfficientNet_32 ---------------------------------------------------
add_executable(12_EfficientNet_32)
target_sources(12_EfficientNet_32 PRIVATE  EfficientNet/EfficientNet_32/main.cpp
    EfficientNet/EfficientNet_32/efficientnet.cpp
    EfficientNet/EfficientNet_32/efficientnet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_EfficientNet_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_EfficientNet_32 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- Inception_V3
add_executable(12_Inception_V3)
target_sources(12_Inception_V3 PRIVATE  Inception/V3/main.cpp
    Inception/V3/inception.cpp
    Inception/V3/inception.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_Inception_V3 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_Inception_V3 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------
# --- GoogleNet torch_cifar10
add_executable(12_GoogleNet_custom_cifar)
target_sources(12_GoogleNet_custom_cifar PRIVATE  GoogleNet/custom_cifar/main.cpp
    GoogleNet/custom_cifar/googlenet.cpp
    GoogleNet/custom_cifar/googlenet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_GoogleNet_custom_cifar ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_GoogleNet_custom_cifar PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#---------------------------------------------------------------------
# --- LeNet torch_cifar10
add_executable(12_LeNet_torch_cifar10)
target_sources(12_LeNet_torch_cifar10 PRIVATE  LeNet/torch_cifar10/main.cpp
    LeNet/torch_cifar10/nnet.cpp
    LeNet/torch_cifar10/nnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(12_LeNet_torch_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_LeNet_torch_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- LeNet custom_cifar10
add_executable(12_LeNet_custom_cifar10)
target_sources(12_LeNet_custom_cifar10 PRIVATE  LeNet/custom_cifar10/main.cpp
    LeNet/custom_cifar10/nnet.cpp
    LeNet/custom_cifar10/nnet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_LeNet_custom_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_LeNet_custom_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------------
# --- MnasNet torchvision
add_executable(12_MnasNet_torchvision)
target_sources(12_MnasNet_torchvision PRIVATE  MnasNet/torchvision/main.cpp
	modelsimpl.h
    MnasNet/torchvision/mnasnet.cpp
    MnasNet/torchvision/mnasnet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_MnasNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_MnasNet_torchvision PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -------------------------------------------------------------------------------
# --- MobileNet torchvision V2 ---------------------------------------------------
add_executable(12_MobileNet_torchvisionV2)
target_sources(12_MobileNet_torchvisionV2 PRIVATE  MobileNet/torchvisionV2/main.cpp
	modelsimpl.h
    MobileNet/torchvisionV2/mobilenetv2.cpp
    MobileNet/torchvisionV2/mobilenetv2.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_MobileNet_torchvisionV2 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_MobileNet_torchvisionV2 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# --- PNASNet_32 ---------------------------------------------------
add_executable(12_PnasNet_32)
target_sources(12_PnasNet_32 PRIVATE  PnasNet/PnasNet_32/main.cpp
    PnasNet/PnasNet_32/pnasnet.cpp
    PnasNet/PnasNet_32/pnasnet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_PnasNet_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_PnasNet_32 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#---------------------------------------------------------------------
# --- PreactResNet_32 torch_cifar10
add_executable(12_PreactResNet_torch_cifar10)
target_sources(12_PreactResNet_torch_cifar10 PRIVATE  PreactResnet/PreactResNet_32/main.cpp
    PreactResnet/PreactResNet_32/PreactResNet.cpp
    PreactResnet/PreactResNet_32/PreactResNet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(12_PreactResNet_torch_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_PreactResNet_torch_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#---------------------------------------------------------------------
# --- RegNet_32 torch_cifar10
add_executable(12_RegNet_torch_cifar10)
target_sources(12_RegNet_torch_cifar10 PRIVATE  RegNet/RegNet_32/main.cpp
    RegNet/RegNet_32/regnet.cpp
    RegNet/RegNet_32/regnet.h
    cifar10.cpp
    cifar10.h
)

target_link_libraries(12_RegNet_torch_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_RegNet_torch_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------
# --- ResNet torch_cifar10
add_executable(12_ResNet_torch_cifar10)
target_sources(12_ResNet_torch_cifar10 PRIVATE ResNet/torch_cifar10/resnet-cifar-demo.cpp
									cifar10.h
									cifar10.cpp
									ResNet/torch_cifar10/resnet.h
									ResNet/torch_cifar10/resnet.cpp
)
target_link_libraries(12_ResNet_torch_cifar10 PRIVATE "${TORCH_LIBRARIES}")

set_target_properties(12_ResNet_torch_cifar10 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
# ---------------------------------------------------------------------

# --- ResNet torchvision
add_executable(12_ResNet_torchvision)
target_sources(12_ResNet_torchvision PRIVATE  ResNet/torchvision/main.cpp
    ResNet/torchvision/resnet.cpp
    ResNet/torchvision/resnet.h
    dataSet.cpp
    dataSet.h
    modelsimpl.h
)

target_link_libraries(12_ResNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_ResNet_torchvision PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# --- MobileNet V1 ---------------------------------------------------
add_executable(12_MobileNet_V1)
target_sources(12_MobileNet_V1 PRIVATE  MobileNet/V1/main.cpp
    MobileNet/V1/mobilenetv1.cpp
    MobileNet/V1/mobilenetv1.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_MobileNet_V1 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_MobileNet_V1 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------------------------
# --- ShuffleNet torchvision V2 ---------------------------------------------------
add_executable(12_ShuffleNet_torchvisionV2)
target_sources(12_ShuffleNet_torchvisionV2 PRIVATE  ShuffleNet/torchvisionV2/main.cpp
	modelsimpl.h
    ShuffleNet/torchvisionV2/shufflenetv2.cpp
    ShuffleNet/torchvisionV2/shufflenetv2.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_ShuffleNet_torchvisionV2 ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_ShuffleNet_torchvisionV2 PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ------------------------------------------------------------------------------
# --- SqueezeNet torchvision ---------------------------------------------------
add_executable(12_SqueezeNet_torchvision)
target_sources(12_SqueezeNet_torchvision PRIVATE  SqueezeNet/torchvision/main.cpp
	modelsimpl.h
    SqueezeNet/torchvision/squeezenet.cpp
    SqueezeNet/torchvision/squeezenet.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_SqueezeNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_SqueezeNet_torchvision PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# --------------------------------------------------------------------
# --- VGG
add_executable(12_VGG_custom)
target_sources(12_VGG_custom PRIVATE  VGG/custom/VGG_demo.cpp
    VGG/custom/vgg.cpp
    VGG/custom/vgg.h
    dataSet.cpp
    dataSet.h
    VGG/custom/Classification.cpp
    VGG/custom/Classification.h
)

target_link_libraries(12_VGG_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_VGG_custom PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
# --------------------------------------------------------------------

# --- VGG_custom_cifar
add_executable(12_VGG_custom_cifar)
target_sources(12_VGG_custom_cifar PRIVATE  VGG/custom_cifar/main.cpp
    VGG/custom_cifar/vgg.cpp
    VGG/custom_cifar/vgg.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_VGG_custom_cifar ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_VGG_custom_cifar PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# -----------------------------------------------------------------------
# --- VGG torchvision 
add_executable(12_VGG_torchvision)
target_sources(12_VGG_torchvision PRIVATE  VGG/torchvision/main.cpp	
    VGG/torchvision/vgg.cpp
    VGG/torchvision/vgg.h
    dataSet.cpp
    dataSet.h
    modelsimpl.h
)

target_link_libraries(12_VGG_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_target_properties(12_VGG_torchvision PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


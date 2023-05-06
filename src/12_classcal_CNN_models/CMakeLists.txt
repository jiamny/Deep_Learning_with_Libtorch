
find_package(OpenMP REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PNG_INCLUDE_DIRS})

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")

list(APPEND requiredlibs ${PNG_LIBRARIES})
list(APPEND requiredlibs ${OpenMP_CXX_LIBRARIES})

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

# ------------------------------------------------------------------
# ----AlexNet_custom
add_executable(12_AlexNet_custom)
target_sources(12_AlexNet_custom PRIVATE  AlexNet/custom/alexnet_demo.cpp
    AlexNet/custom/alexnet.cpp
    AlexNet/custom/alexnet.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_AlexNet_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_AlexNet_custom PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ------------------------------------------------------------------
# ----AlexNet_torchvision
add_executable(12_AlexNet_torchvision)
target_sources(12_AlexNet_torchvision PRIVATE  AlexNet/torchvision/main.cpp
    AlexNet/torchvision/alexnet.cpp
    AlexNet/torchvision/alexnet.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_AlexNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_AlexNet_torchvision PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ------------------------------------------------------------------
# ----DenseNet torchvision
add_executable(12_DenseNet_torchvision)
target_sources(12_DenseNet_torchvision PRIVATE  DenseNet/torchvision/main.cpp
    DenseNet/torchvision/densenet.cpp
    DenseNet/torchvision/densenet.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_DenseNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_DenseNet_torchvision PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --- DPN ---------------------------------------------------
add_executable(12_DPN_32)
target_sources(12_DPN_32 PRIVATE  DPN/DPN_32/main.cpp
    DPN/DPN_32/dpn.cpp
    DPN/DPN_32/dpn.h
    ../07_Dataset_and_dataloader/cifar10.cpp
    ../07_Dataset_and_dataloader/cifar10.h
)

target_link_libraries(12_DPN_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_DPN_32 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --- EfficientNet_32 ---------------------------------------------------
add_executable(12_EfficientNet_32)

target_sources(12_EfficientNet_32 PRIVATE EfficientNet/EfficientNet_32/main.cpp
    EfficientNet/EfficientNet_32/efficientnet.cpp
    EfficientNet/EfficientNet_32/efficientnet.h
    ../07_Dataset_and_dataloader/cifar10.cpp
    ../07_Dataset_and_dataloader/cifar10.h
)

target_link_libraries(12_EfficientNet_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_EfficientNet_32 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------
# --- GoogleNet torch
add_executable(12_GoogleNet_custom)

target_sources(12_GoogleNet_custom PRIVATE  GoogleNet/custom/main.cpp
    GoogleNet/custom/googlenet.cpp
    GoogleNet/custom/googlenet.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_GoogleNet_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_GoogleNet_custom PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------
# --- Inception_V3
add_executable(12_Inception_V3)

target_sources(12_Inception_V3 PRIVATE  Inception/V3/main.cpp
    Inception/V3/inception.cpp
    Inception/V3/inception.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_Inception_V3 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_Inception_V3 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------
# --- MnasNet
add_executable(12_MnasNet)

target_sources(12_MnasNet PRIVATE MnasNet/torchvision/main.cpp
    MnasNet/torchvision/mnasnet.h
    MnasNet/torchvision/mnasnet.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_MnasNet ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_MnasNet PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------
# --- MobileNet torchvision V2
add_executable(12_MobileNet_torchvisionV2)

target_sources(12_MobileNet_torchvisionV2 PRIVATE   MobileNet/torchvisionV2/main.cpp
    MobileNet/torchvisionV2/mobilenetv2.cpp
    MobileNet/torchvisionV2/mobilenetv2.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_MobileNet_torchvisionV2 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_MobileNet_torchvisionV2 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --- MobileNet V1 ---------------------------------------------------
add_executable(12_MobileNet_V1)

target_sources(12_MobileNet_V1 PRIVATE MobileNet/V1/main.cpp
    MobileNet/V1/mobilenetv1.cpp
    MobileNet/V1/mobilenetv1.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_MobileNet_V1 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_MobileNet_V1 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --- PNASNet_custom ---------------------------------------------------
add_executable(12_PnasNet_custom)

target_sources(12_PnasNet_custom PRIVATE  PnasNet/PnasNet_custom/main.cpp
    PnasNet/PnasNet_custom/pnasnet.cpp
    PnasNet/PnasNet_custom/pnasnet.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_PnasNet_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_PnasNet_custom PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#---------------------------------------------------------------------
# --- PreactResNet_32 torch_cifar10
add_executable(12_PreactResNet_32)

target_sources(12_PreactResNet_32 PRIVATE PreactResnet/PreactResNet_32/main.cpp
    PreactResnet/PreactResNet_32/PreactResNet.cpp
    PreactResnet/PreactResNet_32/PreactResNet.h
    ../07_Dataset_and_dataloader/cifar10.cpp
    ../07_Dataset_and_dataloader/cifar10.h
)

target_link_libraries(12_PreactResNet_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_PreactResNet_32 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#---------------------------------------------------------------------
# --- RegNet_32 torch_cifar10
add_executable(12_RegNet_32)

target_sources(12_RegNet_32 PRIVATE RegNet/RegNet_32/main.cpp
    RegNet/RegNet_32/regnet.cpp
    RegNet/RegNet_32/regnet.h
    ../07_Dataset_and_dataloader/cifar10.cpp
    ../07_Dataset_and_dataloader/cifar10.h
)

target_link_libraries(12_RegNet_32 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_RegNet_32 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------
# --- ResNet
add_executable(12_ResNet_custom)

target_sources(12_ResNet_custom PRIVATE ResNet/resnet_custom/resnet.cpp
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_ResNet_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_ResNet_custom PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------
# --- ResNet torchvision
add_executable(12_ResNet_torchvision)

target_sources(12_ResNet_torchvision PRIVATE ResNet/torchvision/main.cpp
    ResNet/torchvision/resnet.cpp
    ResNet/torchvision/resnet.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_ResNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_ResNet_torchvision PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------------------------
# --- ShuffleNet torchvision V2 ---------------------------------------------------
add_executable(12_ShuffleNet_torchvisionV2)

target_sources(12_ShuffleNet_torchvisionV2 PRIVATE ShuffleNet/torchvisionV2/main.cpp
	modelsimpl.h
    ShuffleNet/torchvisionV2/shufflenetv2.cpp
    ShuffleNet/torchvisionV2/shufflenetv2.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_ShuffleNet_torchvisionV2 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(12_ShuffleNet_torchvisionV2 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ------------------------------------------------------------------------------
# --- SqueezeNet torchvision ---------------------------------------------------
add_executable(12_SqueezeNet_torchvision)

target_sources(12_SqueezeNet_torchvision PRIVATE ShuffleNet/torchvisionV2/main.cpp
	modelsimpl.h
    ShuffleNet/torchvisionV2/shufflenetv2.cpp
    ShuffleNet/torchvisionV2/shufflenetv2.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_SqueezeNet_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_SqueezeNet_torchvision PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------------
# --- VGG
add_executable(12_VGG_custom)
target_sources(12_VGG_custom PRIVATE  VGG/custom/VGG_demo.cpp
    VGG/custom/vgg.cpp
    VGG/custom/vgg.h
    VGG/custom/Classification.cpp
    VGG/custom/Classification.h
    dataSet.cpp
    dataSet.h
)

target_link_libraries(12_VGG_custom ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_target_properties(12_VGG_custom PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# --------------------------------------------------------------------
# --- VGG_custom_mc
add_executable(12_VGG_custom_mc)

target_sources(12_VGG_custom_mc PRIVATE VGG/custom_mc/main.cpp
    VGG/custom_mc/vgg.cpp
    VGG/custom_mc/vgg.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_VGG_custom_mc ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_VGG_custom_mc PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -----------------------------------------------------------------------
# --- VGG torchvision 
add_executable(12_VGG_torchvision)

target_sources(12_VGG_torchvision PRIVATE VGG/torchvision/main.cpp	
    VGG/torchvision/vgg.cpp
    VGG/torchvision/vgg.h
    ../image_tools/dataloader.hpp
	../image_tools/dataloader.cpp
	../image_tools/datasets.hpp
	../image_tools/datasets.cpp
	../image_tools/transforms.hpp
	../image_tools/transforms.cpp
)

target_link_libraries(12_VGG_torchvision ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(12_VGG_torchvision PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)




"""
This python script converts the network into Script Module
"""

import torch 
from torchvision.models import vgg16, vgg16_bn, VGG16_BN_Weights

'''
UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed 
in the future. The current behavior is equivalent to passing `weights=VGG16_BN_Weights.IMAGENET1K_V1`. 
You can also use `weights=VGG16_BN_Weights.DEFAULT` to get the most up-to-date weights.
'''
model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT) #pretrained=True)
for k,v in model.named_parameters():
    print(k)

model=model.to(torch.device("cuda")) # Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu
model.eval() 
var=torch.ones((1,3,224,224)).cuda()
traced_script_module = torch.jit.trace(model, var) 
traced_script_module.save("vgg16_bn.pt")

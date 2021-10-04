"""
This python script converts the network into Script Module
"""

import torch 
from torchvision.models import vgg16,vgg16_bn

model = vgg16_bn(pretrained=True)
for k,v in model.named_parameters():
    print(k)

model=model.to(torch.device("cpu")) 
model.eval() 
var=torch.ones((1,3,224,224)) 
traced_script_module = torch.jit.trace(model, var) 
traced_script_module.save("vgg16_bn.pt")

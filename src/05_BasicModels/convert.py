"""
This python script converts the network into Script Module
"""

import torch
import torchvision.models as models

resnet18 = models.resnet34(pretrained=True)

# use an example input to trace the operations of the model
example_input = torch.rand(1, 3, 224, 224) # 224 is the least input size, depends on the dataset you use

script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet34.pt')

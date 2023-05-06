import torch
import torchvision

# An instance of your model.
#model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model = (torchvision)
models.quantization.resnet18(pretrained=True, quantize=True)
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("traced_qresnet_model.pt")


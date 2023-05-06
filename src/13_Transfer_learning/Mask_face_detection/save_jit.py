import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# Set upgrading the gradients to False
for param in model.parameters():
    param.requires_grad = False

model.eval()

# Save the model except the final FC Layer
resnet18 = torch.nn.Sequential(*list(model.children())[:-1])

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(resnet18, example)
traced_script_module.save("traced_resnet_without_last_layer.pt")


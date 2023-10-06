#pragma once

#include <torch/torch.h>

void toggle_grad(std::shared_ptr<torch::nn::Module> model, bool requires_grad = true);
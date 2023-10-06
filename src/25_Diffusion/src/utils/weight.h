#pragma once

std::function<void(torch::nn::Module &)> weights_norm_init(double var_scale = 0.02);
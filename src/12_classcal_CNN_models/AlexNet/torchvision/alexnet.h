#pragma once

#include <torch/torch.h>

void weights_init(torch::nn::Module &m);

// AlexNet model architecture from the
// "One weird trick..." <https://arxiv.org/abs/1404.5997> paper.
struct AlexNetImpl : torch::nn::Module {
  torch::nn::Sequential features{nullptr}, classifier{nullptr};

  explicit AlexNetImpl(int64_t num_classes = 1000);
  void init();
  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(AlexNet);

#pragma once
#ifndef GoogleNet_H
#define GoogleNet_H

#include <torch/torch.h>


struct BasicConv2dImpl : torch::nn::Module {
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};

  explicit BasicConv2dImpl(torch::nn::Conv2dOptions options);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BasicConv2d);

struct InceptionImpl : torch::nn::Module {
  BasicConv2d branch1{nullptr};
  torch::nn::Sequential branch2, branch3, branch4;

  InceptionImpl(
      int64_t in_channels,
      int64_t ch1x1,
      int64_t ch3x3red,
      int64_t ch3x3,
      int64_t ch5x5red,
      int64_t ch5x5,
      int64_t pool_proj);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Inception);

struct InceptionAuxImpl : torch::nn::Module {
  BasicConv2d conv{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};

  InceptionAuxImpl(int64_t in_channels, int64_t num_classes);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(InceptionAux);


struct GoogLeNetOutput {
  torch::Tensor output;
  torch::Tensor aux1;
  torch::Tensor aux2;
};

struct GoogLeNetImpl : torch::nn::Module {
  bool aux_logits, transform_input;

  BasicConv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};

  Inception inception3a{nullptr}, inception3b{nullptr},
      inception4a{nullptr}, inception4b{nullptr}, inception4c{nullptr},
      inception4d{nullptr}, inception4e{nullptr}, inception5a{nullptr},
      inception5b{nullptr};

  InceptionAux aux1{nullptr}, aux2{nullptr};

  torch::nn::Dropout dropout{nullptr};
  torch::nn::Linear fc{nullptr};

  explicit GoogLeNetImpl(
      int64_t num_classes = 1000,
      bool aux_logits = true,
      bool transform_input = false,
      bool init_weights = true);

  void _initialize_weights();

  GoogLeNetOutput forward(torch::Tensor x);
};

TORCH_MODULE(GoogLeNet);

#endif


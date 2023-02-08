#pragma once

#include <torch/torch.h>

struct MobileBlockImpl : torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::Conv2d conv2{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::BatchNorm2d bn2{nullptr};

    //'''Depthwise conv + Pointwise conv'''
	MobileBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MobileBlock);

struct MobileNetV1Impl : torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Linear linear{nullptr};
	torch::nn::Sequential layers{nullptr};
	std::vector<std::vector<int64_t>> cfg = {{16, 1}, {32, 2}, {32, 1}, {64, 2}, {64, 1}, {128, 2}, {128, 1}, {256,2},
					{256, 1}, {256, 2}, {512, 1}, {1024, 2}, {1024, 2}, {1024, 1}};

    explicit MobileNetV1Impl(int64_t num_classes);

    torch::nn::Sequential _make_layers(int64_t in_planes);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(MobileNetV1);

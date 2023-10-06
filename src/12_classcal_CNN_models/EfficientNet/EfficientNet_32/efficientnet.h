// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <map>

torch::Tensor swish(torch::Tensor x);

torch::Tensor drop_connect(torch::Tensor x, double drop_ratio);

struct SEImpl : public torch::nn::Module {
	//Squeeze-and-Excitation block with Swish.
	torch::nn::Conv2d se1{nullptr}, se2{nullptr};
	explicit SEImpl(int64_t in_planes, int64_t se_planes, torch::Device device);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(SE);

//'''expansion + depthwise + pointwise + squeeze-excitation'''
struct Block_Impl : public torch::nn::Module {
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
	SE se{nullptr};
	int64_t stride;
	int64_t expand_ratio{1};
	double se_ratio{0.};
	double drop_rate{0.};
	bool has_skip{false};
	bool training{true};

	explicit Block_Impl( int64_t in_planes,
            int64_t out_planes,
            int64_t kernel_size,
            int64_t stride_,
            int64_t expand_ratio_,
            double se_ratio_,
            double drop_rate_,
			torch::Device device);

	torch::Tensor forward(torch::Tensor x);

};

TORCH_MODULE(Block_);

struct EfficientNetImpl : public torch::nn::Module {
	std::map<std::string, std::vector<int64_t>> cfg;
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Sequential layers;
	torch::nn::Linear linear{nullptr};
	torch::nn::AdaptiveAvgPool2d adavgpool{nullptr};

	explicit EfficientNetImpl(std::map<std::string, std::vector<int64_t>> cfg, int64_t num_classes, torch::Device device);

	torch::Tensor forward(torch::Tensor x);
	torch::nn::Sequential  _make_layers(int64_t in_planes, torch::Device device);
};

TORCH_MODULE(EfficientNet);

EfficientNet EfficientNetB0(int64_t num_classes, torch::Device device=torch::kCPU);


// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <cmath>

/*
Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
 */

struct PreActBlockImpl : torch::nn::Module {
    // Pre-activation version of the BasicBlock.
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
	int64_t expansion = 1;
	torch::nn::Sequential shortcut;
	bool hasShortcut{false};

	explicit PreActBlockImpl(int64_t in_planes, int64_t planes, int64_t stride); // stride=1

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PreActBlock);

struct PreActBottleneckImpl : torch::nn::Module {
	// Pre-activation version of the original Bottleneck module.
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
	int64_t expansion{4};
	torch::nn::Sequential shortcut;
	bool hasShortcut{false};

	explicit PreActBottleneckImpl(int64_t in_planes, int64_t planes, int64_t stride /*1*/);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PreActBottleneck);

struct PreActResNetBBImpl : torch::nn::Module {
	int64_t expansion{1};
	int64_t in_planes = 64;
	std::vector<PreActBlock> layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::Linear linear{nullptr};

	explicit PreActResNetBBImpl(std::vector<int64_t> num_blocks, int64_t num_classes);

	std::vector<PreActBlock> _make_layer(int64_t planes, int64_t num_blocks, int64_t stride);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PreActResNetBB);

struct PreActResNetBNImpl : torch::nn::Module {
	int64_t expansion{4};
	int64_t in_planes = 64;
	std::vector<PreActBottleneck> layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::Linear linear{nullptr};

	explicit PreActResNetBNImpl(std::vector<int64_t> num_blocks, int64_t num_classes);

	std::vector<PreActBottleneck> _make_layer(int64_t planes, int64_t num_blocks, int64_t stride);
	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PreActResNetBN);

PreActResNetBB PreActResNet18(int64_t num_classes);

PreActResNetBB PreActResNet34(int64_t num_classes);

PreActResNetBN PreActResNet50(int64_t num_classes);

PreActResNetBN PreActResNet101(int64_t num_classes);

PreActResNetBN PreActResNet152(int64_t num_classes);






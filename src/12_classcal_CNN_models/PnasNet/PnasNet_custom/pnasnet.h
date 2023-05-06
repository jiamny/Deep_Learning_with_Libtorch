// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <cmath>

// Progressive Neural Architecture Search

struct SepConvImpl : torch::nn::Module {
    //Separable Convolution.'''
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};

	explicit SepConvImpl(int in_planes, int64_t out_planes, int64_t kernel_size, int64_t stride);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(SepConv);

struct CellAImpl : torch::nn::Module {
	int64_t stride{2};
	SepConv sep_conv1{nullptr};
	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};

	explicit CellAImpl(int64_t in_planes, int64_t out_planes, int64_t stride /*1*/);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CellA);

struct CellBImpl : torch::nn::Module {
	int64_t stride{2};
	SepConv sep_conv1{nullptr};
	SepConv sep_conv2{nullptr};
	SepConv sep_conv3{nullptr};
	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

	explicit CellBImpl(int64_t in_planes, int64_t out_planes, int64_t stride /*1*/);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(CellB);

struct PNASNetImpl : torch::nn::Module {

	int64_t in_planes;
	std::string cell_type;
	std::vector<CellA> a_layer1, a_layer3, a_layer5;
	CellA a_layer2{nullptr}, a_layer4{nullptr};

	std::vector<CellB> b_layer1, b_layer3, b_layer5;
	CellB b_layer2{nullptr}, b_layer4{nullptr};

	torch::nn::Conv2d conv1{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr};
	torch::nn::Sequential linear;

	explicit PNASNetImpl(std::string cell_type, int64_t num_cells, int64_t num_planes, int64_t num_classes);

	std::vector<CellA> a_make_layer(int64_t num_planes, int64_t num_cells);
	CellA a_downsample(int64_t planes);

	std::vector<CellB> b_make_layer(int64_t num_planes, int64_t num_cells);
	CellB b_downsample(int64_t planes);

	torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(PNASNet);

PNASNet PNASNetA(std::string cell_type, int64_t num_cells, int64_t num_planes, int64_t num_classes);
//    return PNASNet(CellA, num_cells=6, num_planes=44)

PNASNet  PNASNetB(std::string cell_type, int64_t num_cells, int64_t num_planes, int64_t num_classes);
//    return PNASNet(CellB, num_cells=6, num_planes=32)






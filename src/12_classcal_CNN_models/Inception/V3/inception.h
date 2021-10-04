#pragma once

#include <torch/torch.h>

torch::nn::Sequential ConvBNReLU(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding); // stride=1, padding = 0
torch::nn::Sequential ConvBNReLUFactorization(int64_t in_channels, int64_t out_channels,
		const std::vector<int64_t> &kernel_sizes, const std::vector<int64_t> &paddings);

struct InceptionV3ModuleAImpl : torch::nn::Module {
   torch::nn::Sequential branch1, branch2, branch3, branch4;

   InceptionV3ModuleAImpl(int64_t in_channels, int64_t out_channels1, int64_t out_channels2reduce, int64_t  out_channels2,
		  int64_t out_channels3reduce, int64_t  out_channels3, int64_t out_channels4);

   torch::Tensor forward(const torch::Tensor& x);
};


struct InceptionV3ModuleBImpl : torch::nn::Module {
	torch::nn::Sequential branch1, branch2, branch3, branch4;

	InceptionV3ModuleBImpl( int64_t in_channels, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2,
			 int64_t out_channels3reduce, int64_t out_channels3, int64_t out_channels4);

	torch::Tensor forward(const torch::Tensor& x);

};

struct InceptionV3ModuleCImpl : torch::nn::Module {
	torch::nn::Sequential branch1, branch2_conv1, branch3_conv1, branch3_conv2, branch4;
	torch::nn::Sequential branch2_conv2a, branch2_conv2b, branch3_conv3a, branch3_conv3b;

	// in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4
	InceptionV3ModuleCImpl( int64_t in_channels, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2,
			 int64_t out_channels3reduce, int64_t out_channels3, int64_t out_channels4);

	torch::Tensor forward(const torch::Tensor& x);

};


struct InceptionV3ModuleDImpl : torch::nn::Module {
	torch::nn::Sequential branch1, branch2;

	InceptionV3ModuleDImpl( int64_t in_channels, int64_t out_channels1reduce, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2);

	torch::Tensor forward(const torch::Tensor& x);

};


struct InceptionV3ModuleEImpl : torch::nn::Module {
	torch::nn::Sequential branch1, branch2;

	InceptionV3ModuleEImpl( int64_t in_channels, int64_t out_channels1reduce, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2);

	torch::Tensor forward(const torch::Tensor& x);

};

struct InceptionAux_Impl : torch::nn::Module {
	torch::nn::Sequential aux_conv1, aux_conv2;
	torch::nn::Linear fc;

	InceptionAux_Impl( int64_t in_channels, int64_t num_classes);

	torch::Tensor forward(torch::Tensor x);

};

TORCH_MODULE(InceptionV3ModuleA);
TORCH_MODULE(InceptionV3ModuleB);
TORCH_MODULE(InceptionV3ModuleC);
TORCH_MODULE(InceptionV3ModuleD);
TORCH_MODULE(InceptionV3ModuleE);
TORCH_MODULE(InceptionAux_);


struct InceptionV3Output {
  torch::Tensor output;
  torch::Tensor aux;
};

// Inception v3 model architecture from
//"Rethinking the Inception Architecture for Computer Vision"
//<http://arxiv.org/abs/1512.00567>
struct InceptionV3_Impl : torch::nn::Module {
  std::string stage;
  int64_t num_classes;
  torch::nn::Sequential block1, block2;

  InceptionV3ModuleA block3_1 = InceptionV3ModuleA(192, 64, 48, 64, 64, 96, 32);
  InceptionV3ModuleA block3_2 = InceptionV3ModuleA(256, 64, 48, 64, 64, 96, 64);
  InceptionV3ModuleA block3_3 = InceptionV3ModuleA(288, 64, 48, 64, 64, 96, 64);

  InceptionV3ModuleD block4_1 = InceptionV3ModuleD(288, 384, 384, 64, 96);
  InceptionV3ModuleB block4_2 = InceptionV3ModuleB(768, 192, 128, 192, 128, 192, 192);
  InceptionV3ModuleB block4_3 = InceptionV3ModuleB(768, 192, 160, 192, 160, 192, 192);
  InceptionV3ModuleB block4_4 = InceptionV3ModuleB(768, 192, 160, 192, 160, 192, 192);
  InceptionV3ModuleB block4_5 = InceptionV3ModuleB(768, 192, 192, 192, 192, 192, 192);

  InceptionV3ModuleE block5_1 = InceptionV3ModuleE(768, 192, 320, 192, 192);
  InceptionV3ModuleC block5_2 = InceptionV3ModuleC(1280, 320, 384, 384, 448, 384, 192);
  InceptionV3ModuleC block5_3 = InceptionV3ModuleC(2048, 320, 384, 384, 448, 384, 192);

  torch::nn::Sequential linear;

  explicit InceptionV3_Impl(int64_t num_classes, std::string stage);

  InceptionV3Output forward(torch::Tensor x);
};

TORCH_MODULE(InceptionV3_);




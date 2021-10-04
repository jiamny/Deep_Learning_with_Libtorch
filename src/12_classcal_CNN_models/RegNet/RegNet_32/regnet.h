#pragma once


#include <torch/torch.h>
#include <map>

struct SE_Impl : torch::nn::Module {

  torch::nn::Conv2d se1{nullptr}, se2{nullptr};

  SE_Impl(int64_t in_planes, int64_t se_planes);

  torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(SE_);

struct BlockRegImpl : torch::nn::Module {

  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};

  bool with_se{false};
  SE_ se{nullptr};
  torch::nn::Sequential shortcut{nullptr};
  bool useShortcut{false};

  BlockRegImpl(int64_t w_in, int64_t w_out, int64_t stride, int64_t group_width, double bottleneck_ratio, double se_ratio);

  torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(BlockReg);


struct RegNetImpl : torch::nn::Module {
  int64_t in_planes{64};
  std::map<std::string, std::vector<int64_t>> cfg;
  std::map<std::string, double> cfg2;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  std::vector<BlockReg> layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
  torch::nn::Linear linear{nullptr};

  std::vector<BlockReg> _make_layer(int64_t idx);

  explicit RegNetImpl(std::map<std::string, std::vector<int64_t>> cfg,
		  std::map<std::string, double> cfg2, int64_t num_classes);

  torch::Tensor forward(torch::Tensor X);
};

TORCH_MODULE(RegNet);

RegNet RegNetX_200MF(int64_t num_classes);
RegNet RegNetX_400MF(int64_t num_classes);
RegNet RegNetY_400MF(int64_t num_classes);




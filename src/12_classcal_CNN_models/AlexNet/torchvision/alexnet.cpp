#include "alexnet.h"

#include "../../modelsimpl.h"

AlexNetImpl::AlexNetImpl(int64_t num_classes) {
  features = torch::nn::Sequential(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 11).stride(4).padding(2)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 3, 2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 5).padding(2)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 3, 2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
      torch::nn::Functional(modelsimpl::relu_),
      torch::nn::Functional(modelsimpl::max_pool2d, 3, 2));

  classifier = torch::nn::Sequential(
      torch::nn::Dropout(),
      torch::nn::Linear(256 * 6 * 6, 4096),
      torch::nn::Functional(torch::relu),
      torch::nn::Dropout(),
      torch::nn::Linear(4096, 4096),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(4096, num_classes));

  register_module("features", features);
  register_module("classifier", classifier);
}

torch::Tensor AlexNetImpl::forward(torch::Tensor x) {
  x = features->forward(x);
  x = torch::adaptive_avg_pool2d(x, {6, 6});
  x = x.view({x.size(0), -1});
  x = classifier->forward(x);

  return x;
}

void AlexNetImpl::init(){
    this->apply(weights_init);
    torch::nn::init::constant_(*(this->features[3]->named_parameters(false).find("bias")), /*bias=*/1.0);
    torch::nn::init::constant_(*(this->features[6]->named_parameters(false).find("bias")), /*bias=*/1.0);
    torch::nn::init::constant_(*(this->features[10]->named_parameters(false).find("bias")), /*bias=*/1.0);
    return;
}

void weights_init(torch::nn::Module &m){
    if ((typeid(m) == typeid(torch::nn::Conv2d)) || (typeid(m) == typeid(torch::nn::Conv2dImpl))) {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(torch::nn::Linear)) || (typeid(m) == typeid(torch::nn::LinearImpl))){
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
        if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
    }
    return;
}

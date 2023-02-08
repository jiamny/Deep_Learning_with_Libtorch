
#include "mobilenetv1.h"

using Options = torch::nn::Conv2dOptions;


MobileBlockImpl::MobileBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride) {

    conv1 = torch::nn::Conv2d(Options(in_planes, in_planes, 3).stride(stride).padding(1).groups(in_planes).bias(false));
    bn1   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
    conv2 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
    bn2   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
}

torch::Tensor MobileBlockImpl::forward(torch::Tensor x){
	x = torch::relu(bn1->forward(conv1->forward(x)));
    x = torch::relu(bn2->forward(conv2->forward(x)));
    return x;
}

MobileNetV1Impl::MobileNetV1Impl(int64_t num_classes) {
	conv1 = torch::nn::Conv2d(Options(3, 32, 3).stride(1).padding(1).bias(false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
	layers = _make_layers(32);
	linear = torch::nn::Linear(1024, num_classes);

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layers", layers);
    register_module("linear", linear);
}

torch::nn::Sequential MobileNetV1Impl::_make_layers(int64_t in_planes) {
	torch::nn::Sequential tmp;

	for(auto setting : this->cfg) {
		int64_t out_planes = setting[0];
		int64_t stride = setting[1];
		tmp->push_back(MobileBlock(in_planes, out_planes, stride));
		in_planes = out_planes;
	}
    return tmp;
}

torch::Tensor MobileNetV1Impl::forward(torch::Tensor x) {
    x = torch::relu(bn1->forward(conv1->forward(x)));
//    for( auto layer : this->layers ) {
//    	out = layer->forward(out);
//    }
    x = layers->forward(x);
    x = torch::avg_pool2d(x, 2);
    x = x.view({x.size(0), -1});
    x = linear->forward(x);
    return x;
}

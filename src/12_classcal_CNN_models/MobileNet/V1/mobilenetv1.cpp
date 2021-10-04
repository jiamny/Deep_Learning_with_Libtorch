
#include "mobilenetv1.h"

using Options = torch::nn::Conv2dOptions;


MobileBlockImpl::MobileBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride) {

        this->conv1 = torch::nn::Conv2d(Options(in_planes, in_planes, 3).stride(stride).padding(1).groups(in_planes).bias(false));
        this->bn1   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
        this->conv2 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
        this->bn2   = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
}

torch::Tensor MobileBlockImpl::forward(torch::Tensor x){
	auto out = torch::relu(this->bn1->forward(this->conv1->forward(x)));
    out = torch::relu(this->bn2->forward(this->conv2->forward(out)));
    return out;
}

MobileNetV1Impl::MobileNetV1Impl(int64_t num_classes) {
	this->conv1 = torch::nn::Conv2d(Options(3, 32, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
	this->layers = _make_layers(32);
	this->linear->push_back( torch::nn::Linear(1024, num_classes));
}

std::vector<MobileBlock> MobileNetV1Impl::_make_layers(int64_t in_planes) {
	std::vector<MobileBlock> tmp;

	for (auto setting : this->cfg) {
		int64_t out_planes = setting[0];
		int64_t stride = setting[1];
		tmp.push_back(MobileBlock(in_planes, out_planes, stride));
		in_planes = out_planes;
	}
    return tmp;
}

torch::Tensor MobileNetV1Impl::forward(torch::Tensor x) {
	x = this->conv1->forward(x);
	x = this->bn1->forward(x);
    auto out = torch::relu(x);
    for( auto layer : this->layers ) {
    	out = layer->forward(out);
    }
//    out = self.layers(out);
    out = torch::avg_pool2d(out, 2);
    out = out.view({out.size(0), -1});
    out = this->linear->forward(out);
    return out;
}

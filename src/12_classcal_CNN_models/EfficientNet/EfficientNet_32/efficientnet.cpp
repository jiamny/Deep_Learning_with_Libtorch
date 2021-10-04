
#include "efficientnet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

torch::Tensor swish(torch::Tensor x) {
	//x * x.sigmoid()
	return (x * x.sigmoid());
}

torch::Tensor drop_connect(torch::Tensor x, double drop_ratio) {
	auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
	double keep_ratio = 1.0 - drop_ratio;
	torch::Tensor mask = torch::empty({x.sizes()[0], 1, 1, 1}, options); //, x.dtype, x.device);
	mask.bernoulli_(keep_ratio);
	x.div_(keep_ratio);
	x.mul_(mask);
	return x;
}

SEImpl::SEImpl(int64_t in_planes, int64_t se_planes) {
	 this->se1 = torch::nn::Conv2d(Options(in_planes, se_planes, 1).bias(true));
	 this->se2 = torch::nn::Conv2d(Options(se_planes, in_planes, 1).bias(true));
}

torch::Tensor SEImpl::forward(torch::Tensor x) {
   auto  out = torch::adaptive_avg_pool2d(x, {1, 1});
   out = swish(se1->forward(out));
   out = se2->forward(out).sigmoid();
   out = x * out;
   return out;
}

Block_Impl::Block_Impl( int64_t in_planes,
    int64_t out_planes,
    int64_t kernel_size,
    int64_t stride,
    int64_t expand_ratio,
    double se_ratio,
    double drop_rate) {

	this->stride = stride;
	this->drop_rate = drop_rate;
	this->expand_ratio = expand_ratio;
	this->se_ratio = se_ratio;
	this->drop_rate = drop_rate;

	//Expansion
	int64_t planes = expand_ratio * in_planes;

	this->conv1 = torch::nn::Conv2d(Options(in_planes, planes, 1).stride(1).padding(0).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	//Depthwise conv
	this->conv2 = torch::nn::Conv2d(Options(planes, planes, kernel_size).stride(stride).padding( (kernel_size == 3) ? 1 : 2).groups(planes).bias(false));
	this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	//SE layers
	int64_t se_planes = int(in_planes * se_ratio);
	this->se = SE(planes, se_planes);

	//Output
	this->conv3 = torch::nn::Conv2d(Options(planes, out_planes, 1).stride(1).padding(0).bias(false));
	this->bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));

	 // Skip connection if in and out shapes are the same (MV-V2 style)
	 this->has_skip = ((stride == 1) && (in_planes == out_planes));

}

torch::Tensor Block_Impl::forward(torch::Tensor x) {
    auto out = (this->expand_ratio == 1 ) ? x : swish(bn1->forward(conv1->forward(x)));
	out = swish(bn2->forward(conv2->forward(out)));
	out = se->forward(out);
	out = bn3->forward(conv3->forward(out));
	if( has_skip ) {
		if( training &&  drop_rate > 0 )
			out = drop_connect(out, drop_rate);
		out = out + x; // gives you a new tensor with summation ran over out and x == out.add(x)
	}

	return out;
}

EfficientNetImpl::EfficientNetImpl(std::map<std::string, std::vector<int64_t>> cfg, int64_t num_classes) {
	this->cfg = cfg;
	std::vector<int64_t> out_planes = cfg.at("out_planes");
	this->conv1 = torch::nn::Conv2d(Options(3, 32, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
	this->layers = _make_layers(32);
	this->linear->push_back(torch::nn::Linear(out_planes[out_planes.size()-1], num_classes));
}

std::vector<Block_>  EfficientNetImpl::_make_layers(int64_t in_planes) {
	std::vector<Block_> layers;
	std::vector<int64_t> expansion = cfg.at("expansion");
	std::vector<int64_t> out_planes = cfg.at("out_planes");
	std::vector<int64_t> num_blocks = cfg.at("num_blocks");
	std::vector<int64_t> kernel_size = cfg.at("kernel_size");
	std::vector<int64_t> stride = cfg.at("stride");


	for( int j = 0; j < expansion.size(); j++ ) {
		std::vector<int64_t> strides;
		strides.push_back(stride[j]);

		for( int i = 0; i < (num_blocks[j]-1); i++ ) strides.push_back(1);

		for( int s = 0; s < strides.size(); s++ ) {
			layers.push_back(Block_(in_planes,
                    out_planes[j],
                    kernel_size[j],
                    strides[s],
                    expansion[j],
                    0.25,
                    0));
			in_planes = out_planes[j];
		}
	}
	return layers;
}

torch::Tensor EfficientNetImpl::forward(torch::Tensor x) {
	auto out = swish(bn1->forward(conv1->forward(x)));
//	std::cout << "swish -> " << out.sizes() << std::endl;
    //out = self.layers(out)
	for( int i =0; i < layers.size(); i++ ) {
		out = layers[i]->forward(out);
//		std::cout << "layer " << i << " -> " << out.sizes() << std::endl;
	}
    out =  torch::nn::functional::adaptive_avg_pool2d(out, torch::nn::AdaptiveAvgPool2dOptions(1));
//    std::cout << "pool2d -> " << out.sizes() << std::endl;
    out = out.view({out.size(0), -1});
    out = linear->forward(out);
    return out;
}

EfficientNet EfficientNetB0(int64_t num_classes) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("num_blocks", {1, 2, 2, 3, 3, 4, 1}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("expansion", {1, 6, 6, 6, 6, 6, 6}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("out_planes", {16, 24, 40, 80, 112, 192, 320}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("kernel_size", {3, 3, 5, 3, 5, 5, 3}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("stride", {1, 2, 2, 2, 1, 2, 1}));
	return EfficientNet(cfg, num_classes);
}

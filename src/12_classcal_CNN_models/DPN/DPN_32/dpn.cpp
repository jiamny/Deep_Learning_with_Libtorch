#include "dpn.h"
#include <torch/script.h>

using Options = torch::nn::Conv2dOptions;
using torch::indexing::Slice;
using torch::indexing::None;

BottleneckImpl::BottleneckImpl(int64_t last_planes, int64_t in_planes, int64_t out_planes,
		int64_t dense_depth, int64_t stride, bool first_layer) {
	this->out_planes = out_planes;
	this->dense_depth = dense_depth;
	this->first_layer = first_layer;

	this->conv1 = torch::nn::Conv2d(Options(last_planes, in_planes, 1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
	this->conv2 = torch::nn::Conv2d(Options(in_planes, in_planes, 3).stride(stride).padding(1).groups(32).bias(false));
	this->bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));
	this->conv3 = torch::nn::Conv2d(Options(in_planes, out_planes+dense_depth, 1).bias(false));
	this->bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes+dense_depth));

	if( first_layer ) {
		this->shortcut->push_back(torch::nn::Conv2d(Options(last_planes, out_planes+dense_depth, 1).stride(stride).bias(false)));
		this->shortcut->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes+dense_depth)));
	}
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
	 auto out = torch::relu(bn1->forward(conv1->forward(x)));
	 out = torch::relu(bn2->forward(conv2->forward(out)));
	 out = bn3->forward(conv3->forward(out));

	 if( this->first_layer ) x = shortcut->forward(x);

	 int64_t d = this->out_planes;
	 auto _x1 = x.index({Slice(), Slice(None, d), Slice(), Slice()}).clone();
	 auto x1_ = x.index({Slice(), Slice(d, None), Slice(), Slice()}).clone();
//	 std::cout <<"B==>Bx\n" << _x1.sizes() << std::endl;
	 auto _x2 = out.index({Slice(), Slice(None, d), Slice(), Slice()}).clone();
	 auto x2_ = out.index({Slice(), Slice(d, None), Slice(), Slice()}).clone();
//	 std::cout <<"B==>Bout\n" << _x2.sizes() << std::endl;
//	 std::cout <<"B==>x+out\n" << torch::add(_x1, _x2).clone().sizes() << std::endl;
	 // out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
	 out = torch::cat({torch::add(_x1, _x2).clone(), x1_, x2_}, 1);
//	 std::cout <<"B==>out\n" << out.sizes() << std::endl;
	 out = torch::relu(out);
	 return out;
}

DPNImpl::DPNImpl(std::map<std::string, std::vector<int64_t>> cfg, int64_t num_classes) {
	this->in_planes = cfg.at("in_planes");
	this->out_planes = cfg.at("out_planes");
	this->num_blocks = cfg.at("num_blocks");
	this->dense_depth = cfg.at("dense_depth");

	this->conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));
	this->bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
	this->last_planes = 64;
	this->layer1 = _make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], 1);
	this->layer2 = _make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], 2);
	this->layer3 = _make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], 2);
	this->layer4 = _make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], 2);
	this->linear->push_back(torch::nn::Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], num_classes));
}

std::vector<Bottleneck> DPNImpl::_make_layer(int64_t in_planes, int64_t out_planes,
			int64_t num_blocks, int64_t dense_depth, int64_t stride) {
	std::vector<Bottleneck> layers;
	std::vector<int64_t> strides;
	strides.push_back(stride);

	for( int i = 0; i < (num_blocks-1); i++ ) strides.push_back(1);

	// [stride] + [1]*(num_blocks-1)
	for(int64_t j = 0; j < strides.size(); j++ )  {
		layers.push_back(Bottleneck(last_planes, in_planes, out_planes, dense_depth, strides[j], j==0));
		last_planes = out_planes + (j+2) * dense_depth;
	}
	//for i,stride in enumerate(strides):
	return layers;
}

torch::Tensor DPNImpl::forward(torch::Tensor x) {
	auto out = torch::relu(bn1->forward(conv1->forward(x)));
//	std::cout <<"DNP==>out\n" << out.sizes() << std::endl;
	for( auto layer : this->layer1 ) {
	    out = layer->forward(out);
	}
//	std::cout <<"DNP==>ly1\n" << out.sizes() << std::endl;
	for( auto layer : this->layer2 ) {
		out = layer->forward(out);
	}
//	std::cout <<"DNP==>ly2\n" << out.sizes() << std::endl;
	for( auto layer : this->layer3 ) {
		out = layer->forward(out);
	}
//	std::cout <<"DNP==>ly3\n" << out.sizes() << std::endl;
	for( auto layer : this->layer4 ) {
		out = layer->forward(out);
	}
//	std::cout <<"DNP==>ly4\n" << out.sizes() << std::endl;

	out = torch::avg_pool2d(out, 4);
	out = out.view({out.size(0), -1});
	out = linear->forward(out);
	return out;
}

DPN DPN26(int64_t num_classes) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("in_planes", {96,192,384,768}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("out_planes", {256,512,1024,2048}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("num_blocks", {2,2,2,2}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("dense_depth", {16,32,24,128}));
	return DPN(cfg, num_classes);
}

DPN DPN92(int64_t num_classes) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("in_planes", {96,192,384,768}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("out_planes", {256,512,1024,2048}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("num_blocks", {3,4,20,3}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("dense_depth", {16,32,24,128}));
	return DPN(cfg, num_classes);
}


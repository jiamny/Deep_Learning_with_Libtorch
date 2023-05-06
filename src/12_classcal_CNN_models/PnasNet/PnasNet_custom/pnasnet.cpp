
#include "pnasnet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

SepConvImpl::SepConvImpl(int in_planes, int64_t out_planes, int64_t kernel_size, int64_t stride) {

	conv1 = torch::nn::Conv2d(Options(in_planes, out_planes,
	    	                               kernel_size).stride(stride)
										   .padding(static_cast<int>(std::floor((kernel_size-1)/2)))
	    	                               .bias(false)
										   .groups(in_planes));

	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
}

torch::Tensor SepConvImpl::forward(torch::Tensor x){
	return bn1->forward(conv1->forward(x));
}

CellAImpl::CellAImpl(int64_t in_planes, int64_t out_planes, int64_t stride ) {
	stride = stride;
	sep_conv1 = SepConv(in_planes, out_planes, 7, stride);

	if( stride==2 ){
		conv1 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
		bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	}
}

torch::Tensor CellAImpl::forward(torch::Tensor x){
	auto y1 = sep_conv1->forward(x);
	auto y2 = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(3).stride(stride).padding(1));
	if( this->stride==2 )
		y2 = bn1->forward(conv1->forward(y2));
	return torch::relu(y1+y2);
}

CellBImpl::CellBImpl(int64_t in_planes, int64_t out_planes, int64_t stride ){
	stride = stride;

	//Left branch
	sep_conv1 = SepConv(in_planes, out_planes, 7, stride);
	sep_conv2 = SepConv(in_planes, out_planes, 3, stride);

    //Right branch
	sep_conv3 = SepConv(in_planes, out_planes, 5, stride);

	if( stride==2 ) {
		conv1 = torch::nn::Conv2d(Options(in_planes, out_planes, 1).stride(1).padding(0).bias(false));
	    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
	}

	//Reduce channels
	conv2 = torch::nn::Conv2d(Options(2*out_planes, out_planes, 1).stride(1).padding(0).bias(false));
	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_planes));
}

torch::Tensor CellBImpl::forward(torch::Tensor x){
	//Left branch
	auto y1 = sep_conv1->forward(x);
	auto y2 = sep_conv2->forward(x);

	//Right branch
	auto y3 = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(3).stride(stride).padding(1));

	if( stride==2 ) {
		y3 = bn1->forward(conv1->forward(y3));
	}

	auto y4 = sep_conv3->forward(x);
	// Concat & reduce channels
	auto b1 = torch::relu(y1+y2);
	auto b2 = torch::relu(y3+y4);
    auto y = torch::cat({b1,b2}, 1);

    auto z = torch::relu(bn2->forward(conv2->forward(y)));
    return z;
}


PNASNetImpl::PNASNetImpl(std::string cell_tp, int64_t num_cells, int64_t num_pls, int64_t num_classes) {
	in_planes = num_pls;
    cell_type = cell_tp;
    std::cout << cell_type << std::endl;

	conv1 = torch::nn::Conv2d(Options(3, in_planes, 3).stride(1).padding(1).bias(false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));

	if( cell_type == "CellA" ) {
		a_layer1 = a_make_layer(num_pls, num_cells);
		a_layer2 = a_downsample(num_pls*2);
		a_layer3 = a_make_layer(num_pls*2, num_cells);
		a_layer4 = a_downsample(num_pls*4);
		a_layer5 = a_make_layer(num_pls*4, num_cells);
	}

	if( cell_type == "CellB" ) {

		b_layer1 = b_make_layer(num_pls, num_cells);
		b_layer2 = b_downsample(num_pls*2);
		b_layer3 = b_make_layer(num_pls*2, num_cells);
		b_layer4 = b_downsample(num_pls*4);
		b_layer5 = b_make_layer(num_pls*4, num_cells);
	}

	linear->push_back(torch::nn::Linear(num_pls*4, num_classes));
}

CellA PNASNetImpl::a_downsample(int64_t planes) {
	auto layer = CellA(in_planes, planes, 2);
	in_planes = planes;
	return layer;
}

CellB PNASNetImpl::b_downsample(int64_t planes) {
	auto layer = CellB(in_planes, planes, 2);
	in_planes = planes;
	return layer;
}

std::vector<CellA> PNASNetImpl::a_make_layer(int64_t planes, int64_t num_cells) {
	std::vector<CellA> layers;

	for( int i = 0; i < num_cells; i++ ) {
		layers.push_back(CellA( in_planes, planes, 2));
	    //this->in_planes = planes;
	}

	return layers;
}

std::vector<CellB> PNASNetImpl::b_make_layer(int64_t planes, int64_t num_cells) {
	std::vector<CellB> layers;

	for( int i = 0; i < num_cells; i++ ) {
		layers.push_back(CellB(this->in_planes, planes, 2));
	    this->in_planes = planes;
	}

	return layers;
}

torch::Tensor PNASNetImpl::forward(torch::Tensor x) {

	auto out = torch::relu(bn1->forward(conv1->forward(x)));

	if( cell_type == "CellA" ) {
		for( int i =0; i < a_layer1.size(); i++ )
			out = a_layer1[i]->forward(out);

		out = a_layer2->forward(out);

		for( int i =0; i < a_layer3.size(); i++ )
			out = a_layer3[i]->forward(out);

		out = a_layer4->forward(out);

		for( int i =0; i < a_layer5.size(); i++ )
			out = a_layer5[i]->forward(out);
	}

	if( cell_type == "CellB" ) {
		for( int i =0; i < b_layer1.size(); i++ )
			out = b_layer1[i]->forward(out);

		out = b_layer2->forward(out);

		for( int i =0; i < b_layer3.size(); i++ ) {
			out = b_layer3[i]->forward(out);
		}

		out = b_layer4->forward(out);

		for( int i =0; i < b_layer5.size(); i++ ) {
			out = b_layer5[i]->forward(out);
		}
	}

	out = torch::nn::functional::avg_pool2d(out, torch::nn::functional::AvgPool2dFuncOptions(4));
	out = linear->forward(out.view({out.size(0), -1}));

	return out;
}

PNASNet PNASNetA(std::string cell_type, int64_t num_cells, int64_t num_planes, int64_t num_classes) {
	return PNASNet(cell_type, num_cells, num_planes, num_classes);
}

PNASNet PNASNetB(std::string cell_type, int64_t num_cells, int64_t num_planes, int64_t num_classes) {
	std::cout << "cell_type -> " << cell_type << std::endl;
	return PNASNet(cell_type, num_cells, num_planes, num_classes);
}

#include "regnet.h"

using Options = torch::nn::Conv2dOptions;

SE_Impl::SE_Impl(int64_t in_planes, int64_t se_planes) {

	se1 = torch::nn::Conv2d(Options(in_planes, se_planes, 1).bias(true));
	se2 = torch::nn::Conv2d(Options(se_planes, in_planes, 1).bias(true));
}

torch::Tensor SE_Impl::forward(torch::Tensor x) {
	at::Tensor kp(x.clone());

	x = torch::adaptive_avg_pool2d(x, {1, 1});
	x = torch::relu(se1->forward(x));
	x = se2->forward(x).sigmoid();
	x = kp * x;

    return x;
}

BlockRegImpl::BlockRegImpl(int64_t w_in, int64_t w_out, int64_t stride, int64_t group_width, double bottleneck_ratio, double se_ratio) {

		//# 1x1
        int64_t w_b = static_cast<int64_t>(round(w_out * bottleneck_ratio));
        conv1 = torch::nn::Conv2d(Options(w_in, w_b, 1).bias(false));
        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(w_b));

		//# 3x3
        int64_t num_groups = w_b; // group_width
        conv2 = torch::nn::Conv2d(Options(w_b, w_b, 3)
                               .stride(stride)
							   .padding(1)
							   .groups(num_groups)
							   .bias(false));

        bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(w_b));

		//# se
        with_se = (se_ratio > 0) ? true : false;

        if( with_se ) {
            int64_t w_se = static_cast<int64_t>(round(w_in * se_ratio));
            se = SE_(w_b, w_se);
        }

		//# 1x1
		conv3 = torch::nn::Conv2d(Options(w_b, w_out, 1).bias(false));
        bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(w_out));

        if( stride != 1 || w_in != w_out ) {
            shortcut = torch::nn::Sequential(
            		torch::nn::Conv2d(Options(w_in, w_out, 1).stride(stride).bias(false)),
					torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(w_out))
            );
            useShortcut = true;
        }
}

torch::Tensor BlockRegImpl::forward(torch::Tensor x) {

	at::Tensor kp(x.clone());

	x = torch::relu(bn1->forward(conv1->forward(x)));
	x = torch::relu(bn2->forward(conv2->forward(x)));

	if( with_se )
	     x = se->forward(x);

	x = bn3->forward(conv3->forward(x));

	if( useShortcut )
		x += shortcut->forward(kp);
	else
		x += kp;

  return x.relu_(); // out = F.relu(out)
}

RegNetImpl::RegNetImpl(std::map<std::string, std::vector<int64_t>> cfg_, std::map<std::string, double> cfg2_,
						int64_t num_classes) {
	cfg = cfg_;
	cfg2 = cfg2_;
	in_planes = 64;
	std::vector<int64_t> width = cfg.at("widths");
	//std::cout << width.size() << std::endl;

	conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));
	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));

	layer1 = _make_layer(0);
	layer2 = _make_layer(1);
	layer3 = _make_layer(2);
	layer4 = _make_layer(3);
	linear = torch::nn::Linear(width[width.size()-1], num_classes);
}

torch::nn::Sequential RegNetImpl::_make_layer(int64_t idx) {

	int64_t depth = cfg.at("depths")[idx];
	int64_t width = cfg.at("widths")[idx];
	int64_t stride = cfg.at("strides")[idx];

    int64_t group_width = static_cast<int64_t>(cfg2.at("group_width"));
    double bottleneck_ratio = cfg2.at("bottleneck_ratio");
    double se_ratio = cfg2.at("se_ratio");

    torch::nn::Sequential layers;

    for( int i = 0; i < depth; i++ ){
        int64_t s = stride;
        if( i != 0 ) s = 1;

        layers->push_back(BlockReg(in_planes, width, s, group_width, bottleneck_ratio, se_ratio));
        in_planes = width;
    }

	return layers;
}

torch::Tensor RegNetImpl::forward(torch::Tensor x) {

// out = self.conv1(x)
	x = bn1->forward(conv1->forward(x)).relu_();

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

/*
//	std::cout << out.sizes() << std::endl;

	for( int i =0; i < layer1.size(); i++ ) {
			out = layer1[i]->forward(out);
//			std::cout << "layer1 - " << i << " >> " << out.sizes() << std::endl;
	}

	for( int i =0; i < layer2.size(); i++ )
		out = layer2[i]->forward(out);

	for( int i =0; i < layer3.size(); i++ )
		out = layer3[i]->forward(out);

	for( int i =0; i < layer4.size(); i++ )
		out = layer4[i]->forward(out);
*/
	// out = F.adaptive_avg_pool2d(out, (1, 1))
    // out = out.view(out.size(0), -1)
    // out = self.linear(out)
	x = torch::adaptive_avg_pool2d(x, {1, 1});
	x = linear->forward(x.view({x.size(0), -1}));

	return x;
}

RegNet RegNetX_200MF(int64_t num_classes) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	std::map<std::string, double>  cfg2;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("depths", {1, 1, 4, 7}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("widths", {24, 56, 152, 368}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("strides", {1, 1, 2, 2}));
	cfg2.insert(std::pair<std::string,double>("group_width", 8));
	cfg2.insert(std::pair<std::string,double>("bottleneck_ratio", 1));
	cfg2.insert(std::pair<std::string,double>("se_ratio", 0));
	return RegNet(cfg,  cfg2, num_classes);
}

RegNet RegNetX_400MF(int64_t num_classes) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	std::map<std::string, double>  cfg2;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("depths", {1, 2, 7, 12}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("widths", {32, 64, 160, 384}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("strides", {1, 1, 2, 2}));
	cfg2.insert(std::pair<std::string,double>("group_width", 16));
	cfg2.insert(std::pair<std::string,double>("bottleneck_ratio", 1));
	cfg2.insert(std::pair<std::string,double>("se_ratio", 0));
	return RegNet(cfg, cfg2, num_classes);
}

RegNet RegNetY_400MF(int64_t num_classes) {
	std::map<std::string, std::vector<int64_t>>  cfg;
	std::map<std::string, double>  cfg2;
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("depths", {1, 2, 7, 12}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("widths", {32, 64, 160, 384}));
	cfg.insert(std::pair<std::string,std::vector<int64_t>>("strides", {1, 1, 2, 2}));
	cfg2.insert(std::pair<std::string,double>("group_width", 16));
	cfg2.insert(std::pair<std::string,double>("bottleneck_ratio", 1));
	cfg2.insert(std::pair<std::string,double>("se_ratio", 0.25));
	return RegNet(cfg, cfg2, num_classes);
}




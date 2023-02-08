
#include "PreactResNet.h"
#include <torch/torch.h>

using Options = torch::nn::Conv2dOptions;

PreActBlockImpl::PreActBlockImpl(int64_t in_planes, int64_t planes, int64_t stride) {

	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));

	conv1 = torch::nn::Conv2d(Options(in_planes, planes, 3).stride(stride)
										   .padding(1)
	    	                               .bias(false));

	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	conv2 = torch::nn::Conv2d(Options(planes, planes, 3).stride(1)
											   .padding(1)
		    	                               .bias(false));

//	        self.bn1 = nn.BatchNorm2d(in_planes)
//	        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
//	        self.bn2 = nn.BatchNorm2d(planes)
//	        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

	if(stride != 1 || in_planes != expansion*planes ) {
	    shortcut = torch::nn::Sequential(
	    		torch::nn::Conv2d(Options(in_planes, expansion*planes, 1).stride(stride).bias(false)));
	    hasShortcut = true;
	    register_module("shortcut", shortcut);
	}

	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("bn1", bn1);
	register_module("bn2", bn2);
}

torch::Tensor PreActBlockImpl::forward(torch::Tensor x) {
	auto out = torch::relu(bn1->forward(x));
	auto shtc = hasShortcut ? shortcut->forward(out) : x;
	out = conv1->forward(out);
	out = conv2->forward(torch::relu(bn2->forward(out)));
	out += shtc;
	return out;
}

PreActBottleneckImpl::PreActBottleneckImpl(int64_t in_planes, int64_t planes, int64_t stride) {

	bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_planes));

	conv1 = torch::nn::Conv2d(Options(in_planes, planes, 1).bias(false));

	bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	conv2 = torch::nn::Conv2d(Options(planes, planes, 3).stride(stride).padding(1).bias(false));

	bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));

	conv3 = torch::nn::Conv2d(Options(planes, expansion*planes, 1).bias(false));

	//self.bn1 = nn.BatchNorm2d(in_planes)
	//        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
	//        self.bn2 = nn.BatchNorm2d(planes)
	//       self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
	//        self.bn3 = nn.BatchNorm2d(planes)
	//        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

	if( stride != 1 || in_planes != expansion*planes ) {
		shortcut = torch::nn::Sequential(
				torch::nn::Conv2d(Options(in_planes, expansion*planes, 1).stride(stride).bias(false)));
		hasShortcut = true;
		register_module("shortcut", shortcut);
	}

	register_module("conv1", conv1);
	register_module("conv2", conv2);
	register_module("bn1", bn1);
	register_module("bn2", bn2);
	register_module("conv3", conv3);
	register_module("bn3", bn3);
}

torch::Tensor PreActBottleneckImpl::forward(torch::Tensor x) {
	auto out = torch::relu(bn1->forward(x));
	auto shtc = hasShortcut ? shortcut->forward(out) : x;
	out = conv1->forward(out);
	out = conv2->forward(torch::relu(bn2->forward(out)));
	out = conv3->forward(torch::relu(bn3->forward(out)));
	out += shtc;
	return out;
}

PreActResNetBBImpl::PreActResNetBBImpl(std::vector<int64_t> num_blocks, int64_t num_classes) {

	conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));

	layer1 = _make_layer(64, num_blocks[0], 1);
	layer2 = _make_layer(128, num_blocks[1], 2);
	layer3 = _make_layer(256, num_blocks[2], 2);
	layer4 = _make_layer(512, num_blocks[3], 2);

	//self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	//        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
	//        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
	//        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
	//        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
	//        self.linear = nn.Linear(512*block.expansion, num_classes)

	linear = torch::nn::Linear(512*expansion, num_classes);

	register_module("conv1", conv1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("linear", linear);
}


torch::nn::Sequential PreActResNetBBImpl::_make_layer(int64_t planes, int64_t num_blocks, int64_t stride) {

	std::vector<int64_t> strides;
	strides.push_back(stride);

	for( int i = 0; i < (num_blocks-1); i++ ) //[1]*(num_blocks-1)
		strides.push_back(1);

	// for stride in strides:
	// layers.append(block(self.in_planes, planes, stride))
	// self.in_planes = planes * block.expansion
	torch::nn::Sequential layers;

	for( int i = 0; i < strides.size(); i++ ) {
		layers->push_back(PreActBlock(in_planes, planes, strides[i]));
	    in_planes = planes*expansion;
	}

	return layers;
}

torch::Tensor PreActResNetBBImpl::forward(torch::Tensor x) {
// out = self.conv1(x)
	x = conv1->forward(x);
//	std::cout << out.sizes() << std::endl;
	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);
/*
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
	// out = F.avg_pool2d(out, 4)
    // out = out.view(out.size(0), -1)
    // out = self.linear(out)
	x = torch::nn::functional::avg_pool2d(x, torch::nn::functional::AvgPool2dFuncOptions(4));
	x = linear->forward(x.view({x.size(0), -1}));

	return x;
}


PreActResNetBNImpl::PreActResNetBNImpl(std::vector<int64_t> num_blocks, int64_t num_classes) {

	conv1 = torch::nn::Conv2d(Options(3, 64, 3).stride(1).padding(1).bias(false));

	layer1 = _make_layer(64, num_blocks[0], 1);
	layer2 = _make_layer(128, num_blocks[1], 2);
	layer3 = _make_layer(256, num_blocks[2], 2);
	layer4 = _make_layer(512, num_blocks[3], 2);

	//self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
	//        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
	//        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
	//        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
	//        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
	//        self.linear = nn.Linear(512*block.expansion, num_classes)

	linear = torch::nn::Linear(512*expansion, num_classes);

	register_module("conv1", conv1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("linear", linear);
}


torch::nn::Sequential PreActResNetBNImpl::_make_layer(int64_t planes, int64_t num_blocks, int64_t stride) {

	std::vector<int64_t> strides;
	strides.push_back(stride);

	for( int i = 0; i < (num_blocks-1); i++ ) //[1]*(num_blocks-1)
		strides.push_back(1);

	// for stride in strides:
	// layers.append(block(self.in_planes, planes, stride))
	// self.in_planes = planes * block.expansion
	torch::nn::Sequential layers;

	for( int i = 0; i < strides.size(); i++ ) {
		layers->push_back(PreActBottleneck(in_planes, planes, strides[i]));
	    in_planes = planes*expansion;
	}

	return layers;
}

torch::Tensor PreActResNetBNImpl::forward(torch::Tensor x) {
// out = self.conv1(x)
	x = conv1->forward(x);
//	std::cout << out.sizes() << std::endl;
	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

/*
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
	// out = F.avg_pool2d(out, 4)
    // out = out.view(out.size(0), -1)
    // out = self.linear(out)
	x = torch::nn::functional::avg_pool2d(x, torch::nn::functional::AvgPool2dFuncOptions(4));
	x = linear->forward(x.view({x.size(0), -1}));

	return x;
}

PreActResNetBB PreActResNet18(int64_t num_classes){
	std::vector<int64_t> array = {2,2,2,2};
    return PreActResNetBB(array, num_classes);
}

PreActResNetBB PreActResNet34(int64_t num_classes){
	std::vector<int64_t> array = {3,4,6,3};
    return PreActResNetBB(array, num_classes);
}

PreActResNetBN PreActResNet50(int64_t num_classes){
	std::vector<int64_t> array = {3,4,6,3};
    return PreActResNetBN(array, num_classes);
}

PreActResNetBN PreActResNet101(int64_t num_classes){
	std::vector<int64_t> array = {3,4,23,3};
    return PreActResNetBN(array, num_classes);
}

PreActResNetBN PreActResNet152(int64_t num_classes){
	std::vector<int64_t> array = {3,8,36,3};
    return PreActResNetBN(array, num_classes);
}


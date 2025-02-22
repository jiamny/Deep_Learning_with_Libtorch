#include "inception.h"

using Options = torch::nn::Conv2dOptions;

torch::nn::Sequential ConvBNReLU(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding){
    return torch::nn::Sequential(
        torch::nn::Conv2d(Options(in_channels, out_channels, kernel_size).stride(stride).padding(padding)),
        torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels).eps(0.001)),
        torch::nn::ReLU6(true));
}

torch::nn::Sequential ConvBNReLUFactorization(int64_t in_channels, int64_t out_channels,
		const std::vector<int64_t> &kernel_sizes, const std::vector<int64_t> &paddings){
    return torch::nn::Sequential(
        torch::nn::Conv2d(Options(in_channels, out_channels, kernel_sizes).stride(1).padding(paddings)),
        torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels).eps(0.001)),
		torch::nn::ReLU6(true),
		torch::nn::Conv2d(Options(out_channels, out_channels, kernel_sizes).stride(1).padding(paddings)),
		torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels).eps(0.001)),
		torch::nn::ReLU6(true));
}


InceptionV3ModuleAImpl::InceptionV3ModuleAImpl(int64_t in_channels, int64_t out_channels1, int64_t out_channels2reduce, int64_t  out_channels2,
		  int64_t out_channels3reduce, int64_t  out_channels3, int64_t out_channels4) {

	branch1 = ConvBNReLU(in_channels, out_channels1, 1, 1, 0);

	// ConvBNReLU(in_channels, out_channels2reduce, 1)
	branch2 = ConvBNReLU(in_channels, out_channels2reduce, 1, 1, 0);

	//  ConvBNReLU(out_channels2reduce, out_channels2, =5, padding=2)
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2, 5).stride(1).padding(2)));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	branch3 = ConvBNReLU(in_channels, out_channels3reduce, 1, 1, 0);
	// ConvBNReLU(in_channels=, out_channels=out_channels3, kernel_size=3, padding=1)
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3, 3).stride(1).padding(1)));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	// ConvBNReLU(in_channels=, out_channels=out_channels3, kernel_size=3, padding=1)
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3, out_channels3, 3).stride(1).padding(1)));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));

	// nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
	branch4->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
	// ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)
	branch4->push_back(torch::nn::Conv2d(Options(in_channels, out_channels4, 1).stride(1).padding(0)));
	branch4->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels4).eps(0.001)));
	branch4->push_back(torch::nn::ReLU6(true));

	register_module("A_branch1", branch1);
	register_module("A_branch2", branch2);
	register_module("A_branch3", branch3);
	register_module("A_branch4", branch4);
}

torch::Tensor InceptionV3ModuleAImpl::forward(const torch::Tensor& x) {
	torch::Tensor b1 = branch1->forward(x);
	torch::Tensor b2 = branch2->forward(x);
	torch::Tensor b3 = branch3->forward(x);
	torch::Tensor b4 = branch4->forward(x);

	return torch::cat({b1, b2, b3, b4}, 1);
}


InceptionV3ModuleBImpl::InceptionV3ModuleBImpl( int64_t in_channels, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2,
			 int64_t out_channels3reduce, int64_t out_channels3, int64_t out_channels4) {

	//ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)
	branch1 = ConvBNReLU(in_channels, out_channels1, 1, 1, 0);

	// ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
	branch2 = ConvBNReLU(in_channels, out_channels2reduce, 1, 1, 0);

	// ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1,7],paddings=[0,3])
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2reduce, {1,7}).stride(1).padding({0, 3})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2reduce).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2reduce, {1,7}).stride(1).padding({0,3})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2reduce).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	// ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[7,1],paddings=[3, 0])
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2, {7,1}).stride(1).padding({3, 0})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2, out_channels2, {7,1}).stride(1).padding({3,0})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	branch3 = ConvBNReLU(in_channels, out_channels3reduce, 1, 1, 0);
	// ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[7, 1], paddings=[3, 0]),
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3reduce, {7,1}).stride(1).padding({3, 0})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3reduce).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3reduce, {7,1}).stride(1).padding({3, 0})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3reduce).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	// ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[1, 7], paddings=[0, 3])
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3reduce, {1,7}).stride(1).padding({0, 3})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3reduce).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3reduce, {1,7}).stride(1).padding({0, 3})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3reduce).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	// ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=[7, 1], paddings=[3, 0])
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3reduce, {7,1}).stride(1).padding({3, 0})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3reduce).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3reduce, {7,1}).stride(1).padding({3, 0})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3reduce).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));

	// ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3,kernel_sizes=[1, 7], paddings=[0, 3])
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3reduce, out_channels3, {1,7}).stride(1).padding({0, 3})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));
	branch3->push_back(torch::nn::Conv2d(Options(out_channels3, out_channels3, {1,7}).stride(1).padding({0, 3})));
	branch3->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels3).eps(0.001)));
	branch3->push_back(torch::nn::ReLU6(true));

	// nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
	branch4->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
	// ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)
	branch4->push_back(torch::nn::Conv2d(Options(in_channels, out_channels4, 1).stride(1).padding(0)));
	branch4->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels4).eps(0.001)));
	branch4->push_back(torch::nn::ReLU6(true));

	register_module("B_branch1", branch1);
	register_module("B_branch2", branch2);
	register_module("B_branch3", branch3);
	register_module("B_branch4", branch4);
}

torch::Tensor InceptionV3ModuleBImpl::forward(const torch::Tensor& x) {
	torch::Tensor b1 = branch1->forward(x);
	torch::Tensor b2 = branch2->forward(x);
	torch::Tensor b3 = branch3->forward(x);
	torch::Tensor b4 = branch4->forward(x);

//  std::cout <<"--->branch1\n" << branch1.sizes() << std::endl;
//  std::cout <<"--->branch2\n" << branch2.sizes() << std::endl;
//  std::cout <<"--->branch3\n" << branch3.sizes() << std::endl;
//  std::cout <<"--->branch4\n" << branch4.sizes() << std::endl;
//  auto out = torch::cat({branch1, branch2, branch3, branch4}, 1);
//  std::cout <<"B--->out\n" << out.sizes() << std::endl;
  return torch::cat({b1, b2, b3, b4}, 1);
}

InceptionV3ModuleCImpl::InceptionV3ModuleCImpl( int64_t in_channels, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2,
			 int64_t out_channels3reduce, int64_t out_channels3, int64_t out_channels4) {

	branch1 = ConvBNReLU(in_channels, out_channels1, 1, 1, 0);

	branch2_conv1 = ConvBNReLU(in_channels, out_channels2reduce, 1, 1, 0);
	branch2_conv2a = ConvBNReLUFactorization(out_channels2reduce, out_channels2, {1,3}, {0,1});

	branch2_conv2b = ConvBNReLUFactorization(out_channels2reduce, out_channels2, {3,1}, {1, 0});

	branch3_conv1 = ConvBNReLU(in_channels, out_channels3reduce, 1, 1, 0);
	branch3_conv2 = ConvBNReLU(out_channels3reduce, out_channels3, 3, 1, 1);

	branch3_conv3a = ConvBNReLUFactorization(out_channels3, out_channels3, {3, 1}, {1, 0});

	branch3_conv3b = ConvBNReLUFactorization(out_channels3, out_channels3, {1, 3}, {0, 1});

	branch4->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
	branch4->push_back(torch::nn::Conv2d(Options(in_channels, out_channels4, 1).stride(1).padding(0)));
	branch4->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels4).eps(0.001)));
	branch4->push_back(torch::nn::ReLU6(true));

	register_module("C_branch1", branch1);
	register_module("C_branch2_conv1", branch2_conv1);
	register_module("C_branch2_conv2a", branch2_conv2a);
	register_module("C_branch2_conv2b", branch2_conv2b);
	register_module("C_branch3_conv1",  branch3_conv1);
	register_module("C_branch3_conv2",  branch3_conv2);
	register_module("C_branch3_conv3a", branch3_conv3a);
	register_module("C_branch3_conv3b", branch3_conv3b);
	register_module("C_branch4", branch4);

}


torch::Tensor InceptionV3ModuleCImpl::forward(const torch::Tensor& x) {
	torch::Tensor out1 = branch1->forward(x);
	torch::Tensor x2 = branch2_conv1->forward(x);
	torch::Tensor out2 = torch::cat({branch2_conv2a->forward(x2), branch2_conv2b->forward(x2)}, 1);
	torch::Tensor x3 = branch3_conv2->forward(branch3_conv1->forward(x));
	torch::Tensor out3 = torch::cat({branch3_conv3a->forward(x3), branch3_conv3b->forward(x3)}, 1);
	torch::Tensor out4 = branch4->forward(x);

    return torch::cat({out1, out2, out3, out4}, 1);
}

InceptionV3ModuleDImpl::InceptionV3ModuleDImpl( int64_t in_channels, int64_t out_channels1reduce, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2) {
	branch1 = ConvBNReLU(in_channels, out_channels1reduce, 1, 1, 0);
	// ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3,stride=2)
	branch1->push_back(torch::nn::Conv2d(Options(out_channels1reduce, out_channels1, 3).stride(2).padding(0)));
	branch1->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels1).eps(0.001)));
	branch1->push_back(torch::nn::ReLU6(true));

	branch2 = ConvBNReLU(in_channels, out_channels2reduce, 1, 1, 0);
	//ConvBNReLU(in_channels=, out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2, 3).stride(1).padding(1)));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));
	// ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2),
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2, out_channels2, 3).stride(2).padding(0)));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	register_module("D_branch1", branch1);
	register_module("D_branch2", branch2);
}

torch::Tensor InceptionV3ModuleDImpl::forward(const torch::Tensor& x) {
	torch::Tensor out1 = branch1->forward(x);
	torch::Tensor out2 = branch2->forward(x);
	torch::Tensor out3 = torch::max_pool2d(x, 3, 2);

    return torch::cat({out1, out2, out3}, 1);
}

InceptionV3ModuleEImpl::InceptionV3ModuleEImpl( int64_t in_channels, int64_t out_channels1reduce, int64_t out_channels1, int64_t out_channels2reduce, int64_t out_channels2) {

	branch1 = ConvBNReLU(in_channels, out_channels1reduce, 1, 1, 0);
	// ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2)
	branch1->push_back(torch::nn::Conv2d(Options(out_channels1reduce, out_channels1, 3).stride(2).padding(0)));
	branch1->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels1).eps(0.001)));
	branch1->push_back(torch::nn::ReLU6(true));

	//ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
	branch2 = ConvBNReLU(in_channels, out_channels2reduce, 1, 1, 0);

	//ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce,kernel_sizes=[1, 7], paddings=[0, 3]);
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2reduce, {1,7}).stride(1).padding({0, 3})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2reduce).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2reduce, {1,7}).stride(1).padding({0, 3})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2reduce).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	//ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce,kernel_sizes=[7, 1], paddings=[3, 0]);
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2reduce, {7,1}).stride(1).padding({3, 0})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2reduce).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2reduce, {7,1}).stride(1).padding({3, 0})));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2reduce).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	//ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=2)
	branch2->push_back(torch::nn::Conv2d(Options(out_channels2reduce, out_channels2, 3).stride(2).padding(0)));
	branch2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(out_channels2).eps(0.001)));
	branch2->push_back(torch::nn::ReLU6(true));

	// nn.MaxPool2d(kernel_size=3,stride=2)
	//this->branch3->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(0)));

	register_module("E_branch1", branch1);
	register_module("E_branch2", branch2);
}

torch::Tensor InceptionV3ModuleEImpl::forward(const torch::Tensor& x) {
	torch::Tensor out1 = branch1->forward(x);
	torch::Tensor out2 = branch2->forward(x);
	torch::Tensor out3 = torch::max_pool2d(x, 3, 2);

    return torch::cat({out1, out2, out3}, 1);
}


InceptionAux_Impl::InceptionAux_Impl(int64_t in_channels, int64_t num_classes) :
		aux_conv1(ConvBNReLU(in_channels, 128, 1, 1, 0)),
		aux_conv2(torch::nn::Sequential(torch::nn::Conv2d(Options(128, 768, 5).stride(1)))),
		fc(768, num_classes) {

	register_module("aux_conv1", aux_conv1);
	register_module("aux_conv2", aux_conv2);
	register_module("fc", fc);
}

torch::Tensor InceptionAux_Impl::forward(torch::Tensor x) {
	x = torch::avg_pool2d(x, 5, 3);
	x = aux_conv1->forward(x);
	x = aux_conv2->forward(x);
	x = x.view({x.size(0), -1});
	x = torch::dropout(x, 0.7, true);
	x = fc->forward(x);
	return x;
}

InceptionV3_Impl::InceptionV3_Impl(int64_t num_classes_, std::string stage_) {
	stage = stage_;
	num_classes = num_classes_;
	block1 = ConvBNReLU(3, 32, 3, 2, 0);

	//ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3, stride=1)
	block1->push_back(torch::nn::Conv2d(Options(32, 32, 3).stride(1).padding(0)));
	block1->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(32).eps(0.001)));
	block1->push_back(torch::nn::ReLU6(true));

	//ConvBNReLU(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
	block1->push_back(torch::nn::Conv2d(Options(32, 64, 3).stride(1).padding(1)));
	block1->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(64).eps(0.001)));
	block1->push_back(torch::nn::ReLU6(true));

	// nn.MaxPool2d(kernel_size=3, stride=2)
	block1->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(0)));

	block2 = ConvBNReLU(64, 80, 3, 1, 0);
    //ConvBNReLU(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1)
	block2->push_back(torch::nn::Conv2d(Options(80, 192, 3).stride(1).padding(1)));
	block2->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNormOptions(192).eps(0.001)));
	block2->push_back(torch::nn::ReLU6(true));
	//nn.MaxPool2d(kernel_size=3, stride=2)
	block2->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(0)));

	linear =  torch::nn::Linear(2048, num_classes);

	register_module("block1", block1);
	register_module("block2", block2);

	register_module("block3_1", block3_1);
	register_module("block3_2", block3_2);
	register_module("block3_3", block3_3);

	register_module("block4_1", block4_1);
	register_module("block4_2", block4_2);
	register_module("block4_3", block4_3);
	register_module("block4_4", block4_4);
	register_module("block4_5", block4_5);

	register_module("block5_1", block5_1);
	register_module("block5_2", block5_2);
	register_module("block5_3", block5_3);

	register_module("linear", linear);

    // Initializing weights
    for (auto& module : modules(/*include_self=*/false)) {
    	if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
    		torch::nn::init::kaiming_normal_(
            M->weight,
            /*a=*/0,
            torch::kFanOut,
            torch::kReLU);
    	else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
    		torch::nn::init::constant_(M->weight, 1);
    		torch::nn::init::constant_(M->bias, 0);
        } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
    		torch::nn::init::normal_(M->weight, 0.0, 0.01);
    		torch::nn::init::constant_(M->bias, 0);
        }
    }
}

InceptionV3Output InceptionV3_Impl::forward(torch::Tensor x) {
	x = block1->forward(x);
//	std::cout <<"==>1\n" << x.sizes() << std::endl;
	x = block2->forward(x);
//	std::cout <<"==>2\n" << x.sizes() << std::endl;
	x = block3_1->forward(x);
//	std::cout <<"==>3\n" << x.sizes() << std::endl;
	x = block3_2->forward(x);
//	std::cout <<"==>4\n" << x.sizes() << std::endl;
	x = block3_3->forward(x);
//	std::cout <<"==>5\n" << x.sizes() << std::endl;
	x = block4_1->forward(x);
//	std::cout <<"==>6\n" << x.sizes() << std::endl;
	x = block4_2->forward(x);
//	std::cout <<"==>7\n" << x.sizes() << std::endl;
	x = block4_3->forward(x);
//	std::cout <<"==>8\n" << x.sizes() << std::endl;
	x = block4_4->forward(x);
//	std::cout <<"==>9\n" << x.sizes() << std::endl;
	x = block4_5->forward(x);
	at::Tensor aux(x.clone());

//	std::cout <<"==>10\n" << x.sizes() << std::endl;
	x = this->block5_1->forward(x);
	x = this->block5_2->forward(x);
	x = this->block5_3->forward(x);
	x = torch::max_pool2d(x, 8, 1);
	x = torch::dropout(x, 0.5, true);
	x = x.view({x.size(0),-1});
	x = linear->forward(x);

   if( stage == "train" ) {
	   InceptionAux_Impl aux_logits = InceptionAux_Impl(768, num_classes);
	   aux = aux_logits.forward(aux);
       return {aux, x};
   } else {
	   return {{}, x};
   }
}





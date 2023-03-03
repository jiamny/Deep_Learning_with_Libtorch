#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdint.h>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>

#include <dirent.h>           							 //get files in directory
#include <sys/stat.h>
#include <cmath>
#include <map>
#include <tuple>
#include <matplot/matplot.h>
using namespace matplot;

#include "../../image_tools/transforms.hpp"              // transforms_Compose
#include "../../image_tools/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../../image_tools/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths


struct INetImpl : public torch::nn::Module {
	torch::nn::Sequential features{nullptr}, classifier{nullptr};

	explicit INetImpl(int num_classes) {

		features = torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3)),
				torch::nn::Functional(torch::relu),
				torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2})),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5)),
				torch::nn::Functional(torch::relu),
				torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(0.1)),
				torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1}))
				);

		classifier = torch::nn::Sequential(
		      torch::nn::Linear(64, 32),
		      torch::nn::Functional(torch::relu),
			  torch::nn::Linear(32, num_classes));

		register_module("features", features);
		register_module("classifier", classifier);
	}

    torch::Tensor forward( torch::Tensor x) {
    	  x = features->forward(x);
    	  x = x.view({x.size(0), -1});
    	  x = classifier->forward(x);
        return x;
    }
};
TORCH_MODULE(INet);


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	INet net(2);
	net->to(device);

	auto dict = net->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout << (*n).key() << std::endl;
	}
	std::cout << net << "\n\n";

    // Let’s try a random 32x32 input:
    auto input = torch::randn({1, 3, 32, 32}).to(device);
    auto out = net->forward(input);
    std::cout << out << "\n\n";

    std::vector<std::string> class_names = {"airplane", "automobile"};

    constexpr size_t batch_size = 50;
    constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
    constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset

    bool test  = true;						// has test dataset
    int img_size = 32;

    // Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

    std::string dataroot = "/media/stree/localssd/DL_data/cifar/cifar2/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> test_batch;
    torch::Tensor loss, features, labels, output;
    datasets::ImageFolderClassesWithPaths dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, test_dataloader; 	// dataloader;

    // Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, train_shuffle, train_workers);

    std::cout << "total training images : " << dataset.size() << std::endl;

    if( test ) {
    	std::string test_dataroot = "/media/stree/localssd/DL_data/cifar/cifar2/test";
    	test_dataset = datasets::ImageFolderClassesWithPaths(test_dataroot, transform, class_names);

    	test_dataloader = DataLoader::ImageFolderClassesWithPaths(test_dataset, 1, false, 0);

    	std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;
    }

    auto optimizer = torch::optim::SGD(net->parameters(),  0.01);
    //auto loss_func = torch::nn::BCELoss();
    auto loss_func = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	size_t epochs = 50;
	size_t log_step_freq = 100;
	float loss_sum;
	int step;

	std::vector<float> v_epoch, v_train_loss, v_val_loss;

	for( size_t epoch = 0;  epoch < epochs; epoch++ ) {
		// train -------------------------------------------------
		net->train();
		torch::AutoGradMode enable_grad(true);

		loss_sum = 0.0;
		step = 0;

		while( dataloader(mini_batch) ) {
			features = std::get<0>(mini_batch).to(device);
			labels = std::get<1>(mini_batch).to(device);

			optimizer.zero_grad();

			auto output = net->forward(features);
			auto predictions = torch::nn::functional::log_softmax(output, 1);

			loss = loss_func(predictions,labels);

			loss.backward();
			optimizer.step();

			loss_sum += loss.item<float>();

			if( (step + 1) % log_step_freq == 0)
				std::cout <<"step = " << (step + 1) << " loss: " << loss_sum/(step + 1) << std::endl;
			step++;
		}

		// validation ---------------------------------------------
		net->eval();
		torch::NoGradGuard no_grad;

		float val_loss_sum = 0.0;
		int val_step = 0;

		while( test_dataloader(test_batch) ) {
			features = std::get<0>(test_batch).to(device);
			labels = std::get<1>(test_batch).to(device);

			torch::NoGradGuard no_grad;
			auto output = net->forward(features);
			auto predictions = torch::nn::functional::log_softmax(output, 1);
			auto val_loss = loss_func(predictions,labels);
			val_loss_sum += val_loss.item<float>();
			val_step++;
		}
		std::cout << "\nEPOCH = " << epoch << ", loss = " << loss_sum/step << ", val_loss = " << val_loss_sum/val_step << "\n";
		v_epoch.push_back(epoch*1.0);
		v_train_loss.push_back(loss_sum/step);
		v_val_loss.push_back(val_loss_sum/val_step);
	}
	std::cout << "Finished Training...\n";

    tiledlayout(1, 1);
    auto ax1 = nexttile();
    plot(ax1, v_epoch, v_train_loss, "b")->line_width(2);
    hold(ax1, true);
    plot(ax1, v_epoch, v_val_loss, "r-.")->line_width(2);
    hold(ax1, false);
    xlabel(ax1, "epoch");
    ylabel(ax1, "loss");
    legend(ax1, "train loss", "val loss");

    show();
	std::cout << "Done!\n";
	return 0;
}






#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>

#include <dirent.h>           //get files in directory
#include <sys/stat.h>
#include <cmath>
#include <map>
#include <tuple>

#include "../../../image_tools/transforms.hpp"              // transforms_Compose
#include "../../../image_tools/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../../../image_tools/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths


using Options = torch::nn::Conv2dOptions;

struct Residual : public torch::nn::Module {

	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
	bool use_1x1 = false;
    //"""The Residual block of ResNet."""
    explicit Residual(int64_t input_channels, int64_t num_channels, bool use_1x1conv, int64_t strides) { // false, 1
    	use_1x1 = use_1x1conv;
        conv1 = torch::nn::Conv2d(Options(input_channels, num_channels, 3).stride(strides).padding(1));
        conv2 = torch::nn::Conv2d(Options(num_channels, num_channels, 3).padding(1));

        if( use_1x1conv )
            conv3 = torch::nn::Conv2d(Options(input_channels, num_channels, 1).stride(strides));

        bn1 = torch::nn::BatchNorm2d(num_channels);
        bn2 = torch::nn::BatchNorm2d(num_channels);
    }

    torch::Tensor forward(torch::Tensor X) {
        auto Y = torch::relu(bn1->forward(conv1->forward(X)));
        Y = bn2->forward(conv2->forward(Y));
        if(use_1x1)
            X = conv3->forward(X);
        Y += X;
        return torch::relu(Y);
    }
};

/*
 * GoogLeNet uses four modules made up of Inception blocks. However, ResNet uses four modules made up of residual blocks,
 * each of which uses several residual blocks with the same number of output channels. The number of channels in the first
 * module is the same as the number of input channels. Since a maximum pooling layer with a stride of 2 has already been used,
 * it is not necessary to reduce the height and width. In the first residual block for each of the subsequent modules,
 * the number of channels is doubled compared with that of the previous module, and the height and width are halved.
 */

torch::nn::Sequential resnet_block(int64_t input_channels, int64_t num_channels, int num_residuals, bool first_block) {
	torch::nn::Sequential blk;
    for(int i= 0; i < num_residuals; i++ ){
        if( i == 0 && ! first_block )
            blk->push_back(
                Residual(input_channels, num_channels, true, 2));
        else
            blk->push_back(Residual(num_channels, num_channels, false, 1));
    }
    return blk;
}

// ResNet Model
/*
 * The first two layers of ResNet are the same as those of the GoogLeNet we described before: the 7×7 convolutional layer with
 * 64 output channels and a stride of 2 is followed by the 3×3 maximum pooling layer with a stride of 2. The difference is the
 * batch normalization layer added after each convolutional layer in ResNet.
*/

struct  ResNetImpl : public torch::nn::Module {
	torch::nn::Sequential b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr}, b5{nullptr};
	torch::nn::Linear classifier{nullptr};

	ResNetImpl(int64_t num_classes) {
		// The first module uses a 64-channel 7×7 convolutional layer.
		b1 = torch::nn::Sequential(torch::nn::Conv2d(Options(3, 64, 7).stride(2).padding(3)),
											torch::nn::BatchNorm2d(64),
											torch::nn::ReLU(),
											torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
		// Then, we add all the modules to ResNet. Here, two residual blocks are used for each module.
		b2 = resnet_block(64, 64, 2, true);
		b3 = resnet_block(64, 128, 2, false);
		b4 = resnet_block(128, 256, 2, false);
		b5 = resnet_block(256, 512, 2, false);
		classifier = torch::nn::Linear(512*6*6, num_classes);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = b1->forward(x);
		x = b2->forward(x);
		x = b3->forward(x);
		x = b4->forward(x);
		x = b5->forward(x);
		//torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
		//torch::nn::Flatten()
		x = torch::adaptive_avg_pool2d(x, {6, 6});
		x = x.view({x.size(0), -1});

		return classifier->forward(x);
	}
};

TORCH_MODULE(ResNet);

std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num) {
    // (1) Memory Allocation
    std::vector<std::string> class_names = std::vector<std::string>(class_num);

    // (2) Get Class Names
    std::string class_name;
    std::ifstream ifs(path, std::ios::in);
    size_t i = 0;
    if( ! ifs.fail() ) {
    	while( getline(ifs, class_name) ) {
//    		std::cout << class_name.length() << std::endl;
    		if( class_name.length() > 2 ) {
    			class_names.at(i) = class_name;
    			i++;
    		}
    	}
    } else {
    	std::cerr << "Error : can't open the class name file." << std::endl;
    	std::exit(1);
    }

    ifs.close();
    if( i != class_num ){
        std::cerr << "Error : The number of classes does not match the number of lines in the class name file." << std::endl;
        std::exit(1);
    }

    // End Processing
    return class_names;
}

int main() {

	// Device
	bool cpu_only = true;

	torch::Device device( torch::kCPU );

	if( ! cpu_only ) {
		auto cuda_available = torch::cuda::is_available();
		device = cuda_available ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
		std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
	} else {
		std::cout << "Training on CPU." << '\n';
	}

	int64_t img_size = 224;
	size_t batch_size = 16;
	const std::string path = "./data/DogsVSCats_name.txt";
	const size_t class_num = 2;
	const size_t valid_batch_size = 1;
	std::vector<std::string> class_names = Set_Class_Names( path, class_num);
	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset

    bool valid = false;						// has valid dataset
    bool test  = true;						// has test dataset

    // (4) Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.5, 0.5, 0.5}, std::vector<float>{0.5, 0.5, 0.5})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "/media/stree/localssd/DL_data/DogsVSCats/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // (1) Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

	if( valid ) {
		std::string valid_dataroot = "/media/stree/localssd/DL_data/DogsVSCats/valid";
		valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
		valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);

		std::cout << "total validation images : " << valid_dataset.size() << std::endl;
	}
    bool vobose = false;

    ResNet model( class_num );
//	model->init();
	model->to(device);
	std::cout << model << std::endl;

	auto dict = model->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout << (*n).key() << std::endl;
	}

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1, 3, img_size, img_size}).to(device);
	torch::Tensor y = model->forward(x);
	std::cout << y << std::endl;

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	start_epoch = 1;
	total_iter = dataloader.get_count_max();
	total_epoch = 35;

	bool first = true;
	std::vector<float> train_loss_ave;
	std::vector<float> train_epochs;

	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		model->train();
		torch::AutoGradMode enable_grad(true);

		std::cout << "--------------- Training --------------------\n";
		first = true;
		float loss_sum = 0.0;
		while (dataloader(mini_batch)) {
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);

			if( first && vobose ) {
				for(size_t i = 0; i < label.size(0); i++)
					std::cout << label[i].item<int64_t>() << " ";
				std::cout << "\n";
				first = false;
			}

			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			output = model->forward(image);
			auto out = torch::nn::functional::log_softmax(output, 1); // dim
			//std::cout << output.sizes() << "\n" << out.sizes() << std::endl;

			loss = criterion(out, label); //torch::mse_loss(out, label);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

			loss_sum += loss.item<float>();
		}

		train_loss_ave.push_back(loss_sum/total_iter);
		train_epochs.push_back(epoch*1.0);
		std::cout << "epoch: " << epoch << "/"  << total_epoch << ", avg_loss: " << (loss_sum/total_iter) << std::endl;

		// ---------------------------------
		// validation
		// ---------------------------------
		if( valid && (epoch % 5 == 0) ) {
			std::cout << "--------------- validation --------------------\n";
			model->eval();
			torch::NoGradGuard no_grad;

			size_t iteration = 0;
			float total_loss = 0.0;
			size_t total_match = 0, total_counter = 0;
			torch::Tensor responses;
			first = true;
			while (valid_dataloader(mini_batch)){

				image = std::get<0>(mini_batch).to(device);
			    label = std::get<1>(mini_batch).to(device);
			    size_t mini_batch_size = image.size(0);

			    if( first && vobose ) {
			    	for(size_t i = 0; i < label.size(0); i++)
			    		std::cout << label[i].item<int64_t>() << " ";
			    	std::cout << "\n";
			    	first = false;
			    }

			    output = model->forward(image);
			    auto out = torch::nn::functional::log_softmax(output, 1); // dim=
			    loss = criterion(out, label);

			    responses = output.exp().argmax(1); // dim

			    for (size_t i = 0; i < mini_batch_size; i++){
			        int64_t response = responses[i].item<int64_t>();
			        int64_t answer = label[i].item<int64_t>();

			        total_counter++;
			        if (response == answer) total_match++;
			    }
			    total_loss += loss.item<float>();
			    iteration++;
			}
			// (3) Calculate Average Loss
			float ave_loss = total_loss / (float)iteration;

			// (4) Calculate Accuracy
			float total_accuracy = (float)total_match / (float)total_counter;
			std::cout << "Validation accuracy: " << total_accuracy << std::endl << std::endl;
		}
	}

	//
	if( test ) {
		std::cout << "--------------- Testing --------------------\n";
		std::string test_dataroot = "/media/stree/localssd/DL_data/DogsVSCats/test";
		test_dataset = datasets::ImageFolderClassesWithPaths(test_dataroot, transform, class_names);

		test_dataloader = DataLoader::ImageFolderClassesWithPaths(test_dataset, 1, false, 0);

		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		float  ave_loss = 0.0;
		size_t match = 0;
		size_t counter = 0;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> data;
		std::vector<size_t> class_match = std::vector<size_t>(class_num, 0);
		std::vector<size_t> class_counter = std::vector<size_t>(class_num, 0);
		std::vector<float> class_accuracy = std::vector<float>(class_num, 0.0);

	    model->eval();
	    torch::NoGradGuard no_grad;

	    while( test_dataloader(data) ){
	        image = std::get<0>(data).to(device);
	        label = std::get<1>(data).to(device);
	        output = model->forward(image);
	        auto out = torch::nn::functional::log_softmax(output, 1);

	        loss = criterion(out, label);

	        ave_loss += loss.item<float>();

	        output = output.exp();

	        int64_t response = output.argmax(1).item<int64_t>();

	        int64_t answer = label[0].item<int64_t>();
	        counter += 1;
	        class_counter[answer]++;

	        if (response == answer){
	        	class_match[answer]++;
	            match += 1;
	        }
	    }

	    // (7.1) Calculate Average
	    ave_loss = ave_loss / (float)dataset.size();

	    // (7.2) Calculate Accuracy
	    std::cout << "Test accuracy ==========\n";
	    for (size_t i = 0; i < class_num; i++){
	    	class_accuracy[i] = (float)class_match[i] / (float)class_counter[i];
	    	std::cout << class_names[i] << ": " << class_accuracy[i] << "\n";
	    }
	    float accuracy = (float)match / float(counter);
	    std::cout << "\nTest accuracy: " << accuracy << std::endl;
	}

    std::cout << "Done!\n";
}

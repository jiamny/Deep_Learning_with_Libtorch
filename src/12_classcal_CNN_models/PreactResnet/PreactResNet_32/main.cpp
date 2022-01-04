
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

#include "PreactResNet.h"

#include "../../../image_tools/transforms.hpp"              // transforms_Compose
#include "../../../image_tools/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../../../image_tools/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths


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
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	int64_t img_size = 32;
	size_t batch_size = 200;
	const std::string path = "./data/CIFAR10_names.txt";
	const size_t class_num = 10;
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
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/CIFAR10/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // (1) Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

	if( valid ) {
		std::string valid_dataroot = "./data/CIFAR10/valid";
		valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
		valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);

		std::cout << "total validation images : " << valid_dataset.size() << std::endl;
	}
    bool vobose = false;

    PreActResNetBB model = PreActResNet18(class_num);
//	model->init();
	model->to(device);
	std::cout << model << std::endl;

	auto dict = model->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout << (*n).key() << std::endl;
	}

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1,3,img_size, img_size}).to(device);
	torch::Tensor y = model->forward(x);
	std::cout << y << std::endl;

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	start_epoch = 1;
	total_iter = dataloader.get_count_max();
	total_epoch = 1;

	bool first = true;
	std::vector<float> train_loss_ave;
	std::vector<float> train_epochs;

	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		model->train();
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
		std::string test_dataroot = "./data/CIFAR10/test";
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


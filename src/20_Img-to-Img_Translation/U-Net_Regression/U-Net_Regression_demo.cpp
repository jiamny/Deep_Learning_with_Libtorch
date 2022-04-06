
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

#include "networks.hpp"
#include "loss.hpp"

#include "../../image_tools/transforms.hpp"              // transforms_Compose
#include "../../image_tools/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../../image_tools/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include "../../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // Set Seed
	int seed = 0;
    std::srand(seed);
    torch::manual_seed(std::rand());
    if( torch::cuda::is_available() ) {
    	torch::globalContext().setDeterministicCuDNN(true);
    	torch::globalContext().setBenchmarkCuDNN(false);
    }

    int img_size = 256;
    size_t input_nc = 3;	// input image channel : RGB=3, grayscale=1
    size_t output_nc = 3;
    size_t class_num = 3;	// total classes

    // (4) Set Transforms
    std::vector<transforms_Compose> transformI{
            transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
            transforms_ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
            transforms_Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if(input_nc == 1){
    	transformI.insert(transformI.begin(), transforms_Grayscale(1));
    }
    std::vector<transforms_Compose> transformO{
            transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
            transforms_ToTensor(),                                                                            // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
            transforms_Normalize(0.5, 0.5)                                                                    // [0,1] ===> [-1,1]
    };
    if(output_nc == 1){
    	transformO.insert(transformO.begin(), transforms_Grayscale(1));
    }

    size_t nf = 64;				// the number of filters in convolution layer closest to image
    size_t nz = 512;			// dimensions of latent space
    bool   no_dropout = false;  // Dropout off/on

    // Define Network
    UNet unet(nf, img_size, no_dropout, nz, input_nc, output_nc);
    unet->to(device);


    // Calculation of Parameters
    size_t num_params = 0;
    for (auto param : unet->parameters()){
        num_params += param.numel();
    }
    std::cout << "Total number of parameters : " << (float)num_params/1e6f << "M" << std::endl << std::endl;
    std::cout << unet << std::endl;

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1, 3, img_size, img_size}).to(device);
	torch::Tensor y = unet->forward(x);
	std::cout <<  y.sizes() << std::endl;

	const size_t batch_size        = 32;
    constexpr bool   train_shuffle = true;  	// whether to shuffle the training dataset
    constexpr size_t train_workers = 2;  		// the number of workers to retrieve data from the training dataset

    // -----------------------------------
    // Initialization and Declaration
    // -----------------------------------
    bool test  = true;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    torch::Tensor loss, imageI, imageO, output;
    datasets::ImageFolderPairWithPaths dataset;
    DataLoader::ImageFolderPairWithPaths dataloader;

    std::string input_dir = "./data/facades/trainI";
    std::string output_dir = "./data/facades/trainO";

    // get train dataset
    dataset = datasets::ImageFolderPairWithPaths(input_dir, output_dir, transformI, transformO);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, batch_size, train_shuffle, train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    float lr     = 1e-4;
    float beta1  = 0.5;
    float beta2  = 0.999;
    std::string ls = "l2";  //"vanilla (cross-entropy), lsgan (mse), etc.")

    // (3) Set Optimizer Method
    auto optimizer = torch::optim::Adam(unet->parameters(), torch::optim::AdamOptions(lr).betas({beta1, beta2}));

    // (4) Set Loss Function
    auto criterion = Loss(ls);

    std::string train_load_epoch = "";
    size_t epoch;
    size_t total_iter = dataloader.get_count_max();
    size_t start_epoch, total_epoch;
    size_t mini_batch_size;
    total_epoch = 50;

    if(train_load_epoch == ""){
    	unet->apply(weights_init);
        start_epoch = 0;
    }

    std::vector<float> train_loss, train_epochs;

    for (epoch = start_epoch; epoch <= total_epoch; epoch++){

    	unet->train();
    	float tot_loss = 0.0;
    	mini_batch_size = 0;
        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){

            // -----------------------------------
            // c1. U-Net Training Phase
            // -----------------------------------
            imageI = std::get<0>(mini_batch).to(device);
            imageO = std::get<1>(mini_batch).to(device);
            output = unet->forward(imageI);
            loss = criterion(output, imageO);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            tot_loss += loss.item<float>();
            mini_batch_size++;
        }

    	train_loss.push_back(tot_loss/mini_batch_size);
    	train_epochs.push_back(epoch*1.0);

        if (epoch % 10 == 0) {
        	std::cout << "epoch: " << epoch << "/"  << total_epoch << ", train_loss: " << (tot_loss/mini_batch_size) << std::endl;
        }
    }

	// ---- Testing
	if( test ) {
		std::cout << "--------------- Testing --------------------\n";

		float ave_loss;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
		torch::Tensor imageI, imageO, output;
		torch::Tensor loss;
		datasets::ImageFolderPairWithPaths test_dataset;
		DataLoader::ImageFolderPairWithPaths test_dataloader;

		std::string input_dir = "./data/facades/testI";
		std::string output_dir = "./data/facades/testO";
		test_dataset = datasets::ImageFolderPairWithPaths(input_dir, output_dir, transformI, transformO);
		test_dataloader = DataLoader::ImageFolderPairWithPaths(dataset, 1, false, 0);
		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		// (3) Set Loss Function
		auto criterion = Loss(ls);

		// (4) Initialization of Value
		ave_loss = 0.0;

		// (5) Tensor Forward
		unet->eval();

		while(test_dataloader(data)) {

			imageI = std::get<0>(data).to(device);
		    imageO = std::get<1>(data).to(device);

		    if( torch::cuda::is_available() )
		        torch::cuda::synchronize();

		    output = unet->forward(imageI);

		    loss = criterion(output, imageO);

		    ave_loss += loss.item<float>();
		}

		ave_loss = ave_loss / (float)dataset.size();
		std::cout << "<All> ave_loss:" << ave_loss << std::endl;
	}

	plt::figure_size(800, 600);
	plt::named_plot("train_loss", train_epochs, train_loss, "b");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();
	plt::close();

    std::cout << "Done!\n";
}


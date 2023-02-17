
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
    int nc = 3;			// input image channel : RGB=3, grayscale=1

    // Set Transforms
    // (4.1) for Original Dataset
    std::vector<transforms_Compose> transformO{
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                              // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                                      // [0,1] ===> [-1,1]
    };
    if(nc == 1){
        transformO.insert(transformO.begin(), transforms_Grayscale(1));
    }
    // (4.2) for Noised Dataset
    std::vector<transforms_Compose> transformI{
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor()                                               // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
    };
    if(nc == 1){
            transformI.insert(transformI.begin(), transforms_Grayscale(1));
    }
    // (2.1) for Random Valued Impulse Noise (RVIN)
    bool RVIN = false;  		// Random Valued Impulse Noise (RVIN) addition on/off
    float RVIN_prob = 0.01;		// RVIN probability of occurrence in each pixel
    // (2.2) for Salt and Pepper Noise (SPN)
    bool SPN = false;			// "Salt and Pepper Noise (SPN) addition on/off
    float SPN_prob = 0.01;		// SPN probability of occurrence in each pixel
    float SPN_salt_rate = 0.5;	// salt noise probability of occurrence in total noise
    // (2.3) for Gaussian Noise (GN)
    bool GN = false;			// Gaussian Noise (GN) addition on/off
    float GN_prob = 1.0; 		// GN probability of occurrence in each pixel
    float GN_mean = 0.0;		// the mean of GN
    float GN_std = 0.01;		// the standard deviation of GN

    if(RVIN){
        transformI.push_back(transforms_AddRVINoise(RVIN_prob));
    }
    if(SPN){
        transformI.push_back(transforms_AddSPNoise(SPN_prob, SPN_salt_rate));
    }
    if(GN){
        transformI.push_back(transforms_AddGaussNoise(GN_prob, GN_mean, GN_std));
    }
    transformI.push_back(transforms_Normalize(0.5, 0.5));

    size_t nf = 64;			// the number of filters in convolution layer closest to image
    size_t nz = 512;		// dimensions of latent space

    // Define Network
    ConvolutionalAutoEncoder CAE(nf, nc, nz);
    CAE->to(device);

    // Calculation of Parameters
    size_t num_params = 0;
    for (auto param : CAE->parameters()){
        num_params += param.numel();
    }
    std::cout << "Total number of parameters : " << (float)num_params/1e6f << "M" << std::endl << std::endl;
    std::cout << CAE << std::endl;

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1, 3, img_size, img_size}).to(device);
	torch::Tensor y = CAE->forward(x);
	std::cout << y.sizes() << std::endl;

	const size_t valid_batch_size = 1;
	const size_t batch_size       = 32;
    constexpr bool train_shuffle = true;  		// whether to shuffle the training dataset
    constexpr size_t train_workers = 2;  		// the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;  		// whether to shuffle the validation dataset
    constexpr size_t valid_workers = 2;  		// the number of workers to retrieve data from the validation dataset
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    std::string lossfn = "l2";					// "l1 (mean absolute error), l2 (mean squared error),
    											// ssim (structural similarity), etc.")

    // -----------------------------------
    // Initialization and Declaration
    // -----------------------------------
    bool valid = true;
    bool test  = true;
    std::string dataroot, valid_dataroot, test_dataroot;

    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;
    datasets::ImageFolderPairWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderPairWithPaths dataloader, valid_dataloader;

    dataroot = "/media/stree/localssd/DL_data/CelebA/train";

    // get train dataset
    dataset = datasets::ImageFolderPairWithPaths(dataroot, dataroot, transformI, transformO);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // get valid dataset
    if(valid){
        valid_dataroot = "/media/stree/localssd/DL_data/CelebA/valid";
        valid_dataset = datasets::ImageFolderPairWithPaths(valid_dataroot, valid_dataroot, transformI, transformO);
        valid_dataloader = DataLoader::ImageFolderPairWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

	ConvolutionalAutoEncoder model(nf, nc, nz);
	model->to(device);

	float lr = 1e-4;
	float beta1 = 0.5;
	float beta2 = 0.999;
	//Set Optimizer Method
	auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr).betas({beta1, beta2}));

	// Set Loss Function
	auto criterion = Loss(lossfn);

	std::string train_load_epoch = "";
	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	size_t mini_batch_size;
	torch::Tensor loss, imageI, imageO, output;

	// Get Weights and File Processing
	if(train_load_epoch == ""){
		model->apply(weights_init);
		start_epoch = 0;
	}
	start_epoch++;
	total_iter = dataloader.get_count_max();
	total_epoch = 10;

	std::vector<float> train_loss_ave;
	std::vector<float> train_epochs;

	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		model->train();
		torch::AutoGradMode enable_grad(true);

		std::cout << "--------------- Training --------------------\n";

		float loss_sum = 0.0;
		while (dataloader(mini_batch)) {
			imageI = std::get<0>(mini_batch).to(device);
			imageO = std::get<1>(mini_batch).to(device);
			mini_batch_size = imageI.size(0);

            // -----------------------------------
            // c1. Convolutional Auto Encoder Training Phase
            // -----------------------------------
            output = model->forward(imageI);
            loss = criterion(output, imageO);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            loss_sum += loss.cpu().item<float>();
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

			while (valid_dataloader(mini_batch)){
				imageI = std::get<0>(mini_batch).to(device);
				imageO = std::get<1>(mini_batch).to(device);
				output = model->forward(imageI);
				loss = criterion(output, imageO);
				total_loss += loss.cpu().item<float>();
				iteration++;
			}
			//  Calculate Average Loss
			float ave_loss = total_loss / (float)iteration;

			std::cout << "\nAverage loss: " << ave_loss << std::endl;
		}
	}

	// ---- Testing
	if( test ) {
		std::cout << "--------------- Testing --------------------\n";
		std::string input_dir = "/media/stree/localssd/DL_data/CelebA/test";
		std::string output_dir = "/media/stree/localssd/DL_data/CelebA/testO";
		datasets::ImageFolderPairWithPaths test_dataset = datasets::ImageFolderPairWithPaths(input_dir, output_dir, transformI, transformO);
		DataLoader::ImageFolderPairWithPaths test_dataloader = DataLoader::ImageFolderPairWithPaths(test_dataset, 1, false, 0);
		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		criterion = Loss(lossfn);

		float ave_loss, ave_GT_loss;
		torch::Tensor GT_loss;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;

	    model->eval();
	    torch::NoGradGuard no_grad;

	    // Initialization of Value
	    ave_loss = 0.0;
	    ave_GT_loss = 0.0;

	    while( test_dataloader(data) ){
	        imageI = std::get<0>(data).to(device);
	        imageO = std::get<1>(data).to(device);

	        if( torch::cuda::is_available() )
	        	torch::cuda::synchronize();

	        output = model->forward(imageI);

	        loss = criterion(output, imageI);
	        GT_loss = criterion(output, imageO);

	        ave_loss += loss.cpu().item<float>();
	        ave_GT_loss += GT_loss.cpu().item<float>();
	    }

	    // Average
	    ave_loss = ave_loss / (float)test_dataset.size();
	    ave_GT_loss = ave_GT_loss / (float)test_dataset.size();

	    // Average Output
	    std::cout << "<All> " << lossfn << ':' << ave_loss << " GT_" << lossfn << ':' << ave_GT_loss << std::endl;
	}

	plt::figure_size(600, 500);
	plt::named_plot("Train loss", train_epochs, train_loss_ave, "b");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();

    std::cout << "Done!\n";
}


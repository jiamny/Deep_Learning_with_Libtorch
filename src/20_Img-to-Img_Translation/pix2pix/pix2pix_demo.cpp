
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
        transforms_ToTensor(),                                      		// Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                              		// [0,1] ===> [-1,1]
    };
    if(input_nc == 1){
        transformI.insert(transformI.begin(), transforms_Grayscale(1));
    }
    std::vector<transforms_Compose> transformO{
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                              // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                                      // [0,1] ===> [-1,1]
    };
    if(output_nc == 1){
        transformO.insert(transformO.begin(), transforms_Grayscale(1));
    }

    size_t ngf = 64;			// the number of filters in convolution layer closest to image
    size_t ndf = 64;
    size_t nz = 512;			// dimensions of latent space
    size_t n_layers = 3;		// the number of layers in PatchGAN
    bool   no_dropout = false;  // Dropout off/on

    // Define Network
    UNet_Generator gen(ngf, img_size, no_dropout, nz, input_nc, output_nc);
    gen->to(device);
    PatchGAN_Discriminator dis(ndf, input_nc, output_nc, n_layers);
    dis->to(device);

    // Calculation of Parameters
    size_t num_params = 0;
    for (auto param : gen->parameters()){
        num_params += param.numel();
    }
    std::cout << "Total number of parameters : " << (float)num_params/1e6f << "M" << std::endl << std::endl;
    std::cout << gen << std::endl;

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1, 3, img_size, img_size}).to(device);
	torch::Tensor y = gen->forward(x);
	std::cout <<  y.sizes() << std::endl;

	const size_t batch_size        = 1;
    constexpr bool   train_shuffle = true;  	// whether to shuffle the training dataset
    constexpr size_t train_workers = 2;  		// the number of workers to retrieve data from the training dataset

    // -----------------------------------
    // Initialization and Declaration
    // -----------------------------------
    bool test  = true;
    std::string dataroot, valid_dataroot, test_dataroot;
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> mini_batch;

    datasets::ImageFolderPairWithPaths dataset;
    DataLoader::ImageFolderPairWithPaths dataloader;

    std::string input_dir = "./data/facades/trainI";
    std::string output_dir = "./data/facades/trainO";

    // get train dataset
    dataset = datasets::ImageFolderPairWithPaths(input_dir, output_dir, transformI, transformO);
    dataloader = DataLoader::ImageFolderPairWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

	float lr_gen = 2e-4;
	float lr_dis = 2e-4;
	float beta1  = 0.5;
	float beta2  = 0.999;
	float Lambda = 100.0;
	std::string loss = "vanilla";  //"vanilla (cross-entropy), lsgan (mse), etc.")

    // (3) Set Optimizer Method
    auto gen_optimizer = torch::optim::Adam(gen->parameters(), torch::optim::AdamOptions(lr_gen).betas({beta1, beta2}));
    auto dis_optimizer = torch::optim::Adam(dis->parameters(), torch::optim::AdamOptions(lr_dis).betas({beta1, beta2}));

    // (4) Set Loss Function
    auto criterion_GAN = Loss(loss);
    auto criterion_L1 = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kMean));


	// Set Loss Function
	auto criterion = Loss();

	std::string train_load_epoch = "";
	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	size_t mini_batch_size;
//	torch::Tensor loss, imageI, imageO, output;

	// Get Weights and File Processing
	if(train_load_epoch == ""){
	    gen->apply(weights_init);
	    dis->apply(weights_init);
	    start_epoch = 0;
	}
	start_epoch++;
	total_iter = dataloader.get_count_max();
	total_epoch = 300;
    torch::Tensor realI, realO, fakeO, realP, fakeP, pair;
    torch::Tensor dis_real_out, dis_fake_out;
    torch::Tensor gen_loss, G_L1_loss, G_GAN_loss;
    torch::Tensor dis_loss, dis_real_loss, dis_fake_loss;
    torch::Tensor label_real, label_fake;

	std::vector<float> G_loss, G_L_1_loss, D_real_loss, D_fake_loss, train_epochs;

	// (2) Training per Epoch
	for (epoch = start_epoch; epoch <= total_epoch; epoch++){
		gen->train();
	    dis->train();

	    // -----------------------------------
	    // b1. Mini Batch Learning
	    // -----------------------------------
	    float G_ls=0.0, G_L_ls=0.0, D_real_ls=0.0, D_fake_ls=0.0;
	    size_t cnt_iter = 0;

	    while (dataloader(mini_batch)) {

	    	realI = std::get<0>(mini_batch).to(device);
	        realO = std::get<1>(mini_batch).to(device);
	        mini_batch_size = realI.size(0);

	        // -----------------------------------
	        // c1. Discriminator and Generator Training Phase
	        // -----------------------------------

	        // (1) Generator and Discriminator Forward
	        fakeO = gen->forward(realI);
	        fakeP = torch::cat({realI, fakeO.detach()}, /*dim=*/1);
	        dis_fake_out = dis->forward(fakeP);

	        // (2) Set Target Label
	        label_real = torch::full({dis_fake_out.size(0), dis_fake_out.size(1), dis_fake_out.size(2), dis_fake_out.size(3)}, /*value*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
	        label_fake = torch::full({dis_fake_out.size(0), dis_fake_out.size(1), dis_fake_out.size(2), dis_fake_out.size(3)}, /*value*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

	        // (3) Discriminator Training
	        realP = torch::cat({realI, realO}, /*dim=*/1);
	        dis_real_out = dis->forward(realP);
	        dis_fake_loss = criterion_GAN(dis_fake_out, label_fake);
	        dis_real_loss = criterion_GAN(dis_real_out, label_real);
	        dis_loss = dis_real_loss + dis_fake_loss;
	        dis_optimizer.zero_grad();
	        dis_loss.backward();
	        dis_optimizer.step();

	        // (4) Generator Training
	        fakeP = torch::cat({realI, fakeO}, /*dim=*/1);
	        dis_fake_out = dis->forward(fakeP);
	        G_GAN_loss = criterion_GAN(dis_fake_out, label_real);
	        G_L1_loss = criterion_L1(fakeO, realO) * Lambda;
	        gen_loss = G_GAN_loss + G_L1_loss;
	        gen_optimizer.zero_grad();
	        gen_loss.backward();
	        gen_optimizer.step();

	        // -----------------------------------
	        // c2. Record Loss (iteration)
	        // -----------------------------------
	        G_ls += G_GAN_loss.item<float>();
	        G_L_ls += G_L1_loss.item<float>();
	        D_real_ls += dis_real_loss.item<float>();
	        D_fake_ls += dis_fake_loss.item<float>();
	        cnt_iter++;
	    }

	    // -----------------------------------
	    // b2. Record Loss (epoch)
	    // -----------------------------------
	    G_loss.push_back(G_ls/cnt_iter);
	    G_L_1_loss.push_back(G_L_ls/cnt_iter);
	    D_real_loss.push_back(D_real_ls/cnt_iter);
	    D_fake_loss.push_back(D_fake_ls/cnt_iter);
	    train_epochs.push_back(epoch*1.0);
	    if (epoch % 10 == 0)
	    	std::cout << "epoch: " << epoch << "/"  << total_epoch << ", G_loss: " << (G_ls/cnt_iter) << std::endl;
	}

	// ---- Testing
	if( test ) {
		std::cout << "--------------- Testing --------------------\n";

		float ave_loss_l1, ave_loss_l2;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;
		torch::Tensor realI, realO, fakeO;
		torch::Tensor loss_l1, loss_l2;
		datasets::ImageFolderPairWithPaths test_dataset;
		DataLoader::ImageFolderPairWithPaths test_dataloader;

		std::string input_dir = "./data/facades/testI";
		std::string output_dir = "./data/facades/testO";
		test_dataset = datasets::ImageFolderPairWithPaths(input_dir, output_dir, transformI, transformO);
		test_dataloader = DataLoader::ImageFolderPairWithPaths(dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		// (3) Set Loss Function
		auto criterion_L1 = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kMean));
		auto criterion_L2 = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));

		// (4) Initialization of Value
		ave_loss_l1 = 0.0;
		ave_loss_l2 = 0.0;

		// (5) Tensor Forward
		gen->train();  // Dropout is required to make the generated images diverse

		while(test_dataloader(data)){

		    realI = std::get<0>(data).to(device);
		    realO = std::get<1>(data).to(device);

		    if( torch::cuda::is_available() )
		    	torch::cuda::synchronize();

		    fakeO = gen->forward(realI);
		    loss_l1 = criterion_L1(fakeO, realO);
		    loss_l2 = criterion_L2(fakeO, realO);

		    ave_loss_l1 += loss_l1.item<float>();
		    ave_loss_l2 += loss_l2.item<float>();

		    std::cout << '<' << std::get<2>(data).at(0) << "> L1:" << loss_l1.item<float>() << " L2:" << loss_l2.item<float>() << std::endl;
		}
		// (6) Calculate Average
		ave_loss_l1 = ave_loss_l1 / (float)dataset.size();
		ave_loss_l2 = ave_loss_l2 / (float)dataset.size();

		// (7) Average Output
		std::cout << "<All> L1:" << ave_loss_l1 << " L2:" << ave_loss_l2 << std::endl;
	}

	plt::figure_size(800, 600);
	plt::named_plot("G_GAN_loss", train_epochs, G_loss, "b");
	plt::named_plot("G_L1_loss", train_epochs, G_L_1_loss, "c:");
	plt::named_plot("D_real_loss", train_epochs, D_real_loss, "g--");
	plt::named_plot("D_fake_loss", train_epochs, D_fake_loss, "r-.");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();

    std::cout << "Done!\n";
}


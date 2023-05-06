
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
    std::vector<transforms_Compose> transform{
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                              // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(0.5, 0.5)                                      // [0,1] ===> [-1,1]
    };
    if(nc == 1){
        transform.insert(transform.begin(), transforms_Grayscale(1));
    }

    size_t nf = 64;			// the number of filters in convolution layer closest to image
    size_t nz = 512;		// dimensions of latent space
    size_t nd = 512;		// the number of node for linear block in discriminator
    size_t n_blocks = 3;	// the number of linear blocks in discriminator

    // (5) Define Network
    WAE_Encoder enc(nf, img_size, nc, nz);
    enc->to(device);
    WAE_Decoder dec(nf, img_size, nc, nz);
    dec->to(device);
    GAN_Discriminator dis(nz, nd, n_blocks);
    dis->to(device);

    // Calculation of Parameters
    size_t enc_num_params = 0;
    for (auto param : enc->parameters()) enc_num_params += param.numel();
    size_t dec_num_params = 0;
    for (auto param : dec->parameters()) dec_num_params += param.numel();
    size_t dis_num_params = 0;
    for (auto param : dis->parameters()) dis_num_params += param.numel();

    std::cout << "ENC number of parameters : " << (float)enc_num_params/1e6f << "M" << std::endl << std::endl;
    std::cout << enc << std::endl;
    std::cout << "DEC number of parameters : " << (float)dec_num_params/1e6f << "M" << std::endl << std::endl;
    std::cout << dec << std::endl;
    std::cout << "DIS number of parameters : " << (float)dis_num_params/1e6f << "M" << std::endl << std::endl;
    std::cout << dis << std::endl;

	std::cout << "Test model ..." << std::endl;
	torch::Tensor x = torch::randn({1, 3, img_size, img_size}).to(device);
	torch::Tensor y = enc->forward(x);
	std::cout << y.sizes() << std::endl;

	bool synth = false;
	bool sample = false;

	constexpr size_t valid_batch_size = 1;
	constexpr size_t batch_size       = 32;
	constexpr size_t save_sample_iter = 5;      // the frequency of iteration to save sample images
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
    size_t start_epoch, total_epoch;
    std::string train_load_epoch = "";
    std::string dataroot, valid_dataroot, test_dataroot;

    std::tuple<torch::Tensor, std::vector<std::string>> mini_batch;
    datasets::ImageFolderWithPaths dataset, valid_dataset;
    DataLoader::ImageFolderWithPaths dataloader, valid_dataloader;

    dataroot = "/media/stree/localssd/DL_data/CelebA/train";

    // get train dataset
    dataset = datasets::ImageFolderWithPaths(dataroot, transform);
    dataloader = DataLoader::ImageFolderWithPaths(dataset, batch_size, train_shuffle, train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    // get valid dataset
    if(valid){
        valid_dataroot = "/media/stree/localssd/DL_data/CelebA/valid";
        valid_dataset = datasets::ImageFolderWithPaths(valid_dataroot, transform);
        valid_dataloader = DataLoader::ImageFolderWithPaths(valid_dataset, valid_batch_size, valid_shuffle, valid_workers);
        std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    }

    std::cout << "total validation images : " << valid_dataset.size() << std::endl;

    // Set Optimizer Method
    auto enc_optimizer = torch::optim::Adam(enc->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));
    auto dec_optimizer = torch::optim::Adam(dec->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));
    auto dis_optimizer = torch::optim::Adam(dis->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

    // Set Loss Function
    auto criterion = Loss(lossfn);
    auto criterion_GAN = torch::nn::BCEWithLogitsLoss(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean));

	if( train_load_epoch == ""){
	    enc->apply(weights_init);
	    dec->apply(weights_init);
	    dis->apply(weights_init);
	    start_epoch = 0;
	}

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t mini_batch_size;
	total_epoch = 30;

	torch::Tensor image, output, z_real, z_fake;
	torch::Tensor dis_real_out, dis_fake_out;
	torch::Tensor ae_loss, rec_loss, enc_loss, dis_loss, dis_real_loss, dis_fake_loss;
	torch::Tensor label_real, label_fake;

	std::vector<float> train_loss_rec, train_loss_gan, train_loss_dis;
	std::vector<float> train_epochs;
	mini_batch_size = 0;

	float Lambda = 0.01;
	start_epoch++;

	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		enc->train();
		dec->train();
		dis->train();
		torch::AutoGradMode enable_grad(true);

		std::cout << "--------------- Training --------------------\n";

		float loss_sum_rec = 0.0, loss_sum_gan = 0.0, loss_sum_dis=0.0;
		while (dataloader(mini_batch)) {
			image = std::get<0>(mini_batch).to(device);
			mini_batch_size = image.size(0);

            // ---------------------------------------------
            //  Wasserstein Auto Encoder Training Phase
            // ---------------------------------------------

            //  Set Target Label
            label_real = torch::full({(long int)mini_batch_size}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
            label_fake = torch::full({(long int)mini_batch_size}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

            // (2) Discriminator Training
            z_real = torch::randn({(long int)mini_batch_size, (long int)nz}).to(device);
            z_fake = enc->forward(image);
            dis_real_out = dis->forward(z_real).view({-1});
            dis_fake_out = dis->forward(z_fake.detach()).view({-1});
            dis_real_loss = criterion_GAN(dis_real_out, label_real) * Lambda;
            dis_fake_loss = criterion_GAN(dis_fake_out, label_fake) * Lambda;
            dis_loss = dis_real_loss + dis_fake_loss;
            dis_optimizer.zero_grad();
            dis_loss.backward();
            dis_optimizer.step();

            // (3) Auto Encoder Training
            output = dec->forward(z_fake);
            rec_loss = criterion(output, image);
            dis_fake_out = dis->forward(z_fake).view({-1});
            enc_loss = criterion_GAN(dis_fake_out, label_real) * Lambda;
            ae_loss = rec_loss + enc_loss;
            enc_optimizer.zero_grad();
            dec_optimizer.zero_grad();
            ae_loss.backward();
            enc_optimizer.step();
            dec_optimizer.step();

            // -----------------------------------
            // Record Loss (iteration)
            // -----------------------------------
            loss_sum_rec += rec_loss.item<float>();
            loss_sum_gan += enc_loss.item<float>();
            loss_sum_dis += dis_real_loss.item<float>();
		}

		train_loss_rec.push_back(loss_sum_rec/total_iter);
		train_loss_rec.push_back(loss_sum_rec/total_iter);
		train_loss_rec.push_back(loss_sum_rec/total_iter);
		train_epochs.push_back(epoch*1.0);
		std::cout << "epoch: " << epoch << "/"  << total_epoch
				<< ", rec_loss: " << (loss_sum_rec/total_iter)
				<< ", gan_loss: " << (loss_sum_gan/total_iter)
				<< ", dis_loss: " << (loss_sum_dis/total_iter) << std::endl;

		// ---------------------------------
		// validation
		// ---------------------------------
		if( valid && (epoch % 5 == 0) ) {
			std::cout << "--------------- validation --------------------\n";
			enc->eval();
			dec->eval();
			torch::NoGradGuard no_grad;

			size_t iteration = 0;
			float total_rec_loss = 0.0, total_enc_loss = 0.0, total_dis_real_loss = 0.0, total_dis_fake_loss = 0.0;

			while (valid_dataloader(mini_batch)) {

		        image = std::get<0>(mini_batch).to(device);
		        mini_batch_size = image.size(0);

		        // (1.1) Set Target Label
		        label_real = torch::full({(long int)mini_batch_size}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);
		        label_fake = torch::full({(long int)mini_batch_size}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat)).to(device);

		        // (1.2) Discriminator Loss
		        z_real = torch::randn({(long int)mini_batch_size, (long int)nz}).to(device);
		        z_fake = enc->forward(image);
		        dis_real_out = dis->forward(z_real).view({-1});
		        dis_fake_out = dis->forward(z_fake).view({-1});
		        dis_real_loss = criterion_GAN(dis_real_out, label_real) * Lambda;
		        dis_fake_loss = criterion_GAN(dis_fake_out, label_fake) * Lambda;

		        // (1.3) Auto Encoder Loss
		        output = dec->forward(z_fake);
		        rec_loss = criterion(output, image);
		        dis_fake_out = dis->forward(z_fake).view({-1});
		        enc_loss = criterion_GAN(dis_fake_out, label_real) * Lambda;

		        // (1.4) Update Loss
		        total_rec_loss += rec_loss.item<float>();
		        total_enc_loss += enc_loss.item<float>();
		        total_dis_real_loss += dis_real_loss.item<float>();
		        total_dis_fake_loss += dis_fake_loss.item<float>();

			    iteration++;
			}

			//  Calculate Average Loss
			std::cout << "\nAverage rec_loss: " << (total_rec_loss / iteration) << ", "
					  << "Average enc_loss: " << (total_enc_loss / iteration) << ", "
					  << "Average dis_real_loss: " << (total_dis_real_loss / iteration) << std::endl;
		}
	}

	// ---- Testing
	if( test ) {
		std::cout << "--------------- Testing --------------------\n";
		std::string input_dir = "/media/stree/localssd/DL_data/CelebA/test";
		std::string output_dir = "/media/stree/localssd/DL_data/CelebA/testO";
		datasets::ImageFolderPairWithPaths test_dataset = datasets::ImageFolderPairWithPaths(input_dir,
				output_dir, transform, transform);
		DataLoader::ImageFolderPairWithPaths test_dataloader = DataLoader::ImageFolderPairWithPaths(test_dataset, 1, false, 0);
		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		float ave_loss, ave_GT_loss;
		torch::Tensor imageI, imageO;
		torch::Tensor loss, GT_loss;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>, std::vector<std::string>> data;

		// Initialization of Value
		ave_loss = 0.0;
		ave_GT_loss = 0.0;

		// Tensor Forward
		enc->eval();
		dec->eval();
		torch::NoGradGuard no_grad;

	    while( test_dataloader(data) ){
	        imageI = std::get<0>(data).to(device);
	        imageO = std::get<1>(data).to(device);

	        if( torch::cuda::is_available() )
	        	torch::cuda::synchronize();

	        auto z = enc->forward(imageI);
	        output = dec->forward(z);

	        loss = criterion(output, imageI);
	        GT_loss = criterion(output, imageO);

	        ave_loss += loss.item<float>();
	        ave_GT_loss += GT_loss.item<float>();
	    }

	    // Average
	    ave_loss = ave_loss / (float)test_dataset.size();
	    ave_GT_loss = ave_GT_loss / (float)test_dataset.size();

	    // Average Output
	    std::cout << "<All> " << lossfn << ':' << ave_loss << " GT_" << lossfn << ':' << ave_GT_loss << std::endl;
	}

    std::cout << "Done!\n";
}


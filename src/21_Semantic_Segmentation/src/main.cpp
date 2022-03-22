#include<iostream>
#include"Segmentor.h"

int main(int argc, char *argv[]) {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. use GPU." : "use CPU.") << "\n";

	torch::manual_seed(1000);

	// ------------------------------------------------------
	// Create your Segmentation model with Libtorch Segment
	// ------------------------------------------------------
	auto model = UNet(1, "resnet34", "./src/21_Semantic_Segmentation/weights/resnet34.pt");

	model->to(device);
	model->eval();
	auto input = torch::rand({ 1,3,512,512 }).to(device);
	std::cout << input.sizes() <<"\n";
	auto output = model->forward(input);
	int T = 100;
	int64 t0 = cv::getCPUTickCount();
	for (int i = 0; i < T; i++) {
		auto output = model->forward(input);
		output = output.to(device);
	}
	output = output.to(device);
	int64 t1 = cv::getCPUTickCount();
	std::cout << "execution time is " << (t1 - t0) / (double)cv::getTickFrequency() << " seconds" << std::endl;

	// ------------------------------------------------------
	// Training model for person segmentation
	// ------------------------------------------------------
	bool use_tricks = false;
	Segmentor<FPN> segmentor;

	segmentor.Initialize(-1, // gpu id, -1 for cpu
		                512, // resize width
		                512, // resize height
		                {"background","person"},	//class name dict, background included
		                "resnet34",					// backbone name
		                "./src/21_Semantic_Segmentation/weights/resnet34.pt");

	if( use_tricks ) {
		//tricks for data augmentations
		trainTricks tricks;
		tricks.horizontal_flip_prob = 0.5;
		tricks.vertical_flip_prob = 0.5;
		tricks.scale_rotate_prob = 0.3;

		//tricks for training process
		tricks.decay_epochs = { 40, 80 };
		tricks.freeze_epochs = 8;

		segmentor.SetTrainTricks(tricks);
	}

	segmentor.Train(0.0003,	// initial leaning rate
		            300,	// training epochs
		            4,		// batch size
		            "./src/21_Semantic_Segmentation/voc_person_seg",
		            ".jpg",	// image type
		            "./src/21_Semantic_Segmentation/segmentor.pt");


	// ------------------------------------------------------
	// Predicting test
	// ------------------------------------------------------
    cv::Mat image = cv::imread("./src/21_Semantic_Segmentation/voc_person_seg/val/2007_004000.jpg");

    Segmentor<FPN> segmentor2;
    segmentor2.Initialize(-1,512,512,{"background","person"},
    	                         "resnet34","./src/21_Semantic_Segmentation/weights/resnet34.pt");

    segmentor2.LoadWeight("./src/21_Semantic_Segmentation/segmentor.pt");
    segmentor2.Predict(image,"person", "./src/21_Semantic_Segmentation");

    std::cout << "Done!\n";
    return 0;
}

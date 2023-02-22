//
//  classify.cpp
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 15/08/19.
//

#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <torch/script.h>
#include <unistd.h>
#include <fstream>
#include <dirent.h>           //get files in directory
#include <sys/stat.h>

// Utility function to load image from given folder
// File type accepted: .jpg
std::vector<std::string> load_images(std::string path) {
    std::vector<std::string> list_images;
	struct stat s;
	DIR* root_dir;
	struct dirent *dirs;
	if(path.back() != '/') {
		path.push_back('/');
	}

    if((root_dir = opendir(path.c_str())) != NULL) {
    	while ((dirs = readdir(root_dir))) {
        	std::string fd(dirs->d_name);
        	std::string fdpath = path + fd;
        	//std::cout << fdpath << std::endl;

        	if (fd[0] == '.')
        	   continue;

        	//it's a directory
        	if( stat(fdpath.c_str(), &s) == 0 ) {
        		if( s.st_mode & S_IFDIR ){

        			DIR *dir;
        			class dirent *ents;

        			dir = opendir(fdpath.c_str());
        			while ((ents = readdir(dir)) != NULL) {
        			    const std::string filename = ents->d_name;

        	            if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
        	                std::string newf = fdpath + "/" + filename;
        	                //std::cout << newf << std::endl;
        	                // --- Exclude empty image
        	                cv::Mat temp = cv::imread(newf, 1);
        	                if( ! temp.empty() ) list_images.push_back(newf);
        	            }
        			}
        			closedir(dir);
        		}
        	}
    	}
    }
    closedir(root_dir);

    return list_images;
}

void print_probabilities(std::string loc, std::string model_path, std::string model_path_linear, torch::Device device) {
    // Load image with OpenCV.
	//std::cout << loc << std::endl;
    cv::Mat img = cv::imread(loc.c_str(), 1); // 0 - gray image; 1 - color image
    cv::resize(img, img, cv::Size(224, 224)); //, cv::INTER_CUBIC); // cv::INTER_CUBIC cause: error: (-215:Assertion failed) !ssize.empty() in function 'resize'

    // Convert the image and label to a tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, torch::kByte);
    //torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
    img_tensor = img_tensor.to(torch::kF32);
    img_tensor = img_tensor.to(device);
    
    // Load the model.
    torch::jit::script::Module model;
    model = torch::jit::load(model_path);
    model.to(device);
    
    torch::nn::Linear model_linear(512, 2);
    torch::load(model_linear, model_path_linear);
    model_linear->to(device);
    
    // Predict the probabilities for the classes.
    std::vector<torch::jit::IValue> input;
    input.push_back(img_tensor);
    torch::Tensor prob = model.forward(input).toTensor();
    prob = prob.view({prob.size(0), -1});
    prob = model_linear->forward(prob).cpu();
    auto rlt = torch::argmax(prob);

    std::cout << "Image: " << loc << "\n";                 //std::endl;
    std::cout << "Cat prob = " << *(prob.data_ptr<float>()) << "; "; //std::endl;
    std::cout << "Dog prob = " << *(prob.data_ptr<float>()+1);
    if( rlt.item<long>() == 0 )
    	std::cout << "; Pred = cat." << std::endl;
    else
    	std::cout << "; Pred = dog." << std::endl;
}

int main(int arc, char** argv) {

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    // get current directory
	std::cout << "Current path is " << get_current_dir_name() << '\n';

    // argv[1] should is the test image
    std::string img_location = "/media/stree/localssd/DL_data/cat_dog/train"; //argv[1];

    std::vector<std::string> img_list = load_images(img_location);
    
    // argv[2] contains pre-trained model without last layer
    // argv[3] contains trained last FC layer
    std::string model_path = "./models/Transfer_learning/resnet18_without_last_layer.pt"; //argv[2];
    //std::string model_path = "./models/Tf_model.pt";
    std::string model_path_linear = "./models/Tf_model_linear.pt";      //argv[3];

    // Load the model.
    // You can also use: auto model = torch::jit::load(model_path);
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.to(device);
    
    torch::nn::Linear model_linear(512, 2);
    torch::load(model_linear, model_path_linear);
    model_linear->to(device);

    std::string location = "/media/stree/localssd/DL_data/cat_dog/Brittany_02625.jpg";
    print_probabilities(location, model_path, model_path_linear, device);
    /*
     * Tf_model.pt
     Printing for image: ./data/cat_dog/Brittany_02625.jpg Cat prob = -92.1491; Dog prob = 8.07676
     *
     * resnet18_without_last_layer.pt
	 Printing for image: ./data/cat_dog/Brittany_02625.jpg Cat prob = -92.1491; Dog prob = 8.07676
     */

    for( int i = 0; i < img_list.size(); i++ ) {
    	// Print probabilities for dog and cat classes
    	print_probabilities(img_list.at(i), model_path, model_path_linear, device);
    }

    std::cout << "Done!\n";
    return 0;
}

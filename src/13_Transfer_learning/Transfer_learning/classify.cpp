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
#include <dirent.h>
#include <unistd.h>

// Utility function to load image from given folder
// File type accepted: .jpg
std::vector<std::string> load_images(std::string folder_name) {
    std::vector<std::string> list_images;
    std::string base_name = folder_name;
    DIR* dir;
    struct dirent *ent;
    //std::cout << base_name.c_str() << std::endl;
    dir = opendir(base_name.c_str());
    //std::cout << dir << std::endl;

    if(dir != NULL) {
        while((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                std::string newf = base_name + filename;
                //std::cout << newf << std::endl;
                // --- Exclude empty image
                cv::Mat temp = cv::imread(newf, 1);
                if( ! temp.empty() ) list_images.push_back(newf);
            }
        }
    }
    return list_images;
}

void print_probabilities(std::string loc, std::string model_path, std::string model_path_linear) {
    // Load image with OpenCV.
	std::cout << loc << std::endl;
    cv::Mat img = cv::imread(loc.c_str(), 1); // 0 - gray image; 1 - color image
    cv::resize(img, img, cv::Size(224, 224)); //, cv::INTER_CUBIC); // cv::INTER_CUBIC cause: error: (-215:Assertion failed) !ssize.empty() in function 'resize'

    // Convert the image and label to a tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, torch::kByte);
    //torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3 }, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
    img_tensor = img_tensor.to(torch::kF32);
    
    // Load the model.
    torch::jit::script::Module model;
    model = torch::jit::load(model_path);
    
    torch::nn::Linear model_linear(512, 2);
    torch::load(model_linear, model_path_linear);
    
    // Predict the probabilities for the classes.
    std::vector<torch::jit::IValue> input;
    input.push_back(img_tensor);
    torch::Tensor prob = model.forward(input).toTensor();
    prob = prob.view({prob.size(0), -1});
    prob = model_linear(prob);
    
    std::cout << "Printing for image: " << loc << " ";                 //std::endl;
    std::cout << "Cat prob = " << *(prob.data_ptr<float>())*100. << "; "; //std::endl;
    std::cout << "Dog prob = " << *(prob.data_ptr<float>()+1)*100. << std::endl;
}

int main(int arc, char** argv) {
	// get current directory
	char tmp[256];
	getcwd(tmp, 256);
	std::cout << "Current working directory: " << tmp << std::endl;
	std::string cdir(tmp);

    // argv[1] should is the test image
    std::string img_location = cdir + "/data/cat_dog/"; //argv[1];

    std::vector<std::string> img_list = load_images(img_location);
    
    // argv[2] contains pre-trained model without last layer
    // argv[3] contains trained last FC layer
    std::string model_path = "./models/resnet18_without_last_layer.pt"; //argv[2];
    //std::string model_path = "./models/Tf_model.pt";
    std::string model_path_linear = "./models/Tf_model_linear.pt";      //argv[3];

    // Load the model.
    // You can also use: auto model = torch::jit::load(model_path);
    torch::jit::script::Module model = torch::jit::load(model_path);
    
    torch::nn::Linear model_linear(512, 2);
    torch::load(model_linear, model_path_linear);

    std::string location = "./data/cat_dog/Brittany_02625.jpg";
    print_probabilities(location, model_path, model_path_linear);
    /*
     * Tf_model.pt
     Printing for image: ./data/cat_dog/Brittany_02625.jpg Cat prob = -92.1491; Dog prob = 8.07676
     *
     * resnet18_without_last_layer.pt
	 Printing for image: ./data/cat_dog/Brittany_02625.jpg Cat prob = -92.1491; Dog prob = 8.07676
     */

    for( int i = 0; i < img_list.size(); i++ ) {
    	// Print probabilities for dog and cat classes
    	print_probabilities(img_list.at(i), model_path, model_path_linear);
    }

    std::cout << "Done!\n";
    return 0;
}

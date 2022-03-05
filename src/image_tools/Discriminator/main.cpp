#include <iostream>                    // std::cout, std::cerr
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand, std::exit
// For External Library
#include <torch/torch.h>               // torch
#include <opencv2/opencv.hpp>          // cv::Mat
#include <filesystem>
#include <unistd.h>
#include <iomanip>

// For Original Header
#include "options.hpp"                // other functions
#include "networks.hpp"              // MC_Discriminator

// Define Namespace
namespace fs = std::filesystem;

// Function Prototype
void train(Option_Arguments &vm, torch::Device &device, MC_Discriminator &model, std::vector<transforms_Compose> &transform, const std::vector<std::string> class_names);
void test(Option_Arguments &vm, torch::Device &device, MC_Discriminator &model, std::vector<transforms_Compose> &transform, const std::vector<std::string> class_names);
torch::Device Set_Device(Option_Arguments &vm);
template <typename T> void Set_Model_Params(Option_Arguments &vm, T &model, const std::string name);
std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num);
void Set_Options(Option_Arguments &vm, const std::string mode); // int argc, const char *argv[], po::options_description &args,

// -----------------------------------
// 1. Main Function
// -----------------------------------
int main(int argc, const char *argv[]){

	std::cout << "Current path is " << get_current_dir_name() << '\n';

    // (1) Extract Arguments
	Option_Arguments vm;

	std::cout << vm.dataroot << std::endl;

    // (2) Select Device
    torch::Device device = Set_Device(vm);
    std::cout << "using device = " << device << std::endl;

    // (3) Set Seed
    if( vm.seed_random ){
        std::random_device rd;
        std::srand(rd());
        torch::manual_seed(std::rand());
        if( torch::cuda::is_available() ) {
        	torch::globalContext().setDeterministicCuDNN(false);
        	torch::globalContext().setBenchmarkCuDNN(true);
        }
    }
    else{
        std::srand( vm.seed );
        torch::manual_seed(std::rand());
        if( torch::cuda::is_available() ) {
        	torch::globalContext().setDeterministicCuDNN(true);
        	torch::globalContext().setBenchmarkCuDNN(false);
        }
    }

    // (4) Set Transforms
    std::vector<transforms_Compose> transform{
        transforms_Resize(cv::Size(vm.size, vm.size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                                                  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
        transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };
    if(vm.nc == 1){
        transform.insert(transform.begin(), transforms_Grayscale(1));
    }
    
    // (5) Define Network
    MC_Discriminator model(vm);
    model->to(device);
    
    // (6) Make Directories
    std::string dir = "checkpoints/" + vm.dataset;
    fs::create_directories(dir);

    // (7) Save Model Parameters
    Set_Model_Params(vm, model, "Discriminator");

    // (8) Set Class Names
    std::vector<std::string> class_names = Set_Class_Names(vm.class_list, vm.class_num);

    // (9.1) Training Phase
    if(vm.train){
        Set_Options(vm,  "train");
        train(vm, device, model, transform, class_names);
    }

    // (9.2) Test Phase
    if(vm.test){
        Set_Options(vm, "test");
        test(vm, device, model, transform, class_names);
    }

    // End Processing
    return 0;
}


// -----------------------------------
// 2. Device Setting Function
// -----------------------------------
torch::Device Set_Device(Option_Arguments &vm){
	//std::cout << vm.gpu_id << std::endl;
    // (1) GPU Type
	if( torch::cuda::is_available() ) {
		if( vm.gpu_id >=0 ){
			torch::Device device(torch::kCUDA, vm.gpu_id);
		    return device;
		} else {
			torch::Device device(torch::kCUDA);
			return device;
		}
	} else {
		// (2) CPU Type
		torch::Device device(torch::kCPU);
		return device;
	}
}


// -----------------------------------
// 3. Model Parameters Setting Function
// -----------------------------------
template <typename T>
void Set_Model_Params(Option_Arguments &vm, T &model, const std::string name){

    // (1) Make Directory
    std::string dir = "checkpoints/" + vm.dataset + "/model_params/";
    fs::create_directories(dir);

    // (2.1) File Open
    std::string fname = dir + name + ".txt";
    std::ofstream ofs(fname);

    // (2.2) Calculation of Parameters
    size_t num_params = 0;
    for (auto param : model->parameters()){
        num_params += param.numel();
    }
    ofs << "Total number of parameters : " << (float)num_params/1e6f << "M" << std::endl << std::endl;
    ofs << model << std::endl;

    // (2.3) File Close
    ofs.close();

    // End Processing
    return;
}


// -----------------------------------
// 4. Class Names Setting Function
// -----------------------------------
std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num) {
    // (1) Memory Allocation
    std::vector<std::string> class_names = std::vector<std::string>(class_num);
    
    // (2) Get Class Names
    std::string class_name;
    std::ifstream ifs(path, std::ios::in);
    size_t i = 0;
    if( ! ifs.fail() ) {
    	while( getline(ifs, class_name) ) {
    		std::cout << class_name.length() << std::endl;
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

// -----------------------------------
// 5. Options Setting Function
// -----------------------------------
void Set_Options(Option_Arguments &vm, const std::string mode){

    // (1) Make Directory
    std::string dir = "checkpoints/" + vm.dataset + "/options/";
    fs::create_directories(dir);

    std::cout << dir << std::endl;

/*
    // (2) Terminal Output
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << args << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // (3.1) File Open
    std::string fname = dir + mode + ".txt";
    std::ofstream ofs(fname, std::ios::app);

    // (3.2) Arguments Output
    ofs << "--------------------------------------------" << std::endl;
    ofs << "Command Line Arguments:" << std::endl;
    for (int i = 1; i < argc; i++){
        if (i % 2 == 1){
            ofs << "  " << argv[i] << '\t' << std::flush;
        }
        else{
            ofs << argv[i] << std::endl;
        }
    }
    ofs << "--------------------------------------------" << std::endl;
    ofs << args << std::endl;
    ofs << "--------------------------------------------" << std::endl << std::endl;

    // (3.3) File Close
    ofs.close();
*/
    // End Processing
    return;
}


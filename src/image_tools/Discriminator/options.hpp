#pragma once

#ifndef SRC_IMAGE_TOOLS_DISCRIMINATOR_COMMON_HPP_
#define SRC_IMAGE_TOOLS_DISCRIMINATOR_COMMON_HPP_
#include <iostream>                    // std::cout, std::cerr
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <vector>                      // std::vector
#include <random>                      // std::random_device
#include <cstdlib>                     // std::srand, std::rand, std::exit

#include <torch/torch.h>
#include "loss.hpp"

#include "../transforms.hpp"              // transforms_Compose
#include "../datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

typedef struct {
	std::string dataroot = "/media/stree/localssd/DL_data";
	std::string dataset = "17_flowers"; 					// "dataset name"
	std::string class_list = "./data/17_flowers_name.txt";  	// "file name in which class names are listed"
	size_t class_num = 17; 									// "total classes"
	size_t size = 256;										// "image width and height"
	size_t nc = 3;											// "input image channel : RGB=3, grayscale=1"
	int gpu_id = -1;										// "cuda device : 'x=-1' is cpu device"
	bool seed_random = false; 								// "whether to make the seed of random number in a random"
	int seed = 0;											// "seed of random number"

	// (2) Define for Training
	bool train = true;										// "training mode on/off"
	std::string train_dir ="train";							// "training image directory : ./datasets/<dataset>/<train_dir>/<class name>/<image files>"
	size_t epochs = 5;										// "training total epoch"
	size_t batch_size = 32;									// "training batch size"
	std::string train_load_epoch = "";						// "epoch of model to resume learning"
	size_t save_epoch = 5; 									// "frequency of epoch to save model and optimizer"

	// (3) Define for Validation
	bool valid = true;										// "validation mode on/off"
	std::string valid_dir = "valid"; 						// "validation image directory : ./datasets/<dataset>/<valid_dir>/<class name>/<image files>"
	size_t valid_batch_size = 1;							// "validation batch size"
	size_t valid_freq = 1;									// "validation frequency to training epoch"

	// (4) Define for Test
	bool test = true;										// "test mode on/off"
	std::string test_dir = "test";							// "test image directory : ./datasets/<dataset>/<test_dir>/<class name>/<image files>"
	std::string test_load_epoch = "latest";					// "training epoch used for testing"
	std::string test_result_dir = "test_result";				// "test result directory : ./<test_result_dir>"

	// (5) Define for Network Parameter
	float lr = 1e-4;										// "learning rate"
	float beta1 = 0.5;										// "beta 1 in Adam of optimizer method"
	float beta2 = 0.999;									// "beta 2 in Adam of optimizer method"
	size_t nf = 64;											// "the number of filters in convolution layer closest to image"
	bool BN = true;											// "whether to use batch normalization"
} Option_Arguments;


#endif /* SRC_IMAGE_TOOLS_DISCRIMINATOR_COMMON_HPP_ */

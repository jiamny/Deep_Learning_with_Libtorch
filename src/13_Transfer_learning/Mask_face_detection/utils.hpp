#pragma once

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>


std::vector<std::string> listContents(const std::string& dirName, const std::vector<std::string>& extensions); //Lists all files in dirName with extensions.

std::vector<std::string> listFolders(const std::string& dirName); //Lists all folders in dirName with extensions.

torch::Tensor read_image(const std::string& imageName); //Loads image to Tensor with some pre-processing

torch::Tensor read_label(int label); //Converts lable to Tensor


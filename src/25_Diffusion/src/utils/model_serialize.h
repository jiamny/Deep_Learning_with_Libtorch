#pragma once

#include <torch/torch.h>
#include <fstream>

std::vector<char> get_the_bytes(const std::string& filename);

void write_bytes(std::vector<char> bytes, const std::string& filename);

void load_state_dict(const std::shared_ptr<torch::nn::Module>& model, const std::string& pt_pth);

void save_state_dict(const std::shared_ptr<torch::nn::Module>& model, const std::string& pt_pth);

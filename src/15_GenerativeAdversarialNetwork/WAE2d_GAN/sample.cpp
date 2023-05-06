#include <iostream>                    // std::cout
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <utility>                     // std::pair
#include <ios>                         // std::right
#include <iomanip>                     // std::setw, std::setfill
// For External Library
#include <torch/torch.h>               // torch
// For Original Header
#include "networks.hpp"                // WAE_Decoder

// Define Namespace
namespace fs = std::filesystem;

// -------------------
// Sampling Function
// -------------------
void sample(std::string sample_result_dir, size_t sample_total, size_t nz, torch::Device &device, WAE_Decoder &dec){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // Initialization and Declaration
    size_t total, digit;
    std::string path, result_dir, fname;
    std::stringstream ss;
    torch::Tensor z, output;

    // Get Model
    //path = "sample_load_epoch_dec.pth";
    //torch::load(dec, path, device);

    // Image Generation
    dec->eval();
    result_dir = sample_result_dir;
    fs::create_directories(result_dir);
    total = sample_total;
    digit = std::to_string(total - 1).length();
    std::cout << "total sampling images : " << total << std::endl << std::endl;
    for (size_t i = 0; i < total; i++){

        z = torch::randn({1, (long int)nz}).to(device);
        output = dec->forward(z);

        std::cout << '<' << fname << "> Generated!" << std::endl;
    }

    // End Processing
    return;
}

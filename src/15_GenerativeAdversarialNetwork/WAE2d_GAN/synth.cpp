#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <utility>                     // std::pair
// For External Library
#include <torch/torch.h>               // torch
// For Original Header
#include "networks.hpp"                // WAE_Decoder

// Define Namespace
namespace fs = std::filesystem;

// -------------------
// Synthesis Function
// -------------------
void synth(std::string synth_result_dir, size_t nz, float synth_sigma_max, float synth_sigma_inter,
		torch::Device &device, WAE_Decoder &dec){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {-1.0, 1.0};  // range of the value in output images

    // Initialization and Declaration
    size_t max_counter;
    float value;
    std::string path, result_dir;
    torch::Tensor z, output, outputs;

    // Get Model
    //path = "synth_load_epoch_dec.pth";
    //torch::load(dec, path, device);

    // Image Generation
    dec->eval();
    max_counter = (int)(synth_sigma_max / synth_sigma_inter * 2) + 1;
    z = torch::full({1, (long int)nz}, /*value=*/ - synth_sigma_max, torch::TensorOptions().dtype(torch::kFloat)).to(device);
    outputs = dec->forward(z);
    for (size_t i = 1; i < max_counter; i++){
        value = - synth_sigma_max + (float)i * synth_sigma_inter;
        z = torch::full({1, (long int)nz}, /*value=*/value, torch::TensorOptions().dtype(torch::kFloat)).to(device);
        output = dec->forward(z);
        outputs = torch::cat({outputs, output}, /*dim=*/0);
    }
    result_dir = synth_result_dir;
    fs::create_directories(result_dir);

    // End Processing
    return;

}

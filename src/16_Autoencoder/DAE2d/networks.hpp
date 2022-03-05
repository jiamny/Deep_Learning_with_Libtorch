#ifndef NETWORKS_HPP
#define NETWORKS_HPP

// For External Library
#include <torch/torch.h>

// Define Namespace
namespace nn = torch::nn;

// Function Prototype
void weights_init(nn::Module &m);
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);


// -------------------------------------------------
// struct{ConvolutionalAutoEncoderImpl}(nn::Module)
// -------------------------------------------------
struct ConvolutionalAutoEncoderImpl : nn::Module{
private:
    nn::Sequential encoder, decoder;
public:
    ConvolutionalAutoEncoderImpl(){}
    ConvolutionalAutoEncoderImpl(size_t nf, int nc, size_t nz);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ConvolutionalAutoEncoder);


#endif

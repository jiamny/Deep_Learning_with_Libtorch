#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <iostream>
#include <vector>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
void weights_init(nn::Module &m);
void DownSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);
void UpSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias=false);


// -------------------------------------------------
// struct{WAE_EncoderImpl}(nn::Module)
// -------------------------------------------------
struct WAE_EncoderImpl : nn::Module{
private:
    nn::Sequential model;
public:
    WAE_EncoderImpl(){}
    WAE_EncoderImpl(size_t nf, int size, int nc, size_t nz);
    torch::Tensor forward(torch::Tensor x);
};

// -------------------------------------------------
// struct{WAE_DecoderImpl}(nn::Module)
// -------------------------------------------------
struct WAE_DecoderImpl : nn::Module{
private:
    nn::Sequential model;
public:
    WAE_DecoderImpl(){}
    WAE_DecoderImpl(size_t nf, int size, int nc, size_t nz);
    torch::Tensor forward(torch::Tensor z);
};

// ----------------------------------------------------------
// struct{GAN_DiscriminatorImpl}(nn::Module)
// ----------------------------------------------------------
struct GAN_DiscriminatorImpl : nn::Module{
private:
    nn::Sequential model;
public:
    GAN_DiscriminatorImpl(){}
    GAN_DiscriminatorImpl(size_t nz, size_t nd, size_t n_blocks);
    torch::Tensor forward(torch::Tensor z);
};

// -------------------------------------------------
// struct{ViewImpl}(nn::Module)
// -------------------------------------------------
struct ViewImpl : nn::Module{
private:
    std::vector<long int> shape;
public:
    ViewImpl(){}
    ViewImpl(std::vector<long int> shape_);
    torch::Tensor forward(torch::Tensor x);
    void pretty_print(std::ostream& stream) const override;
};

TORCH_MODULE(WAE_Encoder);
TORCH_MODULE(WAE_Decoder);
TORCH_MODULE(GAN_Discriminator);
TORCH_MODULE(View);


#endif

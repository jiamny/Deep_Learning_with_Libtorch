#pragma once

#include <torch/torch.h>

#include <utility>

#include "rotary.h"

// A custom group norm implementation.
class GroupNormCustomImpl : public torch::nn::Module {
public:
    GroupNormCustomImpl(int n_groups, int num_channels, float eps = 1e-6, bool affine = true);

    void reset_paramters();

    torch::Tensor forward(torch::Tensor x);

private:
    int n_groups;
    int num_channels;
    float eps;
    bool affine;
    torch::Tensor weight;
    torch::Tensor bias;
};

TORCH_MODULE(GroupNormCustom);

// Create a 2x Upsample block, if `dim_out` less than 0, the output dim will be `dim`
torch::nn::Sequential Upsample(int dim, int dim_out = -1);

// Create a 2x Downsample block, if `dim_out` less than 0, the output dim will be `dim`
torch::nn::Sequential Downsample(int dim, int dim_out = -1);

// A resudual block impl, support pass `t` emb.
class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int in_c, int out_c, int emb_dim, int n_groups = 32);

    torch::Tensor forward(torch::Tensor x, const torch::Tensor &t);

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::Linear dense{nullptr};
    torch::nn::Sequential fn1;
    torch::nn::Sequential fn2;
    int out_c;
    GroupNormCustom pre_norm{nullptr};
    GroupNormCustom post_norm{nullptr};
};

TORCH_MODULE(ResidualBlock);

struct UnetOptions {

    UnetOptions(int img_height, int img_width, std::vector<int> &scales) :
            scales_(scales), img_height_(img_height), img_width_(img_width) {};

TORCH_ARG(int, img_c) = 3;
TORCH_ARG(int, img_height);
TORCH_ARG(int, img_width);
TORCH_ARG(int, emb_dim) = 64;
TORCH_ARG(int, min_pixel) = 4;
TORCH_ARG(int, n_block) = 2;
TORCH_ARG(int, n_groups) = 32;
TORCH_ARG(int, attn_resolution) = 16;
TORCH_ARG(std::vector<int>, scales);
};

class UnetImpl : public torch::nn::Module {
public:
    UnetImpl(UnetOptions &options);

    UnetImpl(UnetImpl &other);

    torch::Tensor forward(torch::Tensor x, const torch::Tensor &t);

private:
    UnetOptions options;
    int n_block;    // each block contains `n_block * (ResdualBlock)`
    int img_c;
    std::vector<int> scales; // save channel dim for each block.
    std::tuple<int, int> img_size;    // model input size.
    int n_groups;    // global GroupNorm param.
    int min_img_size; // min img_size
    int emb_dim;         // base model capacity setting, scales base on it.
    int min_pixel;       // the minimum resolution allow using downsampling
    int attn_resolution; // when resolution less than `attn_resolution` apply attention mechanism.

    torch::nn::Conv2d stem{nullptr};    // init feature extractor.
    torch::nn::ModuleDict encoder_blocks{nullptr};
    torch::nn::ModuleDict decoder_blocks{nullptr};

    void init(int img_c, std::tuple<int, int> &img_size, std::vector<int> &scales, int emb_dim, int min_pixel = 4,
              int n_block = 2, int n_groups = 32, int attn_resolution = 16);
};

TORCH_MODULE(Unet);

/**
 * A module which performs QKV attention and splits in a different order.
 */
class QKVAttentionImpl : public torch::nn::Module {
public:
    QKVAttentionImpl(int n_heads);

    /**
     * Apply QKV attention
     * @param qkv : an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs
     * @return an [N x (H * c) x T] tensor after attention.
     */
    torch::Tensor forward(const torch::Tensor &qkv) const;

private:
    int n_heads;
};

TORCH_MODULE(QKVAttention);


class AttentionBlockImpl : public torch::nn::Module {
public:
    AttentionBlockImpl(int channels,
                       int num_heads = 1,
                       int num_head_channels = -1,
                       bool use_checkpoint = false,
                       bool use_new_attention_order = false);

    torch::Tensor forward(torch::Tensor x);

    torch::Tensor _forward(torch::Tensor x);

    void zero_init_weights();

    int channels;
    int num_heads;
    GroupNormCustom norm{nullptr};
    bool use_checkpoint;
    torch::nn::Conv1d qkv{nullptr};
    QKVAttention attention{nullptr};
    torch::nn::Conv1d proj_out{nullptr};

    Rotary2D rot_pos;

};

TORCH_MODULE(AttentionBlock);
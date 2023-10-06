#pragma once

#include <torch/torch.h>

/**
 *
 * @param d_model dimension of the model
 * @param length length of positions
 * @return length * d_model position matrix
 */
torch::Tensor positional_encoding_1d(int d_model, int length, float base = 10000);

/**
 * Add rotary position info to 1d tensors.
 * @param sinusoidal sinusoidal encoding, shape of (seq_len, dim)
 * @param tensors tensor's list, which should has same shape of (batch, seq_len, dim)
 * @return tensors with position infos, shame shape as input.
 */
std::vector<torch::Tensor> apply_rotary_position_embeddings(const torch::Tensor& sinusoidal,
                                                            std::vector<torch::Tensor> tensors);

/**
 * Add rotary position info to 2d tensors.
 * @param sinusoidal sinusoidal encoding, shape of (H * W, dim)
 * @param tensors tensor's list, which should has same shape of (batch, dim, H, W)
 * @return tensors with position infos, shame shape as input.
 */
std::vector<torch::Tensor> apply_rotary_position2d_embeddings(const torch::Tensor& sinusoidal,
                                                              std::vector<torch::Tensor> tensors);

/**
 * Ro-PE 2D
 * https://kexue.fm/archives/8397
 */
class Rotary2D {
public:
    Rotary2D(int dim, float base = 10000) : dim(dim), base(base) {};
    /**
     * Getting cached/computed rotary pos embedding according to the input's shape
     * @param x
     * @return
     */
    torch::Tensor forward(const torch::Tensor& x);
private:
    int dim;
    float base;
    int h_size_cached{};
    int w_size_cached{};
    std::shared_ptr<torch::Tensor> pos_cached{nullptr};
};
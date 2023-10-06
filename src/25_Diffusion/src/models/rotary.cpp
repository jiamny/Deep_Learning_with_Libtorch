#include "rotary.h"

#include <utility>

using namespace torch::indexing;

torch::Tensor positional_encoding_1d(int d_model, int length, float base) {

    assert(d_model % 2 == 0 &&
           (std::stringstream() << "Cannot use sin/cos positional encoding with odd dim (got dim=" << d_model
                                << ")").str().c_str());

    auto pe = torch::zeros({length, d_model});
    auto position = torch::linspace(0, 999, length, torch::TensorOptions().dtype(torch::kFloat)).unsqueeze(1);

    auto div_term = torch::exp(
            torch::arange(0, d_model, 2, torch::TensorOptions().dtype(torch::kFloat)) * -(std::log(base) / d_model));

    pe.index_put_({Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
    pe.index_put_({Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));

    return pe;
}

std::vector<torch::Tensor> apply_rotary_position_embeddings(const torch::Tensor &sinusoidal,
                                                            std::vector<torch::Tensor> tensors) {

    assert(tensors.size() > 0 && "at least one input tensor");

    // [cos(theta_0), ...] -> [cos(theta_0), cos(theta_0), ...]
    auto cos_pos = sinusoidal.index({"...", Slice(1, None, 2)})
            .repeat_interleave(2, 1); // (seq_len, dim)
    auto sin_pos = sinusoidal.index({"...", Slice(0, None, 2)})
            .repeat_interleave(2, 1); // (seq_len, dim)
    cos_pos = cos_pos.expand_as(tensors.at(0));
    sin_pos = sin_pos.expand_as(tensors.at(0));

    std::vector<torch::Tensor> outputs;
    for (const auto& t: tensors) {
        auto t_r = torch::empty_like(t);
        t_r.index_put_({"...", Slice(0, None, 2)}, -t.index({"...", Slice(1, None, 2)}));
        t_r.index_put_({"...", Slice(1, None, 2)}, t.index({"...", Slice(0, None, 2)}));
        outputs.push_back(t * cos_pos + t_r * sin_pos);
    }
    return outputs;
}

std::vector<torch::Tensor> apply_rotary_position2d_embeddings(const torch::Tensor &sinusoidal,
                                                              std::vector<torch::Tensor> tensors) {
    auto t0 = tensors.at(0);
    int B = t0.size(0), D = t0.size(1), H = t0.size(2), W = t0.size(3);
    std::vector<torch::Tensor> reshaped_tensors;
    for (const auto &t: tensors) {
        reshaped_tensors.push_back(t.permute({0, 2, 3, 1}).reshape({B, -1, D}));
    }
    reshaped_tensors = apply_rotary_position_embeddings(sinusoidal, reshaped_tensors);
    std::vector<torch::Tensor> reverse_tensors;
    for (const auto &t: reshaped_tensors) {
        reverse_tensors.push_back(t.reshape({B, H, W, D}).permute({0, 3, 1, 2}).contiguous());
    }
    return reverse_tensors;
}

torch::Tensor Rotary2D::forward(const torch::Tensor &x) {
    int H = x.size(2), W = x.size(3);
    assert(H % 2 == 0 && "Apply rotary 2d position embedding requires `H` to be even number.");
    assert(W % 2 == 0 && "Apply rotary 2d position embedding requires `W` to be even number.");
    if (pos_cached == nullptr || W != w_size_cached || H != h_size_cached) {
        h_size_cached = H;
        w_size_cached = W;

        // [cos_x0, sin_x0, cos_x1, sin_x1, ...]
        auto position_x = positional_encoding_1d(H, dim / 2, base); // (H, dim // 2)
        // [cos_y0, sin_y0, cos_y1, sin_y1, ...]
        auto position_y = positional_encoding_1d(W, dim / 2, base);  // (H, dim // 2)

        position_x = position_x.reshape({H, -1, 2}); // (H, dim // 4, 2)
        position_y = position_y.reshape({W, -1, 2}); // (W, dim // 4, 2)

        // [cos_x0, sin_x0, cos_y0, sin_y0, ...]
        // position_x[:, 0::2], position_y[:, 0::2], position_x[:, 1::2], position_y[:, 1::2] # (:, dim // 2, 2)

        auto pos_tmp = torch::empty({H * W, dim}, x.options());
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                auto emb = torch::cat({
                                              position_x.index({i, Slice(0, None, 2)}),
                                              position_y.index({j, Slice(0, None, 2)}),
                                              position_x.index({i, Slice(1, None, 2)}),
                                              position_y.index({j, Slice(1, None, 2)})
                                      }, 0).flatten(-2);
                pos_tmp.index_put_({i * W + j}, emb.to(x.options()));
            }
        }
        pos_cached = std::make_shared<torch::Tensor>(pos_tmp);
    }
    return *pos_cached;
}
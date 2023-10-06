#include "unet.h"

#include <utility>
#include "../utils/hand.h"
#include "rotary.h"

GroupNormCustomImpl::GroupNormCustomImpl(int n_groups, int num_channels, float eps, bool affine) {
    this->n_groups = n_groups;
    this->num_channels = num_channels;
    this->eps = eps;
    this->affine = affine;

    if (affine) {
        weight = register_parameter("weight", torch::empty(num_channels));
        bias = register_parameter("bias", torch::empty(num_channels));
    }

    reset_paramters();
}

void GroupNormCustomImpl::reset_paramters() {
    if (affine) {
        torch::nn::init::ones_(weight);
        torch::nn::init::zeros_(bias);
    }
}

torch::Tensor GroupNormCustomImpl::forward(torch::Tensor x) {
    int b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3); // b, c, h, w
    x = x.permute({0, 2, 3, 1});
    // (b, h, w, g, f) s.t. c = g * f
    x = x.view({b, h, w, n_groups, c / n_groups});
    torch::Tensor var, mean;
    std::tie(var, mean) = torch::var_mean(x, {1, 2, 3}, true, true);
    auto norm_x = x.sub(mean).mul(torch::rsqrt(var.add(eps)));
    norm_x = norm_x.flatten(-2);
    if (affine) {
        norm_x = norm_x.mul(weight.view({1, 1, 1, -1})).add(bias.view({1, 1, 1, -1}));
    }
    return norm_x.permute({0, 3, 1, 2});
}

torch::nn::Sequential Upsample(int dim, int dim_out) {

    torch::nn::UpsampleOptions options = torch::nn::UpsampleOptions()
            .scale_factor(std::vector<double>({2, 2}))
            .mode(torch::kBilinear)
            .align_corners(false);
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(dim, default_value(dim_out, dim), {3, 3})
            .padding(1)
            .bias(false);
    torch::nn::Sequential seq(
            torch::nn::Upsample(options),
            torch::nn::Conv2d(conv_options),
            torch::nn::SiLU(),
            GroupNormCustom(32, default_value(dim_out, dim))
    );

    return seq;
}

torch::nn::Sequential Downsample(int dim, int dim_out) {

    torch::nn::Conv2dOptions conv_options =
            torch::nn::Conv2dOptions(dim, default_value(dim_out, dim), {3, 3}).padding(1).bias(false);

    torch::nn::Sequential seq(
            torch::nn::Conv2d(conv_options),
            torch::nn::SiLU(),
            GroupNormCustom(32, default_value(dim_out, dim)),
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}))
    );

    return seq;
}

ResidualBlockImpl::ResidualBlockImpl(int in_c, int out_c, int emb_dim, int n_groups) {
    this->out_c = out_c;
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, out_c, {1, 1}).bias(false));
    dense = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, out_c).bias(false));
    fn1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, {3, 3}).padding(1).bias(false)));
    fn1->push_back(torch::nn::SiLU());
    fn2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, {3, 3}).padding(1).bias(false)));
    fn2->push_back(torch::nn::SiLU());
    pre_norm = GroupNormCustom(n_groups, out_c);
    post_norm = GroupNormCustom(n_groups, out_c);

    register_module("conv", conv);
    register_module("dense", dense);
    register_module("fn1", fn1);
    register_module("fn2", fn2);
    register_module("pre_norm", pre_norm);
    register_module("post_norm", post_norm);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x, const torch::Tensor &t) {
    torch::Tensor xi;
    if (x.size(1) == out_c) {
        xi = x.clone();
    } else {
        x = conv(x);
        xi = x.clone();
    }

    x = pre_norm(x);
    x = fn1->forward(x);
    x = x + dense(t).unsqueeze(-1).unsqueeze(-1);
    x = post_norm(x);
    x = fn2->forward(x);

    return xi + x;
}


void UnetImpl::init(int img_c,
                    std::tuple<int, int> &img_size,
                    std::vector<int> &scales,
                    int emb_dim,
                    int min_pixel,
                    int n_block,
                    int n_groups,
                    int attn_resolution) {
    this->img_c = img_c;
    this->n_groups = n_groups;
    this->n_block = n_block;
    this->scales = scales;
    this->img_size = img_size;
    this->emb_dim = emb_dim;
    this->min_pixel = min_pixel;

    int img_height, img_width;
    std::tie(img_height, img_width) = img_size;
    min_img_size = std::min(img_height, img_width);

    stem = torch::nn::Conv2d(torch::nn::Conv2dOptions(img_c, emb_dim, {3, 3}).padding(1));
    auto skip_pooling = 0;
    auto cur_c = emb_dim;

    torch::OrderedDict<std::string, std::shared_ptr<Module>> enc_blocks;

    std::vector<std::tuple<int, int>> chs;

    // add enc blocks
    for (size_t i = 0; i < scales.size(); i++) {
        auto scale = scales[i];

        // sevaral residual blocks
        for (size_t j = 0; j < n_block; j++) {
            chs.emplace_back(cur_c, scale * emb_dim);
            auto block = ResidualBlock(cur_c, scale * emb_dim, emb_dim, n_groups);
            cur_c = scale * emb_dim;
            enc_blocks.insert((std::stringstream() << "enc_block_" << i * n_block + j).str(), block.ptr());
        }

        if (min_img_size <= attn_resolution) {
            enc_blocks.insert((std::stringstream() << "attn_enc_block_" << i * n_block).str(),
                              AttentionBlock(cur_c, 8, cur_c / 8).ptr());
        }

        // downsample block if not reach to `min_pixel`.
        if (min_img_size > min_pixel) {
            enc_blocks.insert((std::stringstream() << "down_block_" << i).str(), Downsample(cur_c).ptr());
            min_img_size = min_img_size / 2;
        } else {
            skip_pooling += 1; // log how many times skip pooling.
        }
    }

    // add mid blocks
    enc_blocks.insert((std::stringstream() << "enc_block_" << scales.size() * n_block).str(),
                      ResidualBlock(cur_c, cur_c, emb_dim, n_groups).ptr());
    this->encoder_blocks = torch::nn::ModuleDict(enc_blocks);

    std::reverse(chs.begin(), chs.end()); // decoder chs reversed.

    torch::OrderedDict<std::string, std::shared_ptr<Module>> dec_blocks;

    // add dec blocks, in reverse scales.
    size_t m = 0;
    for (int i = scales.size() - 1; i > -1; i--) {
        auto rev_scale = scales[i];
        if (m >= skip_pooling) {
            dec_blocks.insert((std::stringstream() << "up_block_" << m).str(), Upsample(cur_c).ptr());
            min_img_size *= 2;
        }

        for (size_t j = 0; j < n_block; j++) {
            int out_channels;
            int in_channels;
            std::tie(out_channels, in_channels) = chs[m * n_block + j];
            dec_blocks.insert((std::stringstream() << "dec_block_" << m * n_block + j).str(),
                              ResidualBlock(in_channels, out_channels, emb_dim, n_groups).ptr());
            cur_c = out_channels;
        }

        if (min_img_size <= attn_resolution) {
            dec_blocks.insert((std::stringstream() << "attn_dec_block_" << m * n_block).str(),
                              AttentionBlock(cur_c, 8, cur_c / 8).ptr());
        }

        m++;
    }

    // finaly add a to_rgb block
    torch::nn::Sequential to_rgb(
            GroupNormCustom(n_groups, cur_c),
            torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(cur_c, img_c, {3, 3}).padding(1).bias(false))
    );

    dec_blocks.insert(std::string("to_rgb"), to_rgb.ptr());
    this->decoder_blocks = torch::nn::ModuleDict(dec_blocks);

    stem = register_module("stem", stem);
    encoder_blocks = register_module("encoder_blocks", encoder_blocks);
    decoder_blocks = register_module("decoder_blocks", decoder_blocks);
}

UnetImpl::UnetImpl(UnetOptions& options): options(options) {
    auto img_size_ = std::make_tuple(options.img_height(), options.img_width());
    init(options.img_c(),
         img_size_,
         options.scales(),
         options.emb_dim(),
         options.min_pixel(), options.n_block(), options.n_groups(), options.attn_resolution());
}

UnetImpl::UnetImpl(UnetImpl &other):UnetImpl(other.options) {}

torch::Tensor UnetImpl::forward(torch::Tensor x, const torch::Tensor &t) {
    x = stem(x);

    std::vector<torch::Tensor> inners;

    inners.push_back(x);
    for (const auto &item: encoder_blocks->items()) {
        auto name = item.first;
        auto module = item.second;
        // resudial block
        if (startswith(name, "enc")) {
            x = module->as<ResidualBlock>()->forward(x, t);
            inners.push_back(x);
        } else if (startswith(name, "attn")) {
            x = module->as<AttentionBlock>()->forward(x);
        }
            // downsample block
        else {
            x = module->as<torch::nn::Sequential>()->forward(x);
            inners.push_back(x);
        }
    }

    // drop last two (contains middle block output)
    auto inners_ = std::vector<torch::Tensor>(inners.begin(), inners.end() - 2);

    for (const auto &item: decoder_blocks->items()) {
        auto name = item.first;
        auto module = item.second;

        // upsample block
        if (startswith(name, "up")) {
            x = module->as<torch::nn::Sequential>()->forward(x);
            torch::Tensor xi = inners_.back();
            inners_.pop_back(); // pop()
            x = x + xi;
        }
            // resudial block
        else if (startswith(name, "dec")) {
            torch::Tensor xi = inners_.back();
            inners_.pop_back(); // pop()
            x = module->as<ResidualBlock>()->forward(x, t);
            x = x + xi;
        } else if (startswith(name, "attn")) {

            x = module->as<AttentionBlock>()->forward(x);
        } else {
            x = module->as<torch::nn::Sequential>()->forward(x);
        }
    }

    return x;
}

QKVAttentionImpl::QKVAttentionImpl(int n_heads) : n_heads(n_heads) {
}

torch::Tensor QKVAttentionImpl::forward(const torch::Tensor &qkv) const {
    int bs = qkv.size(0), width = qkv.size(1), length = qkv.size(2);
    assert(width % (3 * n_heads) == 0);
    auto ch = width / (3 * n_heads);
    auto q_k_v = qkv.chunk(3, 1); // 3 x (B, C, -1)
    auto q = q_k_v.at(0),
            k = q_k_v.at(1),
            v = q_k_v.at(2);
    auto scale = 1 / std::sqrt(std::sqrt(ch));
    auto weight = torch::einsum("bct,bcs->bts", {
            (q * scale).reshape({bs * n_heads, ch, length}),
            (k * scale).reshape({bs * n_heads, ch, length})
    }); // More stable with f16 than dividing afterwards
    weight = torch::softmax(weight.to(torch::kFloat16), -1).to(weight.dtype());
    auto a = torch::einsum("bts,bcs->bct",
                           {weight, v.reshape({bs * n_heads, ch, length})});
    return a.reshape({bs, -1, length});
}

AttentionBlockImpl::AttentionBlockImpl(int channels, int num_heads, int num_head_channels,
                                       bool use_checkpoint, bool use_new_attention_order) : rot_pos(
        Rotary2D(channels)) {

    this->channels = channels;
    if (num_head_channels == -1) {
        this->num_heads = num_heads;
    } else {
        assert(channels % num_head_channels == 0 &&
               (std::stringstream() << "q,k,v channels " << channels << " is not divisible by num_head_channels "
                                    << num_head_channels)
                       .str().c_str());
        this->num_heads = channels / num_head_channels;
    }
    this->use_checkpoint = use_checkpoint;
    this->norm = GroupNormCustom(std::min(32, channels / 4), channels);
    this->qkv = torch::nn::Conv1d(torch::nn::Conv1dOptions(channels, channels * 3, 1));
    this->attention = QKVAttention(num_heads);
    this->proj_out = torch::nn::Conv1d(torch::nn::Conv1dOptions(channels, channels, 1));

    register_module("norm", norm);
    register_module("qkv", qkv);
    register_module("attention", attention);
    register_module("proj_out", proj_out);

    zero_init_weights();
}

void AttentionBlockImpl::zero_init_weights() {
    torch::NoGradGuard no_grad;
    for (const auto &p: proj_out->parameters()) {
        p.zero_();
    }
}

torch::Tensor AttentionBlockImpl::_forward(torch::Tensor x) {
    int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);

    x = norm->forward(x);
    auto xi = x.view({B, C, -1});
    auto qkv_ = qkv->forward(xi).permute({0, 2, 1}); // (B, -1,  3 x C)

    // -------------- Apply rotary 2d position embedding -----------
    auto q_k_v = qkv_.chunk(3, 2); // (B, -1, C)
    auto q_ = q_k_v.at(0), k_ = q_k_v.at(1), v_ = q_k_v.at(2);
    auto q_k = apply_rotary_position_embeddings(rot_pos.forward(x), {q_, k_});
    q_ = q_k.at(0);
    k_ = q_k.at(1);
    qkv_ = torch::cat({q_, k_, v_}, 2).permute({0, 2, 1}); // (B, 3 x C, -1)
    // --------------------------------------------------------------

    auto h = attention->forward(qkv_);
    h = proj_out->forward(h);
    return (xi + h).reshape({B, C, H, W});
}

torch::Tensor AttentionBlockImpl::forward(torch::Tensor x) {
    return _forward(std::move(x));
}
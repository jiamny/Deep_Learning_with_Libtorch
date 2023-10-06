#pragma once

#include <iostream>
#include <fstream>
#include "models/ddpm.h"

class Trainer {
public:
    Trainer(std::shared_ptr<Sampler> sampler,
            const std::tuple<int, int>& img_size,
            std::string &exp_name,
            int train_batch_size = 32,
            double train_lr = 2e-4,
            int train_num_epochs = 10000,
            double ema_decay = 0.995,
            int num_workers = 2,
            int save_and_sample_every = 100,
            int accumulation_steps = 2,
            bool amp_enable = false);

    void train(std::string dataset_path, std::string image_type="jpg");

private:
    std::shared_ptr<Sampler> sampler{nullptr};
    std::shared_ptr<Sampler> sampler_shadow{nullptr};
    std::tuple<int, int> img_size;
    int train_batch_size;
    double train_lr;
    int train_num_epochs;
    double ema_decay;
    int num_workers;
    int save_and_sample_every;
    int accumulation_steps;
    bool amp_enable;
    std::string sample_path;
    std::string checkpoint_path;
    torch::Device device = torch::Device(torch::kCPU);
};


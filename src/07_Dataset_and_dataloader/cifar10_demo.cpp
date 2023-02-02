// Copyright 2020-present pytorch-cpp Authors
// Original: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "cifar10.h"

int main() {

    std::cout << "Load cifar10 ...\n\n";

    // Loading and normalizing CIFAR10
    const std::string CIFAR_data_path = "/media/stree/localssd/DL_data/cifar/cifar10/";

    auto train_dataset = CIFAR10(CIFAR_data_path)
        .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), 4);

    for(auto &batch: *train_loader){
            auto data = batch.data;
            auto target = batch.target;
            std::cout << data.sizes() << " " << target << std::endl;
    }

    std::cout << "Done!\n";
    return 0;
}

#pragma once

#include <torch/torch.h>


/*
update the model_target using exponential moving average
@param model_tgt: target model
@param model_src: source model
@param beta: value of decay beta
@ret: None (updates the target model)
*/
void update_average(std::shared_ptr<torch::nn::Module> model_tgt, std::shared_ptr<torch::nn::Module> model_src,
                    double beta = 0.99);
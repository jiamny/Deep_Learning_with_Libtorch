#include "grad.h"

void toggle_grad(std::shared_ptr<torch::nn::Module> model, bool requires_grad) {
    for (auto mm: model->parameters()) {
        mm.set_requires_grad(requires_grad);
    }
}
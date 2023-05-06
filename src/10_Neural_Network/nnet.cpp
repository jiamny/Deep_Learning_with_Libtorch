// Copyright 2020-present pytorch-cpp Authors

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

class NetImpl : public torch::nn::Module {
 public:
    NetImpl();
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;

 private:
    int num_flat_features(torch::Tensor x);
};

TORCH_MODULE(Net);

NetImpl::NetImpl() :
    conv1(torch::nn::Conv2dOptions(1, 6, 3)),
    conv2(torch::nn::Conv2dOptions(6, 16, 3)),
    fc1(torch::nn::LinearOptions(16 * 6 * 6, 120)),
    fc2(torch::nn::LinearOptions(120, 84)),
    fc3(torch::nn::LinearOptions(84, 10)) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

int NetImpl::num_flat_features(torch::Tensor x) {
    auto sz = x.sizes().slice(1);  // all dimensions except the batch dimension
    int num_features = 1;
    for (auto s : sz) {
        num_features *= s;
    }
    return num_features;
}

torch::Tensor NetImpl::forward(torch::Tensor x) {
    // Max pooling over a (2, 2) window
    auto out = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 2}))->forward(torch::relu(conv1->forward(x)));
    // If the size is a square you can only specify a single number
    out = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))->forward(torch::relu(conv2->forward(out)));
    out = out.view({-1, num_flat_features(out)});
    out = torch::relu(fc1->forward(out));
    out = torch::relu(fc2->forward(out));
    return fc3->forward(out);
}


int main() {
    std::cout << "Deep Learning with PyTorch: A 60 Minute Blitz\n\n";
    std::cout << "Neural Networks\n\n";

    std::cout << "Define the network\n\n";
    Net net = Net();
    net->to(torch::kCPU);
    std::cout << net << "\n\n";

    // The learnable parameters of a model are returned by net.parameters():
    auto params = net->parameters();
    std::cout << params.size() << '\n';
    std::cout << params.at(0).sizes() << "\n\n";  // conv1's .weight

    // Let’s try a random 32x32 input:
    auto input = torch::randn({1, 1, 32, 32});
    auto out = net->forward(input);
    std::cout << out << "\n\n";

    // Zero the gradient buffers of all parameters and backprops with random gradients:
    net->zero_grad();
    out.backward(torch::randn({1, 10}));

    std::cout << "Loss Function\n\n";

    auto output = net->forward(input);
    auto target = torch::randn(10);  // a dummy target, for example
    target = target.view({1, -1});  // make it the same shape as output
    torch::nn::MSELoss criterion;
    auto loss = criterion(output, target);
    std::cout << loss << "\n\n";

    // For illustration, let us follow a few steps backward:
    std::cout << "loss.grad_fn:\n" << loss.grad_fn() << '\n';  // MSELoss

    std::cout << "Backprop\n\n";

    // Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward:
    net->zero_grad();  // zeroes the gradient buffers of all parameters
    std::cout << "conv1.bias.grad before backward:\n" << net->conv1->bias.grad() << '\n';
    loss.backward();
    std::cout << "conv1.bias.grad after backward:\n" << net->conv1->bias.grad() << "\n\n";

    std::cout << "Update the weights\n\n";

    // create your optimizer
    auto learning_rate = 0.01;
    auto optimizer = torch::optim::SGD(net->parameters(), torch::optim::SGDOptions(learning_rate));
    // in your training loop:
    optimizer.zero_grad();   // zero the gradient buffers
    output = net->forward(input);
    loss = criterion(output, target);
    loss.backward();
    optimizer.step();    // Does the update
}





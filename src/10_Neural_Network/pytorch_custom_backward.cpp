// An example of using the PyTorch C++ API to implement a custom forward and backward function

#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/functions/utils.h>


using torch::Tensor;
using at::Scalar;

using torch::autograd::Node;
//using torch::autograd::Function;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;

using torch::autograd::variable_list;
using torch::autograd::tensor_list;

//using torch::autograd::as_variable;
//using torch::autograd::as_variable_ref;

using torch::autograd::compute_requires_grad;
using torch::autograd::collect_next_edges;
using torch::autograd::flatten_tensor_args;


struct MyPowBackward : public Node {
    // Public members that we use to store the forward pass, such that we can use it in gradient calculation
    SavedVariable self_;
    Scalar exponent_;

    // The following function is called during the backward pass
    variable_list apply(variable_list&& grads) override {
        std::cout << "-> Computing MyPow Backward!" << std::endl;
        
        // Our function had one output, so we only expect 1 gradient
        auto& grad = grads[0];
        // Grab the data out of the saved variable
        auto self = self_.unpack();
        double exponent = exponent_.toDouble();

        // Variable list to hold the gradients at the function's input variables
        variable_list grad_inputs(1); 

        // Do gradient computation for each of the inputs
        if (should_compute_output(0)) {
            auto grad_result = exponent != 0.0 ? grad * exponent * self.pow(exponent - 1) : torch::zeros_like(self);
            grad_inputs[0] = grad_result;
        }

        return grad_inputs;
    }

    // Apparently we need to manually handle destruction of SavedVaribles
    void release_variables() override {
        self_.reset_data();
        //self_.reset_grad_function(); // not available in new version
    }
};


Tensor MyPowForward(const Tensor & self, Scalar exponent) {
    std::cout << "-> Computing MyPow Forward!" << std::endl;
    // Compute the function's output
    auto& self_ = self;
    auto result = self_.data().pow(exponent); // compute the output based on the tensor's data
    //auto result = as_variable(tmp);

    // Prepare the infrastructure for computing the function's gradient
    if (compute_requires_grad(self)) {
        // Initialize the gradient function
        auto grad_fn = std::shared_ptr<MyPowBackward>(new MyPowBackward(), deleteNode);

        // Connect into the autograd graph
        grad_fn->set_next_edges(collect_next_edges( self ));

        // Save the function arguments for use in the backwards pass
        grad_fn->self_ = SavedVariable(self, false);
        grad_fn->exponent_ = exponent;

        // Attach the gradient function to the result
        set_history(flatten_tensor_args( result ), grad_fn);
    }

    return result;
}


int main() {

	/*
	 * Simple C++ custom autograd function code throws error "CUDA error: driver shutting down"
	 * terminate called after throwing an instance of 'c10::Error'
	 *  what():  CUDA error: driver shutting down
	 *  CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
	 */

	auto cuda_available = torch::cuda::is_available(); // add this line will let everything OK.

    auto a = 3*torch::ones({3,3});
    a.set_requires_grad(true);

    std::cout << "Begin Forward Pass" << std::endl;
    auto b = MyPowForward(a, 2).sum();

    std::cout << "Begin Backward Pass" << std::endl;
    b.backward();

    std::cout << a.grad() << std::endl;
    std::cout << "Done!\n";

    return 0;
}

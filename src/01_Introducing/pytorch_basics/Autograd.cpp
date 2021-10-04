#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

using namespace torch::autograd;

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

    std::cout << "PyTorch autograd\n\n";

    auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    /***************************************************************
     * Autograd - simple example, requires_grad by default is false
     ***************************************************************/
    torch::Tensor x = torch::tensor(2.0, torch::requires_grad(true));
    torch::Tensor y = x + 2;
    torch::Tensor z = y.pow(2) + 3;

    // Compute the gradients
    z.backward();

    // Print out the gradients
    std::cout << x.grad() << '\n';  // x.grad() = 8

    /***************************************************************
     * Autograd - more complex example
     ***************************************************************/
    x = torch::randn({3, 4}, torch::requires_grad(true));
    y = torch::randn({3, 5}, torch::requires_grad(true));
    torch::Tensor w = torch::randn({4, 5}, torch::requires_grad(true)).to(dtype_option);

    torch::Tensor out = torch::mean(y - torch::matmul(x, w)); // torch::matmul for matrix multiply
    out.backward();
    // Print out the gradients of x (d_out/d_x)
    std::cout << x.grad() << '\n';

    // Print out the gradients of y (d_out/d_y)
    std::cout << y.grad() << '\n';

    // Print out the gradients of w (d_out/d_w)
    std::cout << "W = "  << w.grad() << '\n';

    /***************************************************************
     * Autograd - retain_graph or not
     ***************************************************************/
    x = torch::tensor(3., torch::requires_grad(true));
    y = x * 2 + x.pow(2) + 3;
    std::cout << y << '\n';

    // by default pytorch uses the calculating graph once and discard it
    //y.backward(torch::nullopt, /*keep_graph=*/ true, /*create_graph=*/ false); // save calculating graph

    std::cout << x.grad() << '\n'; // 8

    y.backward();                  // autograd again, but this time not save calculating graph

    std::cout << x.grad() << '\n'; // 16, here we did autograd twice, each time the gradient is 8 so the results = 8*2

    /***************************************************************
     * Autograd - exercise
     ***************************************************************/
    x = torch::tensor({2.0, 3.0}, torch::requires_grad(true)).to(dtype_option);

    std::cout << "x{2,3} = " << x << std::endl;

    torch::Tensor k = torch::zeros(2, torch::dtype(torch::kFloat32));
    k[0] = x[0].pow(2) + 3 * x[1];
    k[1] = x[1].pow(2) + 2 * x[0];

    // print k
    std::cout << "k = [k0, k1]\n" << k << '\n'; // [13, 13]

    torch::Tensor j = torch::zeros({2, 2}, torch::dtype(torch::kFloat32));

    k.backward(torch::tensor({1, 0}).to(torch::kFloat32), true, false);

    j[0] = x.grad();

    std::cout << "j[0] = [d_k0/d_x0, d_k0/d_x1]\n" << j[0] << '\n';

    x.grad().data().zero_();    // set the previously got gradient to zero

    k.backward(torch::tensor({0, 1}).to(torch::kFloat32));
    j[1] = x.grad().data();

    std::cout << "j[1] = [d_k1/d_x0, d_k1/d_x1]\n" << j[1] << '\n';

    // print j
    std::cout << "j = \n" << j << '\n';  // [[4, 3][2, 6]]

    /***************************************************************
     * Autograd - when we so not need backward, turn off autograd
     ***************************************************************/
    x = torch::ones(1, torch::requires_grad(true)).to(dtype_option);
    w = torch::rand(1, torch::requires_grad(true)).to(dtype_option);
    y = x * w;

    // x=1 w=1 y=1
    std::cout << "x = " << x.requires_grad() << " w = " << w.requires_grad() << " y = " <<  y.requires_grad() << '\n';

    // turn off undeclared autograding
    std::cout << "Turn off autograd ----\n";

//    torch::NoGradGuard no_guard; // has to do this!!! torch.set_set_grad_enabled(false);
    torch::autograd::AutoGradMode no_guard(false);

    x = torch::ones(1).to(dtype_option);
    w = torch::rand(1, torch::requires_grad(true)).to(dtype_option);
    y = x * w;

    std::cout << "x = " << x.requires_grad() << " w = " << w.requires_grad() << " y = " <<  y.requires_grad() << '\n';

    //t.set_grad_enabled(True)
    torch::autograd::AutoGradMode guard(true);

    // modify tensor data but don't want recorded by autograd
    torch::Tensor a = torch::ones({3,4}, torch::requires_grad(true)).to(dtype_option);
    torch::Tensor b = torch::ones({3,4}, torch::requires_grad(true)).to(dtype_option);
    torch::Tensor c = a * b;

    std::cout << "c depends a and b so it is automatically set requires_grad(true)\n";
    std::cout << "a = " << a.requires_grad() << " b = " << b.requires_grad() << " c = " <<  c.requires_grad() << '\n';

    std::cout << "If user created tensor is at leaf node of the graph, its grad_fn is None\n";
    std::cout << "a.is_leaf() = " << a.is_leaf() << " b.is_leaf() = " << b.is_leaf()
    		<< " c.is_leaf() = " <<  c.is_leaf() << '\n';

    std::cout << "a: \n";
    std::cout << a.data() << '\n';  // a tensor

    std::cout << a.data().requires_grad() << '\n'; // out of calculating graph

    torch::Tensor d = a.data().sigmoid_(); // sigmoid_ is an inplace operation，will change data in a
    std::cout << d.requires_grad() << '\n';

    std::cout << "a changed: \n";
    std::cout << a.data() << '\n';

    // use tensor.data or tensor.detach()
    std::cout << a.requires_grad() << '\n';

    // tensor=a.data, but if tensor is modified，backward may have error
    torch::Tensor tensor = a.detach();
    std::cout << tensor.requires_grad() << '\n';

    //tensor's statistics
    auto mean = tensor.mean();
	auto std = tensor.std();
	auto maximum = tensor.max();

	std::cout <<"mean = " << mean.data() << "\nstd = " << std.data() << "\nmax = " << maximum.data() << '\n';

    /***************************************************************
     * In backward non-leaf node's gradient will be cleaned
     * after calculation. To check those variable's gradients,
     * there are two ways to check gradient autograd.grad() or hook()
     ***************************************************************/

	x = torch::ones(3, torch::requires_grad(true)).to(dtype_option);
	w = torch::rand(3, torch::requires_grad(true)).to(dtype_option);
	y = x * w;

	z = y.sum();

	std::cout << "==>>> y depends w，and w.requires_grad = true\n";
	std::cout << x.requires_grad() << '\n';
	std::cout << w.requires_grad() << '\n';
	std::cout << y.requires_grad() << '\n';

	// y.grad is None
	std::cout << "==>>> non-leaf node tensor reset after getting gradient，y.grad will be None\n";
	z.backward();
	std::cout << "x.grad() = \n" << x.grad() << '\n';
	std::cout << "w.grad() = \n" << w.grad() << '\n';
	std::cout << "y.grad() = \n" << y.grad() << '\n';

	// ----------------------------------------------------------------
	// First method：use grad to get gradient
	// ----------------------------------------------------------------
	x = torch::ones(3, torch::requires_grad(true)).to(dtype_option);
	w = torch::rand(3, torch::requires_grad(true)).to(dtype_option);
	y = x * w;

	z = y.sum();
	//imply call backward() to get d_z/d_y gradient
//	at::Tensor grad_outputs = torch::zeros(3, torch::requires_grad(true)).to(dtype_option);
//	torch::autograd::grad(z, y, t_lst, torch::nullopt, false, false);
//	std::cout << "y.grad = " << y.grad() << '\n';

	// ----------------------------------------------------------------
	// Second method：use hook to get gradient
	// ----------------------------------------------------------------
	x = torch::ones(3, torch::requires_grad(true)).to(dtype_option);
	w = torch::rand(3, torch::requires_grad(true)).to(dtype_option);
	y = x * w;

	// define hook - hook is a function，input gradient，no return value
	std::function<void(Variable)> variable_hook([](Variable grad){
		std::cout << "Hook the grad = \n" << grad.data() << '\n';
	});

	// register hook
	auto hook_handle = y.register_hook(variable_hook);
	z = y.sum();
	z.backward();

	//unless you will use the hook every time, otherwise remember to remove the hook after use.
	y.remove_hook(hook_handle);

	// -------------------------------------------------------------------------------
	// the difference between variable's grad andbackward()'s grad_variables
	//
	// - variable 𝐱's gradient is the gradient of target function 𝑓(𝑥) to 𝐱.
	// - in y.backward(grad_variables), the grad_variables = ∂𝑧/∂𝑦 of
	//   the chain ∂𝑧/∂𝑥=(∂𝑧/∂𝑦)*(∂𝑦/∂𝑥). z.backward() equally to y.backward(grad_y).
	// -------------------------------------------------------------------------------

	x = torch::arange(0.0, 3.0, torch::requires_grad(true)).to(dtype_option);
	std::cout << "x = \n" << x.data() << '\n';
	y = x.pow(2) + x*2;
	z = y.sum();
	z.backward(); // backward from z
	std::cout << "Backward from z and x.grad() = \n" << x.grad() << '\n'; // [2, 4, 6]

	x = torch::arange(0.0, 3.0, torch::requires_grad(true)).to(dtype_option);
	y = x.pow(2) + x*2;
	z = y.sum();
	auto y_gradient = torch::tensor({1,1,1}).to(dtype_option); // dz/dy
	y.backward(y_gradient); // backward from y
	std::cout << "Backward from y and x.grad() = \n" << x.grad() << '\n'; // [2, 4, 6]

	std::cout << "Done!\n";
    return(0);
}

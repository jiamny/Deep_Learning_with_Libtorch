#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <map>


int main() {

	/*
	 * Simple C++ custom autograd function code throws error "CUDA error: driver shutting down"
	 * terminate called after throwing an instance of 'c10::Error'
	 *  what():  CUDA error: driver shutting down
	 *  CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
	 */
	torch::cuda::is_available();  // add this line will let everything OK.


	/***************************************************************************
	 * 一，利用backward方法求导数
	 * backward 方法通常在一个标量张量上调用，该方法求得的梯度将存在对应自变量张量的grad属性下。
	 * 如果调用的张量非标量，则要传入一个和它同形状 的gradient参数张量。
	 * 相当于用该gradient参数张量与调用张量作向量点乘，得到的标量结果再反向传播。
	 */
	// 1, 标量的反向传播
	// # f(x) = a*x**2 + b*x + c的导数
	std::cout << "\n一，利用backward方法求导数:\n";

	auto x = torch::tensor(0.0, torch::requires_grad(true)); // x 需要被求导
	auto a = torch::tensor(1.0);
	auto b = torch::tensor(-2.0);
	auto c = torch::tensor(1.0);
	auto y = a*torch::pow(x,2) + b*x + c;

	y.backward();
	auto dy_dx = x.grad();
	std::cout << "dy_dx: " << dy_dx << std::endl;

	// 2, 非标量的反向传播
	//# f(x) = a*x**2 + b*x + c

	x = torch::tensor({{0.0,0.0}, {1.0,2.0}}, torch::requires_grad(true)); // x 需要被求导
	a = torch::tensor(1.0);
	b = torch::tensor(-2.0);
	c = torch::tensor(1.0);
	y = a*torch::pow(x,2) + b*x + c;

	auto gd = torch::tensor({{1.0,1.0}, {1.0,1.0}});

	std::cout << "x:\n" << x << std::endl;
	std::cout << "y:\n" << y << std::endl;
	y.backward(gd);
	auto x_grad = x.grad();
	std::cout << "x_grad:\n" << x_grad << std::endl;

	// 3, 非标量的反向传播可以用标量的反向传播实现
	//# f(x) = a*x**2 + b*x + c

	x = torch::tensor({{0.0,0.0}, {1.0,2.0}}, torch::requires_grad(true)); // x 需要被求导
	a = torch::tensor(1.0);
	b = torch::tensor(-2.0);
	c = torch::tensor(1.0);
	y = a*torch::pow(x,2) + b*x + c;

	auto gradient = torch::tensor({{1.0,1.0}, {1.0,1.0}});
	auto z = torch::sum(y*gradient);

	std::cout << "x:\n" << x << std::endl;
	std::cout << "y:\n" << y << std::endl;
	z.backward();
	x_grad = x.grad();
	std::cout << "x_grad:\n" << x_grad << std::endl;

	/***************************************************
	 * 二，利用autograd.grad方法求导数
	 */
	// f(x) = a*x**2 + b*x + c的导数
	std::cout << "\n二，利用autograd.grad方法求导数:\n";

	x = torch::tensor(0.0, torch::requires_grad(true));	// x 需要被求导
	a = torch::tensor(1.0);
	b = torch::tensor(-2.0);
	c = torch::tensor(1.0);
	y = a*torch::pow(x,2) + b*x + c;

	// create_graph 设置为 True 将允许创建更高阶的导数
	dy_dx = torch::autograd::grad({y}, {x}, {}, c10::nullopt, true)[0];
	std::cout << "dy_dx.data:\n" << dy_dx.data() << std::endl;

	// 求二阶导数
	std::cout << "\n求二阶导数:\n";
	auto dy2_dx2 = torch::autograd::grad({dy_dx}, {x})[0];
	std::cout << "dy2_dx2.data:\n" << dy2_dx2.data() << std::endl;

	auto x1 = torch::tensor(1.0, torch::requires_grad(true)); // x需要被求导
	auto x2 = torch::tensor(2.0, torch::requires_grad(true));

	auto y1 = x1*x2;
	auto y2 = x1+x2;


	// 允许同时对多个自变量求导数
	std::cout << "\n允许同时对多个自变量求导数\n";
	std::vector<at::Tensor> rlt = torch::autograd::grad({y1}, {x1,x2},{}, true);
	std::cout << "dy1_dx1:\n" << rlt[0] << std::endl;
	std::cout << "dy1_dx2:\n" << rlt[0] << std::endl;

	// 如果有多个因变量，相当于把多个因变量的梯度结果求和
	std::cout << "\n如果有多个因变量，相当于把多个因变量的梯度结果求和:\n";
	rlt = torch::autograd::grad({y1,y2}, {x1,x2});
	std::cout << "dy12_dx1:\n" << rlt[0] << std::endl;
	std::cout << "dy12_dx2:\n" << rlt[0] << std::endl;

	/**************************************************
	 * 三，利用自动微分和优化器求最小值
	 */
	// f(x) = a*x**2 + b*x + c的最小值
	std::cout << "\n三，利用自动微分和优化器求最小值:\n";

	x = torch::tensor(0.0, torch::requires_grad(true)); 		// x 需要被求导
	a = torch::tensor(1.0);
	b = torch::tensor(-2.0);
	c = torch::tensor(1.0);

	auto optimizer = torch::optim::SGD(/*params=*/{x}, /*lr =*/ 0.01);

/*
def f(x):
    result = a*torch.pow(x,2) + b*x + c
    return(result)
*/
    for( int i = 0; i < 500; i++ ) {
    	optimizer.zero_grad();
    	//y = f(x)
    	y = a*torch::pow(x,2) + b*x + c;
		y.backward();
		optimizer.step();
    }
    std::cout << "y=\n" << (a*torch::pow(x,2) + b*x + c).data().item<float>() << "\n" << "x=\n" << x.data().item<float>() << std::endl;

	return 0;
}



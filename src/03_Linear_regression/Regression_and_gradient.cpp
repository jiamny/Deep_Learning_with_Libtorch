/*
 * Regression_and_gradient.cpp
 */
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cstdio>

#include "../matplotlibcpp.h"

using namespace torch::autograd;
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

torch::Tensor multi_linear(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
	return( torch::add(torch::mm(x, w), b));
}

// 计算误差
torch::Tensor get_loss(torch::Tensor y_pred, torch::Tensor y_train){
	return torch::mean( torch::sub(y_pred, y_train).pow(2) );
}


int main() {
    std::cout << "Linear Regression\n\n";
    std::cout << "Training on CPU.\n";

    auto dtype_option = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

	//# 定义一个多变量函数
    double array [] = {0.5, 3.0, 2.4};
    //auto w_target = torch::from_blob(array, {3}, dtype_option); // 定义参数
    auto w_target = torch::tensor({0.5, 3.0, 2.4}, dtype_option); // 定义参数
    auto b_target = torch::tensor({0.9}, dtype_option);           // 定义参数

    std::printf("函数的式子:  y = %.2f + %.2f * x + %.2f * x^2 + %.2f * x^3", b_target.item<double>(),
    		w_target[0].item<double>(), w_target[1].item<double>(), w_target[2].item<double>()); // 打印出函数的式子

    plt::figure_size(1200, 800);
    plt::subplot2grid(2, 2, 0, 0, 1, 1);  // 2 rows, 2 column,

    // 画出这个函数的曲线
    auto x_sample = torch::arange(-3.0, 3.1, 0.1, dtype_option);
    //auto x_sample = torch::tensor({-3.0, 3.1, 0.1}, dtype_option);
    auto f1 = torch::mul(x_sample, w_target[0].item<double>());
    auto f2 = torch::mul(x_sample.pow(2), w_target[1].item<double>());
    auto f3 = torch::mul(x_sample.pow(3), w_target[2].item<double>());
	torch::Tensor y_sample = torch::add(torch::add(torch::add(f3, f2), f1), b_target.item<double>());
//	std::cout << "x_sample = " << x_sample << '\n';
//	std::cout << "y_sample = " << y_sample << '\n';

	std::vector<double> xx(x_sample.data_ptr<double>(), x_sample.data_ptr<double>() + x_sample.numel());
	std::vector<double> yy(y_sample.data_ptr<double>(), y_sample.data_ptr<double>() + y_sample.numel());
	plt::plot(xx, yy);
	plt::title("Original curve");
//	plt::show();

	// # 构建数据 x 和 y
	// # x 是一个如下矩阵 [x, x^2, x^3]
	// # y 是函数的结果 [y]

	auto x_sample2 = x_sample.pow(2);
	auto x_sample3 = x_sample.pow(3);
	auto x_train = torch::stack({x_sample, x_sample2, x_sample3}, 1).to(dtype_option); // default is  61 x 3
	std::cout << x_train.sizes() << '\n';
	//auto y_train = torch::tensor(yy, dtype_option).view({63,1});

	// # 定义参数和模型
	auto w = torch::randn({3, 1}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU).requires_grad(true));
	auto b = torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU).requires_grad(true));

	// # 画出更新之前的模型
	auto y_pred = multi_linear(x_train, w, b);
	std::cout << "y_pred = " << y_pred.sizes() << '\n';

	auto x_train1 = x_train.index({Slice(),0}).to(dtype_option).unsqueeze(1); //   view({63, 1});
//std::cout << "x_train1 = " << x_train1.data() << '\n';
	std::vector<double> xx2(x_train1.data_ptr<double>(),
			x_train1.data_ptr<double>() + x_train1.numel());
//std::cout << xx2 << '\n';
	std::vector<double> yy2(y_pred.data_ptr<double>(), y_pred.data_ptr<double>() + y_pred.numel());

	plt::subplot2grid(2, 2, 0, 1, 1, 1);
	plt::named_plot("real curve", xx, yy, "r");
	plt::named_plot("pred curve", xx, yy2, "b");
	plt::legend();
	plt::title("Real vs. predicted");
//	plt::show();

	auto y_train = torch::unsqueeze(y_sample, 1);

	// # 计算误差 get_loss
	auto loss = get_loss(y_pred, y_train);
	std::cout << "Loss = " << loss.item<double>() << '\n';
	std::cout << "w ==== " << w.grad() << '\n';

	// 自动求导
	loss.backward();

	// 查看一下 w 和 b 的梯度
	std::cout << "w.grad = " << w.grad() << '\n';
	std::cout << "b.grad = " << b.grad() << '\n';

	// 更新一下参数
	w.data().sub_(0.001 * w.grad());
	b.data().sub_(0.001 * b.grad());

	// 画出更新一次之后的模型
	auto y_pred2 = multi_linear(x_train, w, b);

	plt::subplot2grid(2, 2, 1, 0, 1, 1);
	std::vector<double> yy3(y_pred2.data_ptr<double>(), y_pred2.data_ptr<double>() + y_pred2.numel());
	plt::named_plot("fitting curve", xx, yy3, "b");
	plt::named_plot("real curve", xx, yy, "r");
	plt::legend();
	plt::title("After one epoch - real vs. fitting");
//	plt::show();

	//  进行 100 次参数更新
	for( int e =0; e < 100; e++ ){
		y_pred = multi_linear(x_train, w, b);
		loss = get_loss(y_pred, y_train);

		w.grad().zero_();
		b.grad().zero_();

		loss.backward();

		// 更新参数
		w.data().sub_(0.001 * w.grad());
		b.data().sub_(0.001 * b.grad());

		if( (e + 1) % 20 == 0 )
			std::cout << "Epoch: " << (e+1) << ", loss: " <<  loss.item<double>() << '\n';
	}

	// 查看一下 w 和 b 的梯度
	std::cout << "w.grad = " << w.grad() << '\n';
	std::cout << "b.grad = " << b.grad() << '\n';

	// 画出更新之后的结果
	auto y_pred3 = multi_linear(x_train, w, b);

	plt::subplot2grid(2, 2, 1, 1, 1, 1);
	std::vector<double> yy4(y_pred3.data_ptr<double>(), y_pred3.data_ptr<double>() + y_pred3.numel());
	plt::named_plot("fitting curve", xx, yy4, "b");
	plt::named_plot("real curve", xx, yy, "r");
	plt::legend();
	plt::title("After 100 epochs - real vs. fitting");
	plt::show();

    return 0;
}



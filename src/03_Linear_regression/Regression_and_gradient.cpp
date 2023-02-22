/*
 * Regression_and_gradient.cpp
 */
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <matplot/matplot.h>

using namespace matplot;
using namespace torch::autograd;

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
    std::cout << "Regression and gradient\n\n";

	// Device
	torch::Device device = torch::Device(torch::kCPU);
    auto dtype_option = torch::TensorOptions().dtype(torch::kDouble).device(device);

	//# 定义一个多变量函数
    double array [] = {0.5, 3.0, 2.4};
    //auto w_target = torch::from_blob(array, {3}, dtype_option); // 定义参数
    auto w_target = torch::tensor({0.5, 3.0, 2.4}, dtype_option); // 定义参数
    auto b_target = torch::tensor({0.9}, dtype_option);           // 定义参数

    std::printf("函数的式子:  y = %.2f + %.2f * x + %.2f * x^2 + %.2f * x^3", b_target.item<double>(),
    		w_target[0].item<double>(), w_target[1].item<double>(), w_target[2].item<double>()); // 打印出函数的式子

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

	// # 构建数据 x 和 y
	// # x 是一个如下矩阵 [x, x^2, x^3]
	// # y 是函数的结果 [y]

	auto x_sample2 = x_sample.pow(2);
	auto x_sample3 = x_sample.pow(3);
	auto x_train = torch::stack({x_sample, x_sample2, x_sample3}, 1).to(dtype_option); // default is  61 x 3
	std::cout << x_train.sizes() << '\n';
	//auto y_train = torch::tensor(yy, dtype_option).view({63,1});

	// # 定义参数和模型
	auto w = torch::randn({3, 1}, torch::TensorOptions().dtype(torch::kDouble).device(device).requires_grad(true));
	auto b = torch::zeros({1}, torch::TensorOptions().dtype(torch::kDouble).device(device).requires_grad(true));

	// # 画出更新之前的模型
	auto y_pred = multi_linear(x_train, w, b);
	std::cout << "y_pred = " << y_pred.sizes() << '\n';

	auto x_train1 = x_train.index({Slice(),0}).to(dtype_option).unsqueeze(1); //   view({63, 1});
//std::cout << "x_train1 = " << x_train1.data() << '\n';
	std::vector<double> xx2(x_train1.data_ptr<double>(),
			x_train1.data_ptr<double>() + x_train1.numel());
//std::cout << xx2 << '\n';
	std::vector<double> yy2(y_pred.data_ptr<double>(), y_pred.data_ptr<double>() + y_pred.numel());

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

	std::vector<double> yy3(y_pred2.data_ptr<double>(), y_pred2.data_ptr<double>() + y_pred2.numel());

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

	std::vector<double> yy4(y_pred3.data_ptr<double>(), y_pred3.data_ptr<double>() + y_pred3.numel());

	tiledlayout(2, 2);
	auto ax1 = nexttile();
	plot(ax1, xx, yy);
	matplot::title(ax1, "Original curve");

	auto ax2 = nexttile();
	plot(ax2, xx, yy);
	matplot::title(ax2, "Real vs. predicted");
	hold(ax2, true);
	plot(ax2, xx, yy2, "-.r");
	hold(ax2, false);

	auto ax3 = nexttile();
	plot(ax3, xx, yy);
	matplot::title(ax3, "One epoch - real vs. fitted");
	hold(ax3, true);
	plot(ax3, xx, yy3, "-.r");
	hold(ax3, false);
	legend(ax3, "real", "fitted");

	auto ax4 = nexttile();
	plot(ax4, xx, yy);
	matplot::title(ax4, "100 epochs - real vs. fitted");
	hold(ax4, true);
	plot(ax4, xx, yy4, "-.r");
	hold(ax4, false);
	legend(ax4, "real", "fitted");

	matplot::show();

	std::cout<< "Done!\n";
    return 0;
}



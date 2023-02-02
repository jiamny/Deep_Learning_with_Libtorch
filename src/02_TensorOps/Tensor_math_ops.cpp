#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	torch::manual_seed(1000);

	/**************************************************************
	 * 一，标量运算
	 *
	 * 张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。
	 * 标量运算符的特点是对张量实施逐元素运算。有些标量运算符对常用的数学运算符进行了重载。并且支持类似numpy的广播特性。
	 */
	auto a = torch::tensor({{1.0,2.0}, {-3.0,4.0}}).to(torch::kDouble);
	auto b = torch::tensor({{5.0,6.0}, {7.0,8.0}}).to(torch::kDouble);

	std::cout << "a+b:\n" << a+b << std::endl; //运算符重载
	std::cout << "a-b:\n" << a-b << std::endl;
	std::cout << "a*b:\n" << a*b << std::endl;
	std::cout << "a/b:\n" << a/b << std::endl;
	std::cout << "a**2:\n" << torch::pow(a, 2) << std::endl;
	std::cout << "a**2:\n" << a.pow(2) << std::endl;
	std::cout << "a**0.5:\n" << a.pow(0.5) << std::endl;
	std::cout << "a%3:\n" << a % 3 << std::endl;
	std::cout << "a//3:\n" << a.fmod(3) << std::endl;
	std::cout << "a>=2:\n" << (a >= 2) << std::endl;		 // torch::ge(a,2) //ge: greater_equal缩写
	std::cout << "(a>=2)&(a<=3):\n" << ((a>=2)&(a<=3)) << std::endl;
	std::cout << "(a>=2)|(a<=3):\n" << ((a>=2)|(a<=3)) << std::endl;
	std::cout << "(a == 5):\n" << (a == 5) << std::endl;
	std::cout << "sqrt(a):\n" << torch::sqrt(a) << std::endl;

	a = torch::tensor({1.0,8.0});
	b = torch::tensor({5.0,6.0});
	auto c = torch::tensor({6.0,7.0});
	std::cout << "a+b+c:\n" << (a+b+c) << std::endl;
	std::cout << "torch::max(a,b):\n" << torch::max(a,b) << std::endl;
	std::cout << "torch::min(a,b):\n" << torch::min(a,b) << std::endl;

	auto x = torch::tensor({2.6,-2.7});

	std::cout << "torch::round(x):\n" << torch::round(x) << std::endl; //保留整数部分，四舍五入
	std::cout << "torch::floor(x):\n" << torch::floor(x) << std::endl; //保留整数部分，向下归整
	std::cout << "torch::ceil(x):\n" << torch::ceil(x) << std::endl;   //保留整数部分，向上归整
	std::cout << "torch::trunc(x):\n" << torch::trunc(x) << std::endl; //保留整数部分，向0归整

	x = torch::tensor({2.6,-2.7});
	std::cout << "torch::fmod(x,2):\n" << torch::fmod(x,2) << std::endl; 			//作除法取余数
	std::cout << "torch::remainder(x,2):\n" << torch::remainder(x,2) << std::endl;  //作除法取剩余的部分，结果恒正

	// 幅值裁剪
	x = torch::tensor({0.9,-0.8,100.0,-20.0,0.7});
	auto y = torch::clamp(x, /*min=*/-1, /*max =*/ 1);
	auto z = torch::clamp(x, /*max =*/ 1);
	std::cout << "y:\n" << y << std::endl;
	std::cout << "z:\n" << z << std::endl;

	/******************************************************************
	 * 二，向量运算
	 *
	 * 向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。
	 */
	//统计值

	a = torch::arange(1,10).to(torch::kFloat);
	std::cout << "torch::sum(a):\n" << torch::sum(a) << std::endl;
	std::cout << "torch::mean(a):\n" << torch::mean(a) << std::endl;
	std::cout << "torch::max(a):\n" << torch::max(a) << std::endl;
	std::cout << "torch::min(a):\n" << torch::min(a) << std::endl;
	std::cout << "torch::prod(a):\n" << torch::prod(a) << std::endl; 	// 累乘
	std::cout << "torch::std(a):\n" << torch::std(a) << std::endl; 		// 标准差
	std::cout << "torch::var(a):\n" << torch::var(a) << std::endl;		// 方差
	std::cout << "torch::median(a):\n" << torch::median(a) << std::endl;// 中位数

	// 指定维度计算统计值
	b = a.view({3,3});
	std::cout << "b:\n" << b << std::endl;
	std::tuple<torch::Tensor, torch::Tensor> max_test0 = torch::max(b, /*dim =*/ 0);
	std::cout << "torch::max(b, 0).values:\n" << std::get<0>(max_test0) << std::endl;
	std::cout << "torch::max(b, 0).index:\n" << std::get<1>(max_test0) << std::endl;
	std::tuple<torch::Tensor, torch::Tensor> max_test1 = torch::max(b, /*dim =*/ 1);
	std::cout << "torch::max(b, 1).values:\n" << std::get<0>(max_test1) << std::endl;
	std::cout << "torch::max(b, 1).index:\n" << std::get<1>(max_test1) << std::endl;

	// #cum扫描
	a = torch::arange(1,10);
	std::cout << "torch::cumsum(a,0):\n" << torch::cumsum(a, 0) << std::endl;
	std::cout << "torch::cumprod(a,0):\n" << torch::cumprod(a, 0) << std::endl;
	std::tuple<torch::Tensor, torch::Tensor> cummax_0 = torch::cummax(a, 0);
	std::cout << "torch::cummax(a,0).values:\n" << std::get<0>(cummax_0) << std::endl;
	std::cout << "torch::cummax(a,0).index:\n" << std::get<1>(cummax_0) << std::endl;
	std::tuple<torch::Tensor, torch::Tensor> cummin_0 = torch::cummin(a, 0);
	std::cout << "torch::cummin(a,0).values:\n" << std::get<0>(cummin_0) << std::endl;
	std::cout << "torch::cummin(a,0).index:\n" << std::get<1>(cummin_0) << std::endl;

	/********************************************************************
	 * 三，矩阵运算
	 *
	 * 矩阵必须是二维的。类似torch.tensor([1,2,3])这样的不是矩阵。
	 * 矩阵运算包括：矩阵乘法，矩阵转置，矩阵逆，矩阵求迹，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。
	 */
	// 矩阵乘法
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}});
	b = torch::tensor({{2.0,0.0}, {0.0,2.0}});
	std::cout << "(a@b):\n" << torch::mm(a,b) << std::endl;  // 等价于torch.matmul(a,b) 或 torch.mm(a,b), a@b

	// 矩阵转置
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}});
	std::cout << "a.t():\n" << a.t() << std::endl;

	// 矩阵逆，必须为浮点类型
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}});
	std::cout << "a.inverse():\n" << torch::inverse(a) << std::endl;

	// 矩阵求trace
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}});
	std::cout << "a.trace():\n" << torch::trace(a) << std::endl;

	// 矩阵求范数
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}});
	std::cout << "a.norm():\n" << torch::norm(a) << std::endl;

	// 矩阵行列式
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}});
	std::cout << "a.det():\n" << torch::det(a) << std::endl;

	// 矩阵特征值和特征向量
	a = torch::tensor({{1.0,2.0}, {-5.0, 4.0}}, torch::kFloat);
	std::tuple<torch::Tensor, torch::Tensor> eig_rlt = torch::linalg::eig(a);
	std::cout << "torch.eig(a, true):\n" << std::get<0>(eig_rlt) << std::endl;
	std::cout << "torch.eig(a, true):\n" << std::get<1>(eig_rlt) << std::endl;

	// 矩阵svd分解
	// svd分解可以将任意一个矩阵分解为一个正交矩阵u,一个对角阵s和一个正交矩阵v.t()的乘积
	// svd常用于矩阵压缩和降维
	a = torch::tensor({{1.0,2.0}, {3.0,4.0}, {5.0,6.0}});
	torch::Tensor u, s, v;
	std::tie(u,s,v) = torch::svd(a);
	std::cout << "u:\n" << u << std::endl;
	std::cout << "s:\n" << s << std::endl;
	std::cout << "v:\n" << v << std::endl;

	std::cout << "u@torch.diag(s)@v.t():\n" << torch::mm(u, torch::mm(torch::diag(s), v.t())) << std::endl;

	/*****************************************************************
	 * 四，广播机制
	 *
	 *Pytorch的广播规则和numpy是一样的:
	 * 1、如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样。
	 * 2、如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的。
	 * 3、如果两个张量在所有维度上都是相容的，它们就能使用广播。
	 * 4、广播之后，每个维度的长度将取两个张量在该维度长度的较大值。
	 * 5、在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。

	 * torch.broadcast_tensors可以将多个张量根据广播规则转换成相同的维度。
	 */
	a = torch::tensor({1,2,3});
	b = torch::tensor({{0,0,0},{1,1,1},{2,2,2}});
	print(b + a);

	std::vector<torch::Tensor> bts = torch::broadcast_tensors({a,b});
	auto a_broad = bts[0];
	auto b_broad = bts[1];
	std::cout << "a_broad:\n" << a_broad << std::endl;
	std::cout << "b_broad:\n" << b_broad << std::endl;
	std::cout << "a_broad + b_broad:\n" << (a_broad + b_broad) << std::endl;

	std::cout << "\nDone!\n";
	return 0;
}





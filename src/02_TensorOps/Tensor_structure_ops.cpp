#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <vector>


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	torch::manual_seed(1000);

	// 张量的结构操作 - 张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。
	auto a = torch::tensor({1,2,3}, dtype_option);
	std::cout << "a:\n" << a << std::endl;

	auto b = torch::arange(1,10, /*step =*/ 2);
	std::cout << "b:\n" << b << std::endl;

	auto c = torch::linspace(0.0, 2*3.14, 10);
	std::cout << "c:\n" << c << std::endl;

	auto d = torch::zeros({3,3});
	std::cout << "d:\n" << d << std::endl;

	a = torch::ones({3,3}).to(torch::kInt);
	b = torch::zeros_like(a, torch::kFloat);
	std::cout << "a:\n" << a << std::endl;
	std::cout << "b:\n" << b << std::endl;

	torch::fill_(b,5);
	std::cout << "torch::fill_(b,5):\n" << b << std::endl;

	// 均匀随机分布
	torch::manual_seed(0);
	int minval = 0, maxval = 10;
	a = minval + (maxval-minval)*torch::rand({5});
	std::cout << "a:\n" << a << std::endl;

	// 正态分布随机
	b = at::normal(/*mean =*/torch::zeros({3,3}), /*std =*/ torch::ones({3,3}));
	std::cout << "b:\n" << b << std::endl;

	// 正态分布随机
	float mean = 2, std = 5;
	c = std*torch::randn({3,3})+mean;
	std::cout << "c:\n" << c << std::endl;

	// 整数随机排列
	d = torch::randperm(20);
	std::cout << "d:\n" << d << std::endl;

	// 特殊矩阵
	auto I = torch::eye(3,3); 						//单位矩阵
	std::cout << "I:\n" << I << std::endl;
	auto t = torch::diag(torch::tensor({1,2,3}));   //对角矩阵
	std::cout << "t:\n" << t << std::endl;

	/****************************************************************
	 * 二 ，索引切片
	 *
	 * 张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。可以通过索引和切片对部分元素进行修改。
	 * 此外，对于不规则的切片提取,可以使用torch.index_select, torch.masked_select, torch.take
	 * 如果要通过修改张量的某些元素得到新的张量，可以使用torch.where,torch.masked_fill,torch.index_fill
	 */
	//均匀随机分布
	torch::manual_seed(0);
	t = torch::floor(minval + (maxval-minval)*torch::rand({5,5})).to(torch::kInt);
	std::cout << "t:\n" << t << std::endl;

	//第0行
	print(t[0]);

	//倒数第一行
	print(t[-1]);

	using torch::indexing::Slice;
	using torch::indexing::None;
	using torch::indexing::Ellipsis;

	// 第1行第3列
	std::cout << "t[1,3]:\n" << t[1,3] << std::endl;
	std::cout << "t[1][3]:\n" << t[1][3] << std::endl;;

	// 第1行至第3行
	std::cout << "t[1:4,:]:\n" << t.index({Slice(1, 4), Slice()}) << std::endl;

	// 第1行至最后一行，第0列到最后一列每隔两列取一列
	std::cout << "t[1:4,:4:2]:\n" << t.index({Slice(1, 4), Slice({0, 4, 2})}) << std::endl;

	// 可以使用索引和切片修改部分元素
	auto x = torch::tensor({{1,2}, {3,4}}).to(torch::kFloat32); //.requires_grad_();
	//x.data[1,:] = torch.tensor({0.0,0.0});
	//x.index_put_({ "...", Slice(1, 2)}, torch::tensor({0.0,0.0}).index({ "...", 0}));
	auto xx = torch::masked_fill(x, x > 2, 0.0);
	std::cout << "xx:\n" << xx << std::endl;
	std::cout << "x:\n" << x.masked_fill(x > 2, 0.0) << std::endl;

	auto tOne = torch::ones({1, 10, 10, 10});
	std::cout << "tOne.sizes(): " << tOne.sizes() << std::endl;
	auto tZero = torch::zeros({1, 10, 10, 10});
	std::cout << "tZero.sizes(): " << tZero.sizes() << std::endl;
	std::cout << "tZero:\n" << tZero << std::endl;
	tZero.index_put_({ "...", Slice(0, 5), Slice(0, 5), Slice(0, 5) },  tOne.index({ "...", 5, 5, 5 }));
	std::cout << "tZero:\n" << tZero << std::endl;

	a = torch::arange(27).view({3,3,3});
	std::cout << "a:\n" << a << std::endl;

	// 省略号可以表示多个冒号
	//print(a[...,1]);
	std::cout << "a[...,1]:\n" << a.index({"...", 1}) << std::endl;

	/*
	 * 以上切片方式相对规则，对于不规则的切片提取,可以使用torch.index_select, torch.take, torch.gather, torch.masked_select.
	 * 考虑班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4×10×7的张量来表示。
	 */
	minval=0;
	maxval=100;
	auto scores = torch::floor(minval + (maxval-minval)*torch::rand({4,10,7})).to(torch::kInt);
	std::cout << "scores:\n" << scores << std::endl;

	// 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
	std::cout << "抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩:\n";
	std::cout << torch::index_select(scores, /*dim =*/ 1, /*index =*/ torch::tensor({0,5,9})) << std::endl;

	// 抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
	auto q = torch::index_select(torch::index_select(scores, /*dim =*/ 1, /*index =*/ torch::tensor({0,5,9}))
	                   , /*dim=*/ 2, /*index =*/ torch::tensor({1,3,6}));

	std::cout << "抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩:\n" << q << std::endl;

	// 抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
	// take将输入看成一维数组，输出和index同形状
	auto s = torch::take(scores, torch::tensor({0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6}));
	std::cout << "抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩:\n" << s << std::endl;

	// 抽取分数大于等于80分的分数（布尔索引）
	auto g = torch::masked_select(scores,scores>=80);
	std::cout << "抽取分数大于等于80分的分数:\n" << g << std::endl;

	/*************************************************************
	 * 以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。
	 * 如果要通过修改张量的部分元素值得到新的张量，可以使用torch.where,torch.index_fill 和 torch.masked_fill
	 * torch.where可以理解为if的张量版本。
	 * torch.index_fill的选取元素逻辑和torch.index_select相同。
	 * torch.masked_fill的选取元素逻辑和torch.masked_select相同。
	 */
	// 如果分数大于60分，赋值成1，否则赋值成0
	auto ifpass = torch::where(scores>60,torch::tensor(1),torch::tensor(0));
	std::cout << "如果分数大于60分，赋值成1，否则赋值成0:\n" << ifpass << std::endl;

	// 将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分
	torch::index_fill(scores, /*dim =*/ 1, /*index =*/ torch::tensor({0,5,9}), /*value =*/ 100);
	// 等价于 scores.index_fill(dim = 1,index = torch.tensor([0,5,9]),value = 100)
	std::cout << "将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分:\n" << scores << std::endl;

	// 将分数小于60分的分数赋值成60分
	b = torch::masked_fill(scores, scores<60, 60);
	// 等价于b = scores.masked_fill(scores<60,60)
	std::cout << "将分数小于60分的分数赋值成60分:\n" << b << std::endl;

	/************************************************************
	 * 三，维度变换
	 *
	 * 维度变换相关函数主要有 torch.reshape(或者调用张量的view方法), torch.squeeze, torch.unsqueeze, torch.transpose
	 * torch.reshape 可以改变张量的形状。
	 * torch.squeeze 可以减少维度。
	 * torch.unsqueeze 可以增加维度。
	 * torch.transpose 可以交换维度。
	 */
	minval =0;
	maxval = 255;
	a = (minval + (maxval-minval)*torch::rand({1,3,3,2})).to(torch::kInt);
	std::cout << "a.shape:\n" << a.sizes() << std::endl;

	// 改成 （3,6）形状的张量
	std::cout << "a.view([3,6]):\n" << a.view({3,6}) << std::endl;
	std::cout << "torch.reshape(a,[3,6]):\n" << torch::reshape(a,{3,6}) << std::endl;

	// 改回成 [1,3,3,2] 形状的张量
	std::cout << "torch.reshape(a,[1,3,3,2]):\n" << torch::reshape(a,{1,3,3,2}) << std::endl;
	std::cout << "a.view([1,3,3,2]):\n" << a.view({1,3,3,2}) << std::endl;

	/*
	 * 如果张量在某个维度上只有一个元素，利用torch.squeeze可以消除这个维度。torch.unsqueeze的作用和torch.squeeze的作用相反。
	 */
	a = torch::tensor({{1.0,2.0}});
	s = torch::squeeze(a);
	std::cout << "a:\n" << a << std::endl;
	std::cout << "torch::squeeze(a):\n" << s << std::endl;
	std::cout << "a.shape:\n" << a.sizes() << std::endl;
	std::cout << "s.shape:\n" << s.sizes() << std::endl;

	// 在第0维插入长度为1的一个维度
	d = torch::unsqueeze(s, /*axis=*/ 0);
	std::cout << "d:\n" << d << std::endl;
	std::cout << "d.shape:\n" << d.sizes() << std::endl;

	/*
	 * torch.transpose可以交换张量的维度，torch.transpose常用于图片存储格式的变换上。
	 * 如果是二维的矩阵，通常会调用矩阵的转置方法 matrix.t()，等价于 torch.transpose(matrix,0,1)。
	 */
	minval=0;
	maxval=255;
	// Batch,Height,Width,Channel
	auto data = torch::floor(minval + (maxval-minval)*torch::rand({100,256,256,4})).to(torch::kInt);
	std::cout << "data:\n" << data.sizes() << std::endl;

	// 转换成 Pytorch默认的图片格式 Batch,Channel,Height,Width
	auto data_t = torch::transpose(torch::transpose(data,1,2),1,3);
	std::cout << "data_t:\n" << data_t.sizes() << std::endl;

	auto matrix = torch::tensor({{1,2,3},{4,5,6}});
	std::cout << "matrix:\n" << matrix << std::endl;
	std::cout << "matrix.shape:\n" << matrix.sizes() << std::endl;
	print(matrix.t()); //等价于torch.transpose(matrix,0,1)

	/****************************************************************
	 * 四，合并分割
	 *
	 * 可以用torch.cat方法和torch.stack方法将多个张量合并，可以用torch.split方法把一个张量分割成多个张量。
	 * torch.cat和torch.stack有略微的区别，torch.cat是连接，不会增加维度，而torch.stack是堆叠，会增加维度。
	 */
	a = torch::tensor({{1.0,2.0},{3.0,4.0}});
	b = torch::tensor({{5.0,6.0},{7.0,8.0}});
	c = torch::tensor({{9.0,10.0},{11.0,12.0}});

	auto abc_cat = torch::cat({a,b,c}, /*dim =*/ 0);
	std::cout << "abc_cat:\n" << abc_cat << std::endl;
	std::cout << "abc_cat.shape:\n" << abc_cat.sizes() << std::endl;

	auto abc_stack = torch::stack({a,b,c}, /*axis =*/ 0); //torch中dim和axis参数名可以混用
	std::cout << "abc_stack:\n" << abc_stack << std::endl;
	std::cout << "abc_stack.shape:\n" << abc_stack.sizes() << std::endl;

	/*
	 * torch.split是torch.cat的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。
	 */
	std::vector<torch::Tensor> spdt = torch::split(abc_cat, 2, 0); // 每份2个进行分割
	std::cout << "spdt[0]:\n" << spdt[0] << std::endl;
	std::cout << "spdt[1]:\n" << spdt[1] << std::endl;
	std::cout << "spdt[2]:\n" << spdt[2] << std::endl;
	std::cout << "spdt.size():\n" << spdt.size() << std::endl;

	spdt = abc_cat.split_with_sizes({4,1,1}, /*dim =*/ 0); // 每份分别为[4,1,1])
	std::cout << "spdt[0]:\n" << spdt[0] << std::endl;
	std::cout << "spdt[1]:\n" << spdt[1] << std::endl;
	std::cout << "spdt[2]:\n" << spdt[2] << std::endl;

	std::cout << "Done!\n";
	return 0;
}






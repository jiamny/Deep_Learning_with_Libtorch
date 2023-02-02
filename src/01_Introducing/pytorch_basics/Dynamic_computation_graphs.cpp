#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <map>

class MulConstant : public torch::autograd::Function<MulConstant> {
 public:
  static torch::autograd::Variable forward(torch::autograd::AutogradContext *ctx, torch::autograd::Variable variable, double constant) {
    ctx->saved_data["constant"] = constant;
    return variable * constant;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
    return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::autograd::Variable()};
  }
};


class LinearFunction : public torch::autograd::Function<LinearFunction> {
 public:
  // Note that both forward and backward are static functions

  // bias is an optional argument
  static torch::Tensor forward(
		  torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};

int main() {

	/************************************************************
	 * 一，动态计算图简介
	 *
	 * Pytorch的计算图由节点和边组成，节点表示张量或者Function，边表示张量和Function之间的依赖关系。
	 * Pytorch中的计算图是动态图。这里的动态主要有两重含义。
	 * 第一层含义是：计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。
	 * 第二层含义是：计算图在反向传播后立即销毁。下次调用需要重新构建计算图。如果在程序中使用了backward方法执行了反向传播，
	 * 或者利用torch.autograd.grad方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。
	 */
	// 1，计算图的正向传播是立即执行的。
	auto w = torch::tensor({{3.0,1.0}}, torch::requires_grad(true));
	auto b = torch::tensor({{3.0}}, torch::requires_grad(true));
	auto X = torch::randn({10,2});
	auto Y = torch::randn({10,1});
	//auto Y_hat = X @ w.t() + b;		// Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
	auto Y_hat = torch::mm(X, w.t()) + b;
	auto loss = torch::mean(torch::pow(Y_hat-Y,2));

	std::cout << "loss.data:\n" << loss.data() << std::endl;
	std::cout << "Y_hat.data:\n" << Y_hat.data() << std::endl;

	// 2，计算图在反向传播后立即销毁。
	w = torch::tensor({{3.0,1.0}}, torch::requires_grad(true));
	b = torch::tensor({{3.0}}, torch::requires_grad(true));
	X = torch::randn({10,2});
	Y = torch::randn({10,1});
//	Y_hat = X@w.t() + b;				// Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
	Y_hat = torch::mm(X, w.t()) + b;
	loss = torch::mean(torch::pow(Y_hat-Y,2));

	// 计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
	loss.backward();					// loss.backward(retain_graph = True)

	//loss.backward() #如果再次执行反向传播将报错

	/****************************************************
	 * 二，计算图中的Function
	 *
	 * 计算图中的 张量我们已经比较熟悉了, 计算图中的另外一种节点是Function, 实际上就是 Pytorch中各种对张量操作的函数。
	 * 这些Function和我们Python中的函数有一个较大的区别，那就是它同时包括正向计算逻辑和反向传播的逻辑。
	 * 我们可以通过继承torch.autograd.Function来创建这种支持反向传播的Function
	 */
	std::cout << "\n二，计算图中的Function:\n";
	auto x = torch::randn({2, 3}).requires_grad_();
	auto weight = torch::randn({4, 3}).requires_grad_();
	auto y = LinearFunction::apply(x, weight);
	//x.retain_grad();
	//w.retain_grad();
	y.sum().backward();

	std::cout << "x.grad():\n" << x.grad() << std::endl;
	std::cout << "weight.grad:\n" << w.grad() << std::endl;

	// y 的梯度函数即是 LinearFunction.backward
	std::cout << "y.grad_fn: " << y.grad_fn()->name() << std::endl;


	x = torch::randn({2}, torch::requires_grad(true));
	y = MulConstant::apply(x, 5.5);
	y.sum().backward();
	std::cout << x.grad() << std::endl;

	/******************************************************
	 * 三，计算图与反向传播
	 *
	 * loss.backward()语句调用后，依次发生以下计算过程。
	 * 1，loss自己的grad梯度赋值为1，即对自身的梯度为1。
	 * 2，loss根据其自身梯度以及关联的backward方法，计算出其对应的自变量即y1和y2的梯度，将该值赋值到y1.grad和y2.grad。
	 * 3，y2和y1根据其自身梯度以及关联的backward方法, 分别计算出其对应的自变量x的梯度，x.grad将其收到的多个梯度值累加。
	 * （注意，1,2,3步骤的求梯度顺序和对多个梯度值的累加规则恰好是求导链式法则的程序表述）
	 * 正因为求导链式法则衍生的梯度累加规则，张量的grad梯度不会自动清零，在需要的时候需要手动置零。
	 */
	std::cout << "\n三，计算图与反向传播:\n";

	x = torch::tensor({3.0}, torch::requires_grad(true));
	auto y1 = x + 1;
	auto y2 = 2*x;
	loss = torch::pow((y1-y2), 2);

	loss.backward();

	/*************************************************************************
	 * 四，叶子节点和非叶子节点
	 *
	 * 执行下面代码，我们会发现 loss.grad并不是我们期望的1,而是 None。类似地 y1.grad 以及 y2.grad也是 None.
	 * 这是为什么呢？这是由于它们不是叶子节点张量。在反向传播过程中，只有 is_leaf=True 的叶子节点，需要求导的张量的导数结果才会被最后保留下来。
	 * 那么什么是叶子节点张量呢？叶子节点张量需要满足两个条件。
	 * 1，叶子节点张量是由用户直接创建的张量，而非由某个Function通过计算得到的张量。
	 * 2，叶子节点张量的 requires_grad属性必须为True.
	 * Pytorch设计这样的规则主要是为了节约内存或者显存空间，因为几乎所有的时候，用户只会关心他自己直接创建的张量的梯度。
	 * 所有依赖于叶子节点张量的张量, 其requires_grad 属性必定是True的，但其梯度值只在计算过程中被用到，不会最终存储到grad属性中。
	 * 如果需要保留中间计算结果的梯度到grad属性中，可以使用 retain_grad方法。如果仅仅是为了调试代码查看梯度值，可以利用register_hook打印日志。
	 */
	std::cout << "\n四，叶子节点和非叶子节点:\n";

	x = torch::tensor({3.0}, torch::requires_grad(true));
	y1 = x + 1;
	y2 = 2*x;
	loss = torch::pow((y1-y2), 2);
	/*
	 * [W TensorBody.h:480] Warning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
	 * Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field
	 * to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf
	 * Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531
	 * for more informations. (function grad)
	 */
	x.retain_grad();
	y1.retain_grad();
	y2.retain_grad();
	loss.retain_grad();

	loss.backward();

	std::cout << "loss.grad:" << loss.grad() << std::endl;
	std::cout << "y1.grad:" << y1.grad() << std::endl;
	std::cout << "y2.grad:" << y2.grad() << std::endl;
	std::cout << x.grad() << std::endl;

	std::cout << "x.is_leaf:" << x.is_leaf() << std::endl;
	std::cout << "y1.is_leaf:" << y1.is_leaf() << std::endl;
	std::cout << "y2.is_leaf:" << y2.is_leaf() << std::endl;
	std::cout << "loss.is_leaf:" << loss.is_leaf() << std::endl;

	/****************************************************************************
	 * 利用retain_grad可以保留非叶子节点的梯度值，利用register_hook可以查看非叶子节点的梯度值。
	 */
	// 正向传播
	x = torch::tensor({3.0}, torch::requires_grad(true));
	y1 = x + 1;
	y2 = 2*x;
	loss = torch::pow((y1-y2), 2);

	// 非叶子节点梯度显示控制
	y1.register_hook([](torch::Tensor grad){ std::cout << "y1 grad: " << grad; });
	y2.register_hook([](torch::Tensor grad){ std::cout << "y2 grad: " << grad; }); 	//lambda grad: print('y2 grad: ', grad));
	loss.retain_grad();

	// 反向传播
	loss.backward();
	std::cout << "loss.grad:" << loss.grad() << std::endl;
	std::cout << "x.grad:" << x.grad() << std::endl;

	std::cout << "Done!\n";
	return 0;
}





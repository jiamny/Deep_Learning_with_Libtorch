
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <map>

using namespace torch::autograd;

int main() {

	/**************************************************************
	 * auto guess data type
	 */
	std::cout <<"---- auto guess data type:\n";

	auto i = torch::tensor({1});
	std::cout << i << " " << i.dtype() << std::endl;

	auto d = torch::tensor({2.0});
	std::cout << d << " " << d.dtype() << std::endl;

	auto j = torch::tensor({true});
	std::cout << j << " " << j.dtype() << std::endl;

	/**************************************************************
	 * define tensor data tyep
	 */
	std::cout <<"---- define tensor data tyep:\n";

	d = torch::tensor({1},torch::kInt32);
	std::cout << d << " " << d.dtype() << std::endl;
	j = torch::tensor({2.0}, torch::kDouble);
	std::cout << j << " " << j.dtype() << std::endl;

	i = torch::tensor({1});
	std::cout << i << " " << i.dtype() << std::endl;
	d = i.to(torch::kFloat32);
	std::cout << d << " " << d.dtype() << std::endl;
	j = i.type_as(d);
	std::cout << j << " " << j.dtype() << std::endl;

	/***************************************************************
	 * tensor dimension
	 */
	std::cout <<"---- tensor dimension:\n";

	auto scalar = torch::tensor(true);
	std::cout << "dim = " << scalar.dim() << std::endl;
	std::cout << "size = " << scalar.sizes() << std::endl;

	auto vector = torch::tensor({1.0,2.0,3.0,4.0});
	std::cout << "dim = " << vector.dim() << std::endl;
	std::cout << "size = " << vector.sizes() << std::endl;

	auto matrix = torch::tensor({{1.0,2.0},{3.0,4.0}});
	std::cout << "dim = " << matrix.dim() << std::endl;
	std::cout << "size = " << matrix.sizes() << std::endl;

	auto tensor3 = torch::tensor({{{1.0,2.0},{3.0,4.0}},{{5.0,6.0},{7.0,8.0}}});
	std::cout << "dim = " << tensor3.dim() << std::endl;
	std::cout << "size = " << tensor3.sizes() << std::endl;

	auto tensor4 = torch::tensor({{{{1.0,1.0},{2.0,2.0}},{{3.0,3.0},{4.0,4.0}}},
	                        {{{5.0,5.0},{6.0,6.0}},{{7.0,7.0},{8.0,8.0}}}});
	std::cout << "dim = " << tensor4.dim() << std::endl;
	std::cout << "size = " << tensor4.sizes() << std::endl;

	/***************************************************************
	 * change tensor view shape
	 */
	std::cout <<"---- change tensor view shape:\n";

	vector = torch::arange(0,12);
	std::cout << vector << std::endl;

	auto matrix34 = vector.view({3,4});
	std::cout << "dim = "  << matrix34.dim() << std::endl;
	std::cout << "size = " << matrix34.sizes() << std::endl;

	auto matrix43 = vector.view({4,-1});		// -1, 表示该位置长度由程序自动推断
	std::cout << "dim = "  << matrix43.dim() << std::endl;
	std::cout << "size = " << matrix43.sizes() << std::endl;

	/***************************************************************
	 * change tensor shape
	 */
	std::cout <<"---- change tensor shape:\n";

	auto matrix26 = torch::arange(0,12).view({2,6});
	std::cout << "dim = "  << matrix26.dim() << std::endl;
	std::cout << "size = " << matrix26.sizes() << std::endl;

	auto matrix62 = matrix26.t();
	std::cout << "is_contiguous = " << matrix62.is_contiguous() << std::endl;

	//matrix34 = matrix62.view(3,4) #error!
	matrix34 = matrix62.reshape({3, 4}); //等价于matrix34 = matrix62.contiguous().view(3,4)
	std::cout << "dim = "  << matrix34.dim() << std::endl;
	std::cout << "size = " << matrix34.sizes() << std::endl;

	auto tensor = torch::zeros(3);
	//  tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor.
	auto arr = tensor.t(); //numpy_T();
	std::cout << tensor << std::endl;
	std::cout << "tensor.t(): " << arr << std::endl;

	tensor.add_(1);
	std::cout << tensor << std::endl;

	scalar = torch::tensor(1.0);
	auto s = scalar.item<float>();
	std::cout << "s = "  << s << std::endl;
	std::cout << "type(s) = " << scalar.dtype() << std::endl;

	return 0;
}




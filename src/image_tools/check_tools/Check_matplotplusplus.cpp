#include <cmath>
#include <torch/torch.h>
#include <matplot/matplot.h>

using torch::indexing::Slice;
using torch::indexing::None;

using namespace matplot;

int main() {
/*
	std::vector<double> x = matplot::linspace(0, 2 * matplot::pi);
	std::vector<double> y = matplot::transform(x, [](auto x) { return sin(x); });

	matplot::plot(x, y, "-o");
	matplot::hold(true);
	matplot::plot(x, matplot::transform(y, [](auto y) { return -y; }), "--xr");
	matplot::plot(x, matplot::transform(x, [](auto x) { return x / matplot::pi - 1.; }), "-:gs");
	matplot::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");
	matplot::show();

*/

	auto h = figure(true);
	h->size(300, 1200);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(3, 1);
	h->position(0, 0);

    auto x = linspace(0, 10);

    auto y1 = transform(x, [](double x) { return sin(x); });
    auto y2 = transform(x, [](double x) { return cos(x); });

    auto ax1 = h->nexttile();
    matplot::plot(ax1, x, y1);
    //legend(ax1, "sin(x)");

    auto ax2 = h->nexttile();
    matplot::plot(ax2, x, y2);

    hold(ax2, true);
    auto y3 = transform(x, [](double x) { return sin(2 * x); });
    plot(ax2, x, y3);
    legend(ax2, "cos(x)", "sin(2x)");
    hold(ax2, false);

    auto ax3 = h->nexttile();
	std::vector<double> xx(10), yy(10);

	for(int i = 0; i < 10; i++) {
		xx[i] = i * 1.0;
		yy[i] = std::pow(0.8, i);
	}
	plot(ax3, xx, yy, "-o");
	ax3->axis(false);

    show();
	return 0;
}





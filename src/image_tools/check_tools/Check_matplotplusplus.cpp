#include <cmath>
#include <torch/torch.h>
#include <matplot/matplot.h>

using torch::indexing::Slice;
using torch::indexing::None;

using namespace matplot;

int main() {

	int slt = 2;

	switch(slt) {
		case 0: {
			std::vector<double> x = matplot::linspace(0, 2 * matplot::pi);
			std::vector<double> y = matplot::transform(x, [](auto x) { return sin(x); });

			matplot::plot(x, y, "-o");
			matplot::hold(true);
			matplot::plot(x, matplot::transform(y, [](auto y) { return -y; }), "--xr");
			matplot::plot(x, matplot::transform(x, [](auto x) { return x / matplot::pi - 1.; }), "-:gs");
			matplot::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");
			matplot::show();

			break;
		}
		case 1: {

			auto h = figure(true);
			h->size(600, 1200);
			h->add_axes(false);
			h->reactive_mode(false);
			h->tiledlayout(3, 1);
			h->position(0, 0);

		    auto x = linspace(0, 10);

		    auto y1 = transform(x, [](double x) { return sin(x); });
		    auto y2 = transform(x, [](double x) { return cos(x); });

		    auto ax1 = h->nexttile();
		    matplot::plot(ax1, x, y1);
		    legend(ax1, "sin(x)");

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
			break;
		}
		case 2: {
		    std::vector<double> train_loss, train_acc, test_acc;
		    std::vector<double> train_epochs;

		    for(int i = 0; i < 10; i++) {
		    	train_epochs.push_back(i *1.0);
		    	train_loss.push_back( 1.0/(i+1.0) );
		    	std::cout << train_loss[i] << "\n";
		    	train_acc.push_back((i - 0.1)/10.0);
		    	test_acc.push_back((i - 0.5)/10.0);
		    }

		    auto F = figure(true);
		    F->size(1200, 500);
		    F->add_axes(false);
		    F->reactive_mode(false);

		    auto ax1 = subplot(1, 2, 0);
		    ax1->xlabel("epoch");
		    ax1->ylabel("loss");
		    ax1->title("train loss");

		    //auto ax = F->nexttile();
		    plot( train_epochs, train_loss, "-o")->line_width(2)
							.display_name("train loss");
		    legend({});

		    auto ax2 = subplot(1, 2, 1);
		    plot( train_epochs, train_acc, "g--")->line_width(2)
		    				.display_name("train acc");
		    hold( on);
		    plot( train_epochs, test_acc, "r-.")->line_width(2)
		    				.display_name("test acc");
		    hold( on);
		    //auto l = ::matplot::legend(ax, {}); //->location(legend::general_alignment::right);
		    //l->location(legend::general_alignment::topleft);
		    legend( {}); //->location(legend::general_alignment::right);
		    ax2->xlabel("epoch");
		    ax2->ylabel("acc");
		    ax2->title("train & test acc");
		    hold( off);
		    F->draw();
		    show();

/*
		    auto x = linspace(0, pi);
		    auto y1 = transform(x, [](double x) { return cos(x); });
		    auto y2 = transform(x, [](double x) { return cos(2 * x); });
		    auto y3 = transform(x, [](double x) { return cos(3 * x); });
		    auto y4 = transform(x, [](double x) { return cos(4 * x); });

		    plot(x, y1);
		    hold(on);
		    plot(x, y2);
		    plot(x, y3);
		    plot(x, y4);
		    hold(off);

		    auto l = ::matplot::legend({"cos(x)", "cos(2x)", "cos(3x)", "cos(4x)"});
		    l->location(legend::general_alignment::topleft);
		    l->num_rows(2);
*/

			break;
		}
		default: {
			std::cout << "wrong case!\n";
			break;
		}
	}

	std::cout << "Done!\n";
	return 0;
}





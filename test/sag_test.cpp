#include "../src/core/loss/log_loss.hpp"
#include "../src/core/optimizer/sgd/sag.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);

    std::size_t num_features = X.cols();
    
    std::cout << "sag test" <<std::endl;
    loss::LogLoss<double> log_loss;

    VecType w(num_features);
    w.setZero();
    VecType opt_w(num_features);
    
    optimizer::SAG<double, 
        loss::LogLoss<double>, 
        optimizer::VanillaUpdate<double>, 
        optimizer::StepDecay<double>> sag;

    opt_w = sag.optimize(X, y, w);
    std::cout << opt_w <<std::endl;

}


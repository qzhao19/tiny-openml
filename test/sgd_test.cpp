#include "../src/core/loss/log_loss.hpp"
#include "../src/core/optimizer/sgd/sgd.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);


    std::size_t num_features = X.cols();
    
    std::cout << "sgd test" <<std::endl;
    loss::LogLoss<double> log_loss;

    VecType w(num_features);
    w = MatType::Ones(num_features, 1);
    VecType opt_w(num_features);
    // optimizer::VanillaUpdate<double> weight_update;
    // // optimizer::MomentumUpdate<double> weight_update(0.6);
    // optimizer::StepDecay<double> step_decay(0.1);

    optimizer::SGD<double, 
        loss::LogLoss<double>, 
        optimizer::VanillaUpdate<double>, 
        optimizer::StepDecay<double>> sgd;
    opt_w = sgd.optimize(X, y, w);
    
    std::cout << opt_w <<std::endl;

}


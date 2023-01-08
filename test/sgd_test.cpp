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
    
    VecType w(num_features);
    w.setOnes();
    VecType opt_w(num_features);

    loss::LogLoss<double> log_loss;
    optimizer::VanillaUpdate<double> weight_update;
    // optimizer::MomentumUpdate<double> weight_update(0.6);
    optimizer::StepDecay<double> step_decay(0.1);

    optimizer::SGD<double, 
        loss::LogLoss<double>, 
        optimizer::VanillaUpdate<double>, 
        optimizer::StepDecay<double>> sgd(
            w, 
            log_loss, 
            weight_update, 
            step_decay, 
            2000, 16, 5, 0.0001, true, true);
    sgd.optimize(X, y);
    opt_w = sgd.get_coef();
    
    
    std::cout << opt_w <<std::endl;

}


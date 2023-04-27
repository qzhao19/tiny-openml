#include "../src/core/loss/log_loss.hpp"
#include "../src/core/optimizer/sgd/sgd.hpp"

using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    std::size_t num_features = X.cols();
    std::cout << "sgd test" <<std::endl;
    
    MatType w(num_features, 3);
    w.setOnes();
    MatType opt_w(num_features, 3);

    // loss::LogLoss<double> log_loss;
    loss::SoftmaxLoss<double> softmax_loss(0.0);
    optimizer::VanillaUpdate<double> weight_update;
    // optimizer::MomentumUpdate<double> weight_update(0.6);
    optimizer::StepDecay<double> step_decay(0.01);

    optimizer::SGD<double, 
        loss::SoftmaxLoss<double>, 
        optimizer::VanillaUpdate<double>, 
        optimizer::StepDecay<double>> sgd(
            w, 
            softmax_loss, 
            weight_update, 
            step_decay, 
            2000, 16, 5, 0.0001, true, true, true);
    sgd.optimize(X, y);
    opt_w = sgd.get_coef();
    
    
    std::cout << opt_w <<std::endl;

}


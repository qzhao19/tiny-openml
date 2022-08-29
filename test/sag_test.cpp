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

    VecType W(num_features);
    W.setOnes();
    
    // std::cout << W <<std::endl;
    VecType opt_W(num_features);
    optimizer::VanillaUpdate<double> weight_update;
    // optimizer::MomentumUpdate<double> weight_update(0.6);
    optimizer::StepDecay<double> step_decay(0.15);

    optimizer::SAG<double> sag(X, y);
    opt_W = sag.optimize(W, log_loss, weight_update, step_decay);
    
    std::cout << opt_W <<std::endl;

}


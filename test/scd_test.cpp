#include "../src/core/loss/mean_squared_error.hpp"
#include "../src/core/optimizer/sgd/scd.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/boston_house_price.txt", X, y);

    std::size_t num_features = X.cols();
    
    std::cout << "scd test" <<std::endl;
    loss::MSE<double> mse_loss;

    VecType W(num_features);
    W.setOnes();
    
    // std::cout << W <<std::endl;
    VecType opt_W(num_features);
    optimizer::VanillaUpdate<double> weight_update;
    // optimizer::MomentumUpdate<double> weight_update(0.6);
    optimizer::StepDecay<double> step_decay(0.15);

    optimizer::SCD<double> scd(X, y);
    opt_W = scd.optimize(W, mse_loss);
    
    std::cout << opt_W <<std::endl;

}


#include "../src/core/loss/mean_squared_error.hpp"
#include "../src/core/optimizer/sgd/scd.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;
    data::loadtxt<MatType, VecType>("../dataset/diabetes.txt", X, y);


    // MatType X{{0, 0, 0, 0.2, 0.2}, 
    //           {0, 0, 0.3, 0, -0.5},
    //           {0.4, 0.6, 0, 0, 0}};
    // VecType y{{1, 1, -1}};


    std::size_t num_features = X.cols();
    
    std::cout << "scd test" <<std::endl;
    loss::MSE<double> mse_loss;

    VecType W(num_features);
    W.setZero();
    
    // std::cout << W <<std::endl;
    VecType opt_W(num_features);
    optimizer::StepDecay<double> step_decay(0.15);

    optimizer::SCD<double> scd(X, y);
    opt_W = scd.optimize(W, mse_loss);
    
    std::cout << opt_W <<std::endl;

}


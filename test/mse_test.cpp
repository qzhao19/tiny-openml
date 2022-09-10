#include "../src/core/loss/mean_squared_error.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    
    MatType X{{0, 0, 0, 0.2, 0.2}, 
              {0, 0, 0.3, 0, -0.5},
              {0.4, 0.6, 0, 0, 0}};
    VecType y{{1, 1, -1}};

    std::size_t num_samples = X.rows(), num_features = X.cols();

    loss::MSE<double> mse;
    VecType W(num_features);
    W.setZero();
    VecType grad(num_features);
    
    double loss_val = 0.0;
    loss_val = mse.evaluate(X, y, W);
    std::cout << "loss_val" << std::endl;
    std::cout << loss_val << std::endl;

    grad = mse.gradient(X, y, W);
    std::cout << "grad" << std::endl;
    std::cout << grad << std::endl;

}

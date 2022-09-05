#include "../src/core/loss/mean_squared_error.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/boston_house_price.txt", X, y);

    MatType X_train;
    MatType X_test;
    VecType y_train;
    VecType y_test;
    std::tie(X_train, X_test, y_train, y_test) = data::train_test_split<MatType, VecType>(X, y, 0.9);

    std::size_t num_samples = X.rows(), num_features = X.cols();

    loss::MSE<double> mse;
    VecType W(num_features);
    W = Eigen::MatrixXd::Ones(num_features, 1);
    VecType grad(num_features);
    
    double loss_val = 0.0;
    loss_val = mse.evaluate(X, y, W);
    std::cout << "loss_val" << std::endl;
    std::cout << loss_val << std::endl;

    grad = mse.gradient(X, y, W);
    std::cout << "grad" << std::endl;
    std::cout << grad << std::endl;

}

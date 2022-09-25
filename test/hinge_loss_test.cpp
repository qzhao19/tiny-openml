#include "../src/core/loss/hinge_loss.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);

    std::size_t num_samples = X.rows(), num_features = X.cols();
    std::size_t batch_size = 32;
    std::size_t num_iter = num_samples / batch_size;

    loss::HingeLoss<double> hinge_loss;

    VecType W(num_features);
    W = Eigen::MatrixXd::Ones(num_features, 1);
    VecType grad(num_features);
    
    double loss_val = 0.0;
    loss_val = hinge_loss.evaluate(X, y, W);
    std::cout << "loss_val" << std::endl;
    std::cout << loss_val << std::endl;

    grad = hinge_loss.gradient(X, y, W);
    std::cout << "grad" << std::endl;
    std::cout << grad << std::endl;
    // W = W - 0.1 * grad;
    // std::cout << W << std::endl;


}
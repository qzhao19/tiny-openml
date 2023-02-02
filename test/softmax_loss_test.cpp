#include "../src/core/loss/softmax_loss.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    std::size_t num_samples = X.rows(), num_features = X.cols();

    loss::SoftmaxLoss<double> softmax_loss;

    MatType W(num_features, 3);
    W.setOnes();

    MatType grad;
    double loss_val = 0.0;
    loss_val = softmax_loss.evaluate(X, y, W);
    std::cout << "loss_val" << std::endl;
    std::cout << loss_val << std::endl;

    grad = softmax_loss.gradient(X, y, W);
    std::cout << "grad" << std::endl;
    std::cout << grad << std::endl;


}
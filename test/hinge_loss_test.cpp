#include "../src/core/loss/hinge_loss.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X{{-2,4,-1}, {4,1,-1}, {1, 6, -1}, {2, 4, -1}, {6, 2, -1}};
    VecType y{{-1,-1,1,1,1}};

    // MatType X;
    // VecType y;

    // data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);
    // for (int i = 0; i < y.rows(); ++i) {
    //     if (y(i, 0) == 0.0) {
    //         y(i, 0) = -1.0;
    //     }
    // }

    std::size_t num_samples = X.rows(), num_features = X.cols();

    loss::HingeLoss<double> hinge_loss;

    VecType W(num_features);
    W.setOnes();
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
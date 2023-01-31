#include "../src/core/loss/softmax_loss.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/iris.txt", X, y);

    std::size_t num_samples = X.rows(), num_features = X.cols();
    

}
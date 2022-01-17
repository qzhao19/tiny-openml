#include "../src/core/loss/log_loss.hpp"
#include "../src/core/optimizer/sgd/sgd.hpp"
using namespace openml;

int main() {

    using MatType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<double, Eigen::Dynamic, 1>;

    MatType X;
    VecType y;

    data::loadtxt<MatType, VecType>("../dataset/ionosphere.txt", X, y);

    std::size_t num_features = X.cols();
    
    std::cout << "sgd test" <<std::endl;
    loss::LogLoss<double> log_loss;

    VecType W(num_features);
    W = Eigen::MatrixXd::Random(num_features, 1);
    
    optimizer::SGD<double> sgd(X, y);
    sgd.optimize(log_loss, W);
    

}


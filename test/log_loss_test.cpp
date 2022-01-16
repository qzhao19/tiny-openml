#include "../src/core/loss/log_loss.hpp"
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

    loss::LogLoss<double> log_loss;

    VecType W(num_features);
    W = Eigen::MatrixXd::Random(num_features, 1);
    VecType grad(num_features);
    
    MatType X_batch(batch_size, num_features);
    VecType y_batch(batch_size);
    for (std::size_t j = 0; j < num_iter; j++) {
        std::size_t begin = j * batch_size;
        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);

        double loss_val = 0.0;
        loss_val = log_loss.evaluate(X_batch, y_batch, W);
        std::cout << "loss_val" << std::endl;
        std::cout << loss_val << std::endl;

        grad = log_loss.gradient(X_batch, y_batch, W);
        std::cout << "grad" << std::endl;
        std::cout << grad << std::endl;
    }

}

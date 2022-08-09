#ifndef CORE_LOSS_LOG_LOSS_HPP
#define CORE_LOSS_LOG_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class LogLoss {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    LogLoss() {};
    ~LogLoss() {};

    /**
     * evaluate the gradient of the logistic regression log-likelihood function
     * with the given parameters.
     *          L(w) = sum{y_i * Wt * X_i - log[1 + exp(Wt * X_i)]}
     * 
     * @param X ndarray of shape (num_samples, num_features), the matrix of input data
     * @param y ndarray of shape (num_samples) 
     * @param W ndarray of shape (num_features, 1) coefficient of the features 
    */
    double evaluate(const MatType& X, 
        const VecType& y, 
        const VecType& W) const {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        double loss = 0.0;

        // np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y_hat)
        for (std::size_t i = 0; i < num_samples; ++i) {
            double x_w = static_cast<double>(X.row(i) * W);
            loss += (static_cast<double>(y(i, 0)) * x_w - std::log(1.0 + std::exp(x_w)));
        }
        loss = loss / (static_cast<double>(num_samples));
        return loss;
    };

    /**
     * Evaluate the gradient of logistic regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index
     * 
     *      grad = (1 / m) * sum(sigmoid(x) - y) + (lambda / m) * W
    */
    VecType gradient(const MatType& X, 
        const VecType& y,
        const VecType& W) const{
        
        std::size_t num_samples = X.rows();
        std::size_t num_features = X.cols();

        VecType X_w(num_samples); 
        VecType y_hat(num_samples); 
        X_w = X * W;
        y_hat = math::sigmoid<VecType>(X_w);
        
        VecType grad(num_features);
        // grad = (X.transpose() * (y_hat - y) + lambda * W) / num_samples;
        grad = (X.transpose() * (y_hat - y)).array() / (static_cast<DataType>(num_samples));
        return grad;
    };
};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

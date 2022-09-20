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
     * evaluate the logistic regression log-likelihood function
     * with the given parameters.
     *       sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y_hat)
     * 
     * @param X ndarray of shape (num_samples, num_features), the matrix of input data
     * @param y ndarray of shape (num_samples) 
     * @param W ndarray of shape (num_features, 1) coefficient of the features 
    */
    const double evaluate(const MatType& X, 
        const VecType& y, 
        const VecType& W) const {
        
        std::size_t num_samples = X.rows();

        VecType X_w(num_samples); 
        VecType y_hat(num_samples); 
        X_w = X * W;
        y_hat = math::sigmoid<VecType>(X_w);

        VecType tmp1 = static_cast<DataType>(-1) * y.array() * 
            y_hat.array().log();
        VecType tmp2 = (static_cast<DataType>(1) - y.array()) * 
            (static_cast<DataType>(1) - y_hat.array()).log();
        VecType tmp3 = tmp1 - tmp2;
        double loss = static_cast<double>(tmp3.array().sum()) /
            static_cast<double>(num_samples);

        return loss;
    }

    const double evaluate(const MatType& X, 
        const VecType& y, 
        const VecType& W, 
        const double lambda) const {
        
        std::size_t num_samples = X.rows();

        double loss = evaluate(X, y, W);
        
        double reg = static_cast<double>(W.transpose() * W) / 
            (static_cast<double>(num_samples) * 2.0);

        return loss + reg * lambda;
    }

    /**
     * Evaluate the gradient of logistic regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index
     * 
     *      dot(X.T, sigmoid(np.dot(X, W)) - y) / len(y)
    */
    const VecType gradient(const MatType& X, 
        const VecType& y,
        const VecType& W) const{
        
        std::size_t num_samples = X.rows();
        std::size_t num_features = X.cols();

        VecType X_w(num_samples); 
        VecType y_hat(num_samples); 
        X_w = X * W;
        y_hat = math::sigmoid<VecType>(X_w);
        
        VecType grad(num_features);
        grad = (X.transpose() * (y_hat - y)) / static_cast<DataType>(num_samples);
        return grad;
    };

    const VecType gradient(const MatType& X, 
        const VecType& y,
        const VecType& W, 
        const double lambda) const {
        
        std::size_t num_samples = X.rows();
        std::size_t num_features = X.cols();

        VecType grad(num_features);
        grad = gradient(X, y, W);

        VecType reg(num_features);
        reg = W.array() / static_cast<DataType>(num_samples);

        return grad + reg * lambda;

    }
};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

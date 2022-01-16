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

    double lambda;
    std::string penalty;
    
public:
    /**
     * create the log loss constructor
     * 
     * @param lambda double, regularization strength; must be a positive float. 
     * @param penalty string, specify the norm of the penalty {‘l2’, ‘none’}, default='None'
     * 
    */
    LogLoss(const DataType lambda_ = static_cast<DataType>(0), 
        const std::string penalty_ = "None"): lambda(lambda_), 
            penalty(penalty_) {
                // std::size_t num_samples = X_.rows();
                // std::size_t num_features = X_.cols();
            };

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
        
        std::size_t num_samples = X.rows();
        std::size_t num_features = X.cols();

        DataType error;
        DataType penalty_val;
        for (std::size_t i = 0; i < num_samples; i++) {
            double x_w = X.row(i) * W;
            error += ((y(i, 0) * x_w) - 
                    std::log(1.0 + std::exp(x_w)));
        }
        // regularization term
        if (penalty == "None") {
            penalty_val = static_cast<DataType>(0);
        }
        else if (penalty == "l2") {
            penalty_val = W.transpose() * W;
            // penalty_val = penalty_val * lambda;
        }
        penalty_val = (penalty_val * lambda) / (static_cast<double>(num_samples));
        error = error / (-static_cast<double>(num_samples));
        return error + penalty_val;
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
        if (penalty == "None") {
            grad = (X.transpose() * (y_hat - y) + lambda * W) / num_samples;
        }
        else if (penalty == "l2") {
            grad = (X.transpose() * (y_hat - y)) / num_samples + 
                static_cast<DataType>(2) * lambda * W;
        }

        return grad;
    };
};


}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

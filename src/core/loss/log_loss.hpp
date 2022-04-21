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

    double lambda_;
    std::string penalty_;
    
public:
    /**
     * create the log loss constructor
     * 
     * @param lambda double, regularization strength; must be a positive float. 
     * @param penalty string, specify the norm of the penalty {‘l2’, ‘none’}, default='None'
     * 
    */
    LogLoss(const double lambda = 0.0, 
        const std::string penalty = "None"): lambda_(lambda), 
            penalty_(penalty) {};

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

        double error;
        double penalty_val;

        // np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y_hat)
        for (std::size_t i = 0; i < num_samples; i++) {
            double x_w = static_cast<double>(X.row(i) * W);
            error += (std::log(1.0 + std::exp(x_w)) - (y(i, 0) * x_w));
        }

        // regularization term
        if (penalty_ == "None") {
            penalty_val = 0.0;
        }
        else if (penalty_ == "l2") {
            penalty_val = static_cast<double>(W.transpose() * W);
            // penalty_val = penalty_val * lambda;
        }
        penalty_val = (penalty_val * lambda_) / (static_cast<double>(num_samples));
        error = error / (static_cast<double>(num_samples));
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
        if (penalty_ == "None") {
            grad = (X.transpose() * (y_hat - y)) / (static_cast<double>(num_samples));
        }
        else if (penalty_ == "l2") {
            grad = (X.transpose() * (y_hat - y)) / (static_cast<double>(num_samples)) + 
                2 * lambda_ * W;
        }

        return grad;
    };
};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

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

    MatType X;
    VecType y;
    double lambda;
    std::string penalty;
    
    std::size_t num_samples;
    std::size_t num_features;
    
public:
    /**
     * create the log loss constructor
     * 
     * @param X ndarray of shape (num_samples, num_features), input data matrix
     * @param y ndarray of shape (num_samples,) target vector relative to X.
     * @param lambda double, regularization strength; must be a positive float. 
     * @param penalty string, specify the norm of the penalty {‘l2’, ‘none’}, default='None'
     * 
    */
    LogLoss(const MatType& X_, 
        const VecType& y_, 
        const DataType lambda_ = static_cast<DataType>(0), 
        const std::string penalty_ = "None"): X(X_), y(y_), 
            lambda(lambda_), 
            penalty(penalty_) {
                std::size_t num_samples = X.rows();
                std::size_t num_features = X.cols();
            };

    ~LogLoss() {};

    void shuffle() {
        math::shuffle_data(X, y, X, y);
    };

    /**
     * calculate the log loos function value
     * L(w) = sum{y_i * Wt * X_i - log[1 + exp(Wt * X_i)]}
     * 
     * @param W ndarray of shape (num_features, 1) coefficient of the features 
     * @param begin int, the begin index of one batch data
     * @param batch_size int desired batch size for mini batches.
    */
    double evaluate(const VecType& W, 
        const std::size_t begin,
        const std::size_t batch_size) const {
        
        // get one batch data 
        MatType X_batch(batch_size, num_features);
        VecType y_batch(batch_size);
        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);

        DataType error;
        DataType penalty_val;
        for (std::size_t i = 0; i < batch_size; i++) {
            double x_w = X_batch.row(i) * W;
            error += ((y_batch(i, 0) * x_w) - 
                    std::log(1.0 + std::exp(x_w)));
        }
        // regularization term
        if (penalty == "None") {
            penalty_val = static_cast<DataType>(0);
        }
        else if (penalty == "l2") {
            penalty_val = (W.transpose() * W);
            // penalty_val = penalty_val * lambda;
        }
        penalty_val = penalty_val / (static_cast<double>(batch_size));
        error = error / (-static_cast<double>(batch_size));
        return error + penalty_val;
    }

    void gradient(const VecType& W, 
        VecType& grad,
        const std::size_t begin,
        const std::size_t batch_size) {
        
        MatType X_batch(batch_size, num_features);
        VecType y_batch(batch_size);
        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);

        VecType X_w(batch_size); 
        VecType y_hat(batch_size); 
        X_w = X_batch * W;
        y_hat = math::sigmoid<VecType>(X_w);

        if (penalty == "None") {
            grad = (X_batch.transpose() * (y_hat - y_batch) + lambda * W) / batch_size;
        }
        else if (penalty == "l2") {
            grad = (X_batch.transpose() * (y_hat - y_batch)) / batch_size; + 
                static_cast<DataType>(2) * lambda * W;
        }
    };

    const std::size_t get_num_samples() {
        return num_samples;
    };

    const std::size_t get_num_features() {
        return num_features;
    };
};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

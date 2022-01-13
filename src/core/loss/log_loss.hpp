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
    double reg_coef;
    std::size_t num_samples;
    std::size_t num_features;
    std::string penalty;

public:
    /**
     * create the log loss function constructor
     * @param X ndarray of shape (num_samples, num_features), the matrix of input data
     * @param y ndarray of shape (num_samples) 
     * @param penalty string, {‘l2’, ‘none’}, default "l2"
    */
    LogLoss(const MatType& X_, 
        const VecType& y_, 
        const double reg_coef_ = 1.0, 
        const std::string penalty_ = "l2"): X(X_), y(y_), 
            reg_coef(reg_coef_), 
            penalty(penalty_){
                num_samples = X.rows();
                num_features = X.cols();
            };

    ~LogLoss() {};

    /**
     * shuffle the dataset and associated label, when we apply the SGD method.
     * This may be called by the optimizer.
    */
    void shuffle() {
        math::shuffle_data(X, y, X, y);
    };

    /**
     * Evaluate the gradient of logistic regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index. 
     * 
     * @param weight Vector of logistic regression parameters 
     * @param begin Index of the starting point for loss function
     * @param batch_size Number of sample that will be passed 
    */
    double evaluate(const VecType& weight, 
        const std::size_t begin,
        const std::size_t batch_size) const {
        
        MatType X_batch(batch_size, num_features);
        VecType y_batch(batch_size);

        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);

        // L(w) = Sum(y_i * W_t * X_i - log(1 + exp(W_t * X_i)))
        DataType retval;
        // double retval = 0.0;
        for (std::size_t i = 0; i < batch_size; i++) {
            DataType val;
            val = X_batch.row(i) * weight;
            retval += (std::log(1.0 + std::exp(val)) - y_batch(i, 0) * val);
        }

        // l2 penelty: 1/2 * W_T * W
        DataType penalty_val;
        if (penalty == "None") {
            penalty_val = static_cast<DataType>(0);
        }
        else if (penalty == "l2") {
            penalty_val = weight.transpose() * weight;
            penalty_val = penalty_val / static_cast<DataType>(2);
        }
        return reg_coef * retval + penalty_val;

    };

    /**
     * Calculate the gradient of logistic regression function with the given parameters
     * for the given batch size from a given index in the dataset
     * 
     * gradient = (1 / m) * sum(sigmoid(x) - y) + (reg_coef / m) * W
    */
    void gradient(const VecType& weight,
        VecType& grad, 
        const std::size_t begin,
        const std::size_t batch_size) {
        
        MatType X_batch(batch_size, num_features);
        VecType y_batch(batch_size);

        // define dataset of one batch and vector of label associated
        X_batch = X.middleRows(begin, batch_size);
        y_batch = y.middleRows(begin, batch_size);

        VecType X_w(batch_size); 
        VecType h(batch_size);

        // apply sigmoid function to matrix 
        X_w = X_batch * weight;
        h = X_w.unaryExpr(std::ref(math::sigmoid));

        if (penalty == "None") {
            grad = X_batch.transpose() * (h - y_batch);
        }
        else if (penalty == "l2") {
            DataType C = static_cast<DataType>(reg_coef);
            grad = X_batch.transpose() * (h - y_batch) * C + weight;
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

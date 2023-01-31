#ifndef CORE_LOSS_SOFTMAX_LOSS_HPP
#define CORE_LOSS_SOFTMAX_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class SoftmaxLoss {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double lambda_;

public:
    SoftmaxLoss(): lambda_(0.0) {};
    SoftmaxLoss(double lambda): lambda_(lambda){};
    ~SoftmaxLoss() {};

    /**
     * evaluate the softmax loss function
     * with the given parameters.
     *       L(W) = -1/N * logP(Y|X, W)
     *            = 1/N {sum(X_i * W_k) + sum(log(sum(exp(-X_i * W_k))))}
     * 
     * @param X ndarray of shape (num_samples, num_features), the matrix of input data
     * @param y ndarray of shape (num_samples) 
     * @param W ndarray of shape (num_features, num_classes) coefficient of the features 
    */
    const double evaluate(const MatType& X, 
        const VecType& y, 
        const MatType& w) const {
        
        std::size_t num_samples = X.rows();
        std::size_t num_features = X.cols();
        std::size_t num_classes = w.cols();
        MatType xw(num_samples, num_classes);
        xw = X.dot(w);

        xw.noalias() -= xw.maxCoeff();
        MatType exp_xw = xw.array().exp();

        VecType sum_exp_xw = exp_xw.rowwise().sum();

        


    }

    /**
     * Evaluate the gradient of logistic regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index
     * 
     *      dot(X.T, sigmoid(np.dot(X, W)) - y) / len(y)
    */
    const VecType gradient(const MatType& X, 
        const VecType& y,
        const MatType& W) const{
        
    };

};

}
}
#endif /*CORE_LOSS_SOFTMAX_LOSS_HPP*/

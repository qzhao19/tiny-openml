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
        const MatType& W) const {
        
        std::size_t num_classes = W.cols();
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        MatType xw(num_samples, num_classes);
        xw = X * W;

        auto tmp1 = xw.array() - xw.maxCoeff();
        MatType exp_xw = tmp1.array().exp();
        VecType sum_exp_xw = exp_xw.rowwise().sum();

        std::size_t index;
        VecType exp_xw_i(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            index = static_cast<std::size_t>(y(i, 0));
            exp_xw_i(i, 0) = exp_xw(i, index);
        }

        auto tmp2 = exp_xw_i.array() / sum_exp_xw.array();
        double loss = -tmp2.array().log().sum();

        loss /= static_cast<double>(num_samples);

        double reg = (W.transpose() * W).array().sum() / 
            (static_cast<double>(num_samples) * 2.0);
        
        return loss + reg * lambda_;
    }

    /**
     * Evaluate the gradient of softmax regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index
     * 
     *      dot(X.T, sigmoid(np.dot(X, W)) - y) / len(y)
    */
    const MatType gradient(const MatType& X, 
        const VecType& y,
        const MatType& W) const{
        std::size_t num_classes = W.cols();
        std::size_t num_samples = X.rows(), num_features = X.cols();

        MatType grad(num_features, num_classes);
        MatType xw(num_samples, num_classes);
        xw = X * W;

        auto tmp1 = xw.array() - xw.maxCoeff();
        MatType exp_xw = tmp1.array().exp();
        VecType sum_exp_xw = exp_xw.rowwise().sum();

        // std::cout << "sum_exp_xw: " << sum_exp_xw << std::endl;

        MatType normalized_exp_xw = exp_xw.array().colwise() / sum_exp_xw.array();

        // std::cout << "normalized_exp_xw: " << normalized_exp_xw << std::endl;
        std::size_t index;
        for (int i = 0; i < num_samples; ++i) {
            index = static_cast<std::size_t>(y(i, 0));
            normalized_exp_xw(i, index) -= 1.0;
        }

        grad = X.transpose() * normalized_exp_xw;
        grad.array() /= static_cast<double>(num_samples);
    
        MatType reg(num_features, num_classes);
        reg = W.array() / static_cast<DataType>(num_samples);

        return grad + reg * lambda_;
    };

};

}
}
#endif /*CORE_LOSS_SOFTMAX_LOSS_HPP*/

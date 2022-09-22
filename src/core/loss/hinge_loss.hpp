#ifndef CORE_LOSS_LOG_LOSS_HPP
#define CORE_LOSS_LOG_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class HingeLoss {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double lambda_;
    double threshold_;

public:
    HingeLoss(): lambda_(0.0), threshold_(0.0) {};
    HingeLoss(double lambda, 
        double threshold): lambda_(lambda), threshold_(threshold){};
    ~HingeLoss() {};

    /**
     * evaluate the hinge loss function
     * with the given parameters.
     *       f(x) = max(0, 1 - y*g(x)), g(x) = X * W
     * 
     * @param X ndarray of shape (num_samples, num_features), the matrix of input data
     * @param y ndarray of shape (num_samples) 
     * @param W ndarray of shape (num_features, 1) coefficient of the features 
    */
    const double evaluate(const MatType& X, 
        const VecType& y, 
        const VecType& W) const {
        
        std::size_t num_samples = X.rows();

        VecType y_X_w(num_samples); 
        y_X_w = y.transpose() * X * W; 

        double loss;
        loss = std::max(0.0, threshold_ - static_cast<double>(y_X_w));

        double reg = static_cast<double>(W.transpose() * W) / 
            (static_cast<double>(num_samples) * 2.0);

        return loss + reg * lambda_;
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
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType grad(num_features);

        VecType y_X_w(num_samples); 
        y_X_w = y.transpose() * X * W; 

        for (std::size_t i = 0; i < num_samples; ++i) {
            if (y_X_w(i, 0) > threshold_) {
                grad(i, 0) = 0;
            }
            else {
                grad(i, 0) = X.row(i).transpose().array() * y(i, 0);
            }
        }
        return grad;
    };

};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

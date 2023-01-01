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
    HingeLoss(): lambda_(0.0), threshold_(1.0) {};
    HingeLoss(double lambda, 
        double threshold): lambda_(lambda), threshold_(threshold){};
    ~HingeLoss() {};

    /**
     * evaluate the hinge loss objective function with the given parameters.
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
        VecType X_w(num_samples);
        X_w = X * W; 

        double loss = 0.0;
        for (std::size_t i = 0; i < num_samples; ++i) {
            double tmp = std::max(0.0, threshold_ - X_w(i, 0) * y(i, 0));
            loss += tmp;
        }

        // loss = loss / static_cast<double>(num_samples);
        double reg = static_cast<double>(W.transpose() * W) / 
            (static_cast<double>(num_samples) * 2.0);

        return loss + reg * lambda_;
    }

    /**
     * Evaluate the gradient of the hinge loss objective function with the given 
     * parameters
     * 
     *      dw = y_i*w*x > 1 : 0 ? -y_i*x
    */
    const VecType gradient(const MatType& X, 
        const VecType& y,
        const VecType& W) const{
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType grad(num_features);

        VecType X_w(num_samples);
        X_w = X * W;

        for (std::size_t i = 0; i < num_samples; ++i) {
            // auto x_w = X.row(i) * W;
            double x_w_y = X_w(i, 0) * y(i, 0);

            if (x_w_y > threshold_) {
                VecType tmp(num_features);
                tmp.setZero();
                grad.noalias() += tmp;
            }
            else {
                grad.noalias() += (X.row(i).transpose() * (-y(i, 0)));
            }
        }

        VecType reg(num_features);
        reg = W.array() / static_cast<DataType>(num_samples);
        return grad + reg * lambda_;
    };

};

}
}
#endif /*CORE_LOSS_LOG_LOSS_HPP*/

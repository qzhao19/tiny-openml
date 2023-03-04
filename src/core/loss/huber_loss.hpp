#ifndef CORE_LOSS_HUBER_LOSS_HPP
#define CORE_LOSS_HUBER_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class HuberLoss {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double delta_;
    double lambda_;

public:
    HuberLoss(): lambda_(0.0), delta_(1.0) {};
    HuberLoss(double lambda, double delta): lambda_(lambda), delta_(delta){};
    ~HuberLoss() {};

    /**
     * evaluate the huber loss objective function with the given parameters.
     *       f(x) = 1/2 * (y - X_w) ** 2                  if |y - X_w| <= delta
     *              delta * |y - X_w| - 1/2 * delta ** 2  else
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
        double condition = (y - X_w).array().norm();
        if (condition <= delta_) {
            loss = 0.5 * condition;
        }
        else {
            auto tmp = (y - X_w).array() - 0.5 * delta_;
            auto norm = tmp.array().abs().colwise().sum().maxCoeff();
            loss = delta_ * static_cast<double>(norm);
        }

        // loss = loss / static_cast<double>(num_samples);
        double reg = static_cast<double>(W.transpose() * W) / 
            (static_cast<double>(num_samples) * 2.0);

        return loss + reg * lambda_;
    };

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

        double condition = (y - X_w).array().norm();
        if (condition <= delta_) {
            auto tmp = X.transpose() * (X_w - y);
            grad = tmp.array() / num_samples;
        }
        else {
            // grad = self.delta *np.dot(X.T, np.sign(np.dot(X,self.w) - y))/ y.shape[0]
            auto tmp = X.transpose() * (X_w - y).array().sign();
            grad = tmp.array() * (delta_ / num_samples);
        }
        

        VecType reg(num_features);
        reg = W.array() / static_cast<DataType>(num_samples);
        return grad + reg * lambda_;
    };

};

}
}
#endif /*CORE_LOSS_HUBER_LOSS_HPP*/

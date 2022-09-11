#ifndef CORE_LOSS_MEAN_SQUARED_ERROR_HPP
#define CORE_LOSS_MEAN_SQUARED_ERROR_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace loss {

template<typename DataType>
class MSE {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    MSE() {};
    ~MSE() {};

    /**
     * evaluate the mean squared error with the given parameters.
     *       
     * @param X ndarray of shape (num_samples, num_features), the matrix of input data
     * @param y ndarray of shape (num_samples) 
     * @param W ndarray of shape (num_features, 1) coefficient of the features 
    */
    const double evaluate(const MatType& X, 
        const VecType& y, 
        const VecType& W) const {
        
        std::size_t num_samples = X.rows();
        VecType y_hat(num_samples); 
        y_hat = X * W;

        VecType diff = y - y_hat;
        double loss = 0.5 * static_cast<double>(diff.array().pow(2.0).sum()) /
            static_cast<double>(num_samples);

        return loss;
    }

    /**
     * Evaluate the gradient of mean squared error function with the given 
     * parameters using the given batch size from the given point index
     * 
     *      dot(X.T, np.dot(X, W) - y) / len(y)
    */
    const VecType gradient(const MatType& X, 
        const VecType& y,
        const VecType& W) const{
        
        std::size_t num_samples = X.rows();
        std::size_t num_features = X.cols();

        VecType y_hat(num_samples); 
        y_hat = X * W;
        
        VecType grad(num_features);
        grad = (X.transpose() * (y_hat - y)) / (static_cast<DataType>(num_samples));
        return grad;
    };
};

}
}
#endif /*CORE_LOSS_MEAN_SQUARED_ERROR_HPP*/

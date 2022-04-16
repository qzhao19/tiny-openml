#ifndef METHODS_LINEAR_MODEL_RIDGE_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_RIDGE_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class RidgeRegression: public BaseLinearModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    
protected:
    double lambda_;

    /**fit data implementation*/
    void fit_data(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        MatType X_new = X;
        
        // if intercept term exists, append a colmun into X
        if (this->intercept_) {
            VecType ones(num_samples);
            ones.fill(1.0);
            X_new.conservativeResize(num_samples, num_features + 1);
            X_new.col(num_features) = ones;
        }
        
        num_features = X_new.cols();
        // theta = (X.T * X + lambda * eye)^(-1) * X.T * y
        MatType eyes(num_features, num_features);
        eyes.setIdentity();
        
        MatType pinv(num_features, num_features);
        pinv = X_new.transpose() * X_new + lambda_ * eyes;
        pinv = pinv.inverse();

        this->W_ = pinv * X_new.transpose() * y;
    };

public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    RidgeRegression(): BaseLinearModel<DataType>(true), 
        lambda_(0.5) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param lambda: The penalty (aka regularization term) to be used. 
     *      Constant that multiplies the regularization term
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
    */
    RidgeRegression(const double lambda, 
        const bool intercept): 
            BaseLinearModel<DataType>(intercept), 
            lambda_(lambda) {};

    /**deconstructor*/
    ~RidgeRegression() {};

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_RIDGE_REGRESSION_HPP*/

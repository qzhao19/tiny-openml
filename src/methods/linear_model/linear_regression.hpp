#ifndef METHODS_LINEAR_MODEL_LINEAR_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_LINEAR_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class LinearRegression: public BaseLinearModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

protected:
    /**
     * Train the linear regression model, using the sample weights
     * 
     * @param X, the matrix of dataset 
     * @param y, the label vector
    */
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
        // theta = (X.T * X)^(-1) * X.T * y with pseudo_inv
        MatType pinv(num_features, num_features);
        pinv = X_new.transpose() * X_new;
        pinv = pinv.inverse().eval();

        this->W_ = pinv * X_new.transpose() * y;
    };
    
public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    LinearRegression(): BaseLinearModel<DataType>(true) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param intercept_, whether or not to include an intercept term
    */
    explicit LinearRegression(bool intercept): 
        BaseLinearModel<DataType>(intercept) {};

    /**deconstructor*/
    ~LinearRegression() {};

};

} // namespace openml
} // namespace linear_model

#endif /*METHODS_LINEAR_MODEL_LINEAR_REGRESSION_HPP*/

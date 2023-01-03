#ifndef METHODS_LINEAR_MODEL_SOFTMAX_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_SOFTMAX_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class SoftmaxRegression: public BaseLinearModel<DataType> {
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
        
        
    };
    
public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    SoftmaxRegression(): BaseLinearModel<DataType>(true) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param intercept_, whether or not to include an intercept term
    */
    explicit SoftmaxRegression(bool intercept): 
        BaseLinearModel<DataType>(intercept) {};

    /**deconstructor*/
    ~SoftmaxRegression() {};

};

} // namespace openml
} // namespace linear_model

#endif /*METHODS_LINEAR_MODEL_SOFTMAX_REGRESSION_HPP*/

#ifndef METHODS_LINEAR_MODEL_PERCEPTRON_HPP
#define METHODS_LINEAR_MODEL_PERCEPTRON_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class Perceptron: public BaseLinearModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    
protected:
    double lambda_;

    /**fit data implementation*/
    void fit_data(const MatType& X, 
        const VecType& y) {
    };

public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    Perceptron(): BaseLinearModel<DataType>(true), 
        lambda_(0.5) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param lambda: The penalty (aka regularization term) to be used. 
     *      Constant that multiplies the regularization term
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
    */
    Perceptron(const double lambda, 
        const bool intercept): 
            BaseLinearModel<DataType>(intercept), 
            lambda_(lambda) {};

    /**deconstructor*/
    ~Perceptron() {};

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_PERCEPTRON_HPP*/

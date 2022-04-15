#ifndef METHODS_LINEAR_MODEL_BASE_HPP
#define METHODS_LINEAR_MODEL_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml{
namespace linear_model {

template<typename DataType>
class BaseLinearModel {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

protected:
    /**
     * @param W: ndarray_like data of shape [num_samples,]. the parameters that we want ot 
     *           calculate, initialized and filled by constructor for the least square method
    */
    VecType W_;
    
    virtual void fit_data(const MatType& X, 
        const VecType& y) = 0;

    /**
     * Calculate the predicted value y_pred for test dataset
     * 
     * @param X the test sample
     * @return Returns predicted values, ndarray of shape (num_samples,)
    */
    const VecType predict_label(const MatType& X) const {
        // y_pred = X * theta
        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType y_pred(num_samples);

        if (this->intercept) {
            y_pred = X * W_.topRows(num_features);
            VecType b(num_samples);
            b = utils::repeat<VecType>(W.bottomRows(1), num_samples, 0);
            y_pred += b;
            return y_pred;
        }
        else {
            y_pred = X * W_;
            return y_pred;
        }
    }

public:
    /**empty constructor*/
    BaseLinearModel() {};

    /**deconstructor*/
    ~BaseLinearModel() {};

    /**public get_coef interface*/
    virtual const VecType get_coef() const  = 0;


};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_BASE_HPP*/

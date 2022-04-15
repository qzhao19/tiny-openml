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
     * @param W_ ndarray_like data of shape [num_samples,]. 
     *      the parameters that we want ot calculate, initialized and 
     *      filled by constructor for the least square method
     * @param intercept_ bool, default = True. 
     *      whether to fit the intercept for the model.  
    */
    VecType W_;
    bool intercept_;
    
    /**
     * pure virtual function, will be implemeted by child method
     * fit dataset with input X
    */
    virtual void fit_data(const MatType& X, 
        const VecType& y) = 0;

    /**
     * Calculate the predicted value y_pred for test dataset
     * 
     * @param X the test sample
     * @return Returns predicted values, ndarray of shape (num_samples,)
    */
    virtual const VecType predict_label(const MatType& X) const {
        // y_pred = X * theta
        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType y_pred(num_samples);

        if (intercept_) {
            y_pred = X * W_.topRows(num_features);
            VecType b(num_samples);
            b = utils::repeat<VecType>(W_.bottomRows(1), num_samples, 0);
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
    BaseLinearModel(): intercept_(true){};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param intercept_, whether or not to include an intercept term
    */
    explicit BaseLinearModel(bool intercept): intercept_(intercept) {};

    /**deconstructor*/
    ~BaseLinearModel() {};

    /**public get_coef interface*/
    const VecType get_coef() const {
        return W_;
    };

    /**public fit interface*/
    void fit(const MatType& X, 
        const VecType& y) {
        this->fit_data(X, y);
    }

    /**public predict interface*/
    const VecType predict(const MatType& X) const{
        VecType y_pred;
        y_pred = predict_label(X);
        return y_pred;
    };
};

} // namespace openml
} // namespace linear_model

#endif /*METHODS_LINEAR_MODEL_BASE_HPP*/

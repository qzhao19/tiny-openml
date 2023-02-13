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
     * @param w_ ndarray_like data of shape [num_samples,]. 
     *      the parameters that we want ot calculate, initialized and 
     *      filled by constructor for the least square method
     * @param intercept_ bool, default = True. 
     *      whether to fit the intercept for the model.  
    */
    MatType w_;
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
            y_pred = X * w_.topRows(num_features);
            auto b = w_.bottomRows(1).value();
            return y_pred.array() + b;
        }
        else {
            y_pred = X * w_;
            return y_pred;
        }
    }

public:
    /**
     * empty constructor
    */
    BaseLinearModel(): intercept_(true){};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param intercept_, whether or not to include an intercept term
    */
    explicit BaseLinearModel(bool intercept): intercept_(intercept) {};

    /**
     * deconstructor
    */
    ~BaseLinearModel() {};

    /**public get_coef interface*/
    const VecType get_coef() const {
        return w_;
    };

    /**
     * Fit the model according to the given training data
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      Training vector, where n_samples is the number of 
     *      samples and n_features is the number of features.
     * @param y ndarray of shape [num_samples,]
     *      Target vector relative to X.
    */
    void fit(const MatType& X, 
        const VecType& y) {
        this->fit_data(X, y);
    }

    /**
     * Predict class labels for samples in X.
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      The data matrix for which we want to get the predictions.
     * 
     * @return Vector containing the class labels for each sample.
    */
    virtual const VecType predict(const MatType& X) const{
        VecType y_pred;
        y_pred = predict_label(X);
        return y_pred;
    };
};

} // namespace openml
} // namespace linear_model

#endif /*METHODS_LINEAR_MODEL_BASE_HPP*/

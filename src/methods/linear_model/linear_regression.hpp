#ifndef METHODS_LINEAR_MODEL_LINEAR_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_LINEAR_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class LinearRegression {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    
    /**
     * @param W: ndarray_like data of shape [num_samples,]. the parameters that we want ot 
     *           calculate, initialized and filled by constructor for the least square method
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
     * 
    */
    VecType W;
    bool intercept;

protected:
    /**
     * Train the linear regression model, using the sample weights
     * 
     * @param X, the matrix of dataset 
     * @param y, the label vector
    */
    void fit_(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows();

        MatType X_new = X;
        VecType y_new = y;

        // if intercept term exists, append a colmun into X
        if (intercept) {
            VecType one_mat(num_samples);
            one_mat.fill(1.0);
            X_new = utils::hstack<MatType>(X, one_mat);
        }
        
        std::size_t num_features = X_new.cols();
        // theta = (X.T * X)^(-1) * X.T * y
        MatType pseudo_inv(num_features, num_features);
        pseudo_inv = X_new.transpose() * X_new;
        pseudo_inv = pseudo_inv.inverse();

        W = pseudo_inv * X_new.transpose() * y_new;
    };

    /**
     * Calculate the predicted value y_pred for test dataset
     * 
     * @param X the test sample
     * @return Returns predicted values, ndarray of shape (num_samples,)
    */
    const VecType predict_(const MatType& X) const {
        // y_pred = X * theta
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        VecType y_pred(num_samples);
        if (intercept) {
            y_pred = X * W.topRows(num_features);
            VecType b(num_samples);
            b = utils::repeat<VecType>(W.bottomRows(1), num_samples, 0);
            y_pred += b;
            return y_pred;
        }
        else {
            y_pred = X * W;
            return y_pred;
        }
    }

public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    LinearRegression(): intercept(true){};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param intercept, whether or not to include an intercept term
    */
    LinearRegression(const bool intercept_): 
        intercept(intercept_) {};

    /**deconstructor*/
    ~LinearRegression() {};

    /**public fit interface*/
    void fit(const MatType& X, 
        const VecType& y) {
        fit_(X, y);
    }

    /**public predict interface*/
    const VecType predict(const MatType& X) const{
        VecType y_pred;
        y_pred = predict_(X);
        return y_pred;
    };

    /**public get_coef interface*/
    const VecType get_coef() const {
        return W;
    };
 
};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_LINEAR_REGRESSION_HPP*/

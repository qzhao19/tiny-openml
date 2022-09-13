#ifndef METHODS_LINEAR_MODEL_RIDGE_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_RIDGE_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template <typename DataType>
class LassoRegression: public BaseLinearModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    
protected:
    std::size_t max_iter_;
    double lambda_;
    double rho_;

    bool shuffle_;
    bool verbose_;

    /**fit data implementation*/
    void fit_data(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        MatType X_new = X;
        VecType y_new = y;

        // if intercept term exists, append a colmun into X
        if (this->intercept_) {
            VecType ones(num_samples);
            ones.fill(1.0);
            X_new.conservativeResize(num_samples, num_features + 1);
            X_new.col(num_features) = ones;
        }
        
        num_features = X_new.cols();
        VecType opt_W(num_features);
        VecType W(num_features);
        W.setZero();

        loss::MSE<DataType> mse_loss;
        optimizer::SCD<DataType> scd(X_new, 
            y_new, 
            max_iter_, 
            rho_, 
            lambda_, 
            shuffle_, 
            verbose_);
        opt_W = scd.optimize(W, mse_loss);
        this->W_ = opt_W;
    };

public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    LassoRegression(): BaseLinearModel<DataType>(true), 
            max_iter_(5000),
            lambda_(0.001), 
            rho_(1.0), 
            shuffle_(true), 
            verbose_(true) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param lambda: The penalty (aka regularization term) to be used. 
     *      Constant that multiplies the regularization term
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
    */
    LassoRegression(const std::size_t max_iter, 
        const double lambda, 
        const double rho,
        const bool shuffle,
        const bool verbose,
        const bool intercept): 
            BaseLinearModel<DataType>(intercept), 
            max_iter_(max_iter),
            lambda_(lambda), 
            rho_(rho), 
            shuffle_(shuffle), 
            verbose_(verbose) {};

    /**deconstructor*/
    ~LassoRegression() {};

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_RIDGE_REGRESSION_HPP*/

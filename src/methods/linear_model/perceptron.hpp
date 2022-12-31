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
    bool shuffle_;
    bool verbose_;
    double alpha_;
    double tol_;
    double lambda_;
    std::size_t num_iters_no_change_;
    std::size_t batch_size_;
    std::size_t max_iter_;
    std::string penalty_;

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
        VecType w(num_features);
        w.setZero();

        loss::HingeLoss<DataType> hinge_loss(lambda_, 0.0);
        optimizer::StepDecay<DataType> lr_decay(alpha_);
        optimizer::VanillaUpdate<DataType> weight_update;

        optimizer::SGD<DataType, 
            loss::HingeLoss<DataType>, 
            optimizer::VanillaUpdate<DataType>, 
            optimizer::StepDecay<DataType>> sgd(
                w, 
                hinge_loss, 
                weight_update, 
                lr_decay, 
                max_iter_, 
                batch_size_, 
                num_iters_no_change_,
                tol_, 
                shuffle_, 
                verbose_);
        sgd.optimize(X, y);
        this->w_ = sgd.get_coef();
    };

public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    Perceptron(): BaseLinearModel<DataType>(true), 
        alpha_(0.001), 
        lambda_(0.5), 
        tol_(0.0001), 
        batch_size_(512), 
        max_iter_(1000), 
        num_iters_no_change_(5),
        penalty_("l2"), 
        shuffle_(true), 
        verbose_(true) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param lambda: The penalty (aka regularization term) to be used. 
     *      Constant that multiplies the regularization term
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
    */
    Perceptron(const double alpha, 
        const double lambda,
        const double tol, 
        const std::size_t batch_size, 
        const std::size_t max_iter, 
        const std::size_t num_iters_no_change,
        const std::string penalty, 
        bool intercept,
        bool shuffle, 
        bool verbose): 
            BaseLinearModel<DataType>(intercept), 
            shuffle_(shuffle), 
            verbose_(verbose), 
            alpha_(alpha), 
            lambda_(lambda), 
            tol_(tol), 
            batch_size_(batch_size), 
            max_iter_(max_iter), 
            num_iters_no_change_(num_iters_no_change),
            penalty_(penalty_) {};

    /**deconstructor*/
    ~Perceptron() {};

    /**
     * Predict class labels for samples in X.
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      The data matrix for which we want to get the predictions.
     * 
     * @return Vector containing the class labels for each sample.
    */
    const VecType predict(const MatType& X) const {
        std::size_t num_samples = X.rows();
        VecType y_pred(num_samples);
        y_pred = this->predict_label(X);
        for(auto& value : y_pred) {
            if (value > 0.0) {
                value = 1;
            }
            else {
                value = 0;
            }
        }
        return y_pred;
    }

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_PERCEPTRON_HPP*/

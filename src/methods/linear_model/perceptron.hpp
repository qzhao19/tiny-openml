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
        VecType W(num_features);
        W.setRandom();

        loss::HingeLoss<DataType> hinge_loss(lambda_, 1.0);
        optimizer::StepDecay<DataType> lr_decay(alpha_);
        optimizer::VanillaUpdate<DataType> weight_update;
        optimizer::SGD<DataType> sgd(X_new, y_new, 
            max_iter_, 
            batch_size_, 
            alpha_, 
            tol_, 
            shuffle_, 
            verbose_);
        this->W_ = sgd.optimize(W, hinge_loss, weight_update, lr_decay);  
    };

    /**Predict confidence scores for samples.*/
    const VecType compute_decision_function(const MatType& X) const{

        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType decision_boundary(num_samples);

        decision_boundary = X * W_.topRows(num_features);

        if (this->intercept_) {
            VecType b(num_samples);
            b = utils::repeat<VecType>(W_.bottomRows(1), num_samples, 0);
            decision_boundary += b;
        }
        return decision_boundary;
    }

    /** Predict class labels for samples in X.*/
    const VecType predict_label(const MatType& X) const{
    
        // calculate the desicion boundary func
        std::size_t num_samples = X.rows();
        VecType y_pred(num_samples);

        y_pred = compute_decision_function(X);
        return y_pred;
    }


public:
    /**
     * *empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    Perceptron(): BaseLinearModel<DataType>(true), 
        shuffle_(true), 
        verbose_(false), 
        alpha_(0.001), 
        lambda_(0.5), 
        tol_(0.0001), 
        batch_size_(1), 
        max_iter_(1000), 
        penalty_("l2") {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param lambda: The penalty (aka regularization term) to be used. 
     *      Constant that multiplies the regularization term
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
    */
    Perceptron(bool intercept,
        bool shuffle, 
        bool verbose, 
        const double alpha, 
        const double lambda,
        const double tol, 
        const std::size_t batch_size, 
        const std::size_t max_iter, 
        const std::string penalty): 
            BaseLinearModel<DataType>(intercept), 
            shuffle_(shuffle), 
            verbose_(verbose), 
            alpha_(alpha), 
            lambda_(lambda), 
            tol_(tol), 
            batch_size_(batch_size), 
            max_iter_(max_iter), 
            penalty_(penalty_) {};

    /**deconstructor*/
    ~Perceptron() {};

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
        fit_data(X, y);
    }

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
        y_pred = predict_label(X);
        return y_pred;
    }

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_PERCEPTRON_HPP*/

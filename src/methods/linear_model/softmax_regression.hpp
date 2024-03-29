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
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    bool shuffle_;
    bool verbose_;
    double alpha_;
    double lambda_;
    double tol_;
    std::size_t num_iters_no_change_;
    std::size_t batch_size_;
    std::size_t max_iter_;
    std::string solver_;
    std::string penalty_;

    // MatType boundary_;

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
        VecType y_new = y;

        if (this->intercept_) {
            VecType ones(num_samples);
            ones.fill(1.0);
            X_new.conservativeResize(num_samples, num_features + 1);
            X_new.col(num_features) = ones;
        }

        std::set<DataType> label_set{y_new.begin(), y_new.end()};
        std::size_t num_classes = label_set.size();
        num_features = X_new.cols();

        MatType w(num_features, num_classes);
        w.setOnes();

        loss::SoftmaxLoss<DataType> softmax_loss(lambda_);
        optimizer::VanillaUpdate<DataType> w_update;
        optimizer::StepDecay<DataType> lr_decay(alpha_);

        optimizer::SGD<DataType, 
            loss::SoftmaxLoss<DataType>, 
            optimizer::VanillaUpdate<DataType>, 
            optimizer::StepDecay<DataType>> sgd(w, 
                softmax_loss, 
                w_update, 
                lr_decay, 
                max_iter_, 
                batch_size_, 
                num_iters_no_change_, 
                tol_, 
                shuffle_, 
                verbose_, 
                true);
        sgd.optimize(X, y);
        this->w_ = sgd.get_coef();    
    };

    const MatType compute_decision_boundary(const MatType& X) const {
        // y_pred = X * theta
        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_classes = this->w_.cols();
        MatType boundary(num_samples, num_classes);
        if (this->intercept_) {
            auto xw = X * this->w_.topRows(num_features);
            // auto b = this->w_.bottomRows(1).colwise().replicate(num_samples);
            auto b = this->w_.bottomRows(1);
            boundary = xw;
        }
        else {
            boundary = X * this->w_;
        }
        return boundary;
    }
    
public:
    /**
     * default constructor, we initialize the default value of 
     * the intercedpt is true
    */
    SoftmaxRegression(): BaseLinearModel<DataType>(true), 
        alpha_(0.01), 
        lambda_(0.0), 
        tol_(0.0001), 
        batch_size_(4), 
        max_iter_(2000), 
        num_iters_no_change_(5),
        solver_("sgd"),
        penalty_("None"), 
        shuffle_(true), 
        verbose_(true) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * @param intercept_, whether or not to include an intercept term
    */
    SoftmaxRegression(const double alpha, 
        const double lambda,
        const double tol, 
        const std::size_t batch_size, 
        const std::size_t max_iter, 
        const std::size_t num_iters_no_change,
        const std::string solver,
        const std::string penalty, 
        bool shuffle = true, 
        bool verbose = true): BaseLinearModel<DataType>(true), 
            alpha_(alpha), 
            lambda_(lambda), 
            tol_(tol), 
            batch_size_(batch_size), 
            max_iter_(max_iter), 
            num_iters_no_change_(num_iters_no_change),
            solver_(solver),
            penalty_(penalty), 
            shuffle_(shuffle), 
            verbose_(verbose) {};

    /**deconstructor*/
    ~SoftmaxRegression() {};

    /**
     * Predict class labels for samples in X.
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      The data matrix for which we want to get the predictions.
     * @return Vector containing the class labels for each sample.
    */
    const VecType predict(const MatType& X) const{
        VecType y_pred;
        auto boundary = compute_decision_boundary(X);
        auto prob = math::softmax<MatType>(boundary, 1);
        auto tmp = common::argmax<MatType, VecType, IdxVecType>(prob, 1);
        y_pred = tmp.template cast<DataType>();
        return y_pred;
    };

    /**
     * Probability estimates. The returned estimates for all 
     * classes are ordered by the label of classes.
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      The data matrix for which we want to get the predictions.
     * @return Returns the probability of the sample for each class in the model
    */
    const MatType predict_prob(const MatType& X) const {
        auto boundary = compute_decision_boundary(X);
        auto prob = math::softmax<MatType>(boundary, 1);
        return prob;
    }
};

} // namespace openml
} // namespace linear_model

#endif /*METHODS_LINEAR_MODEL_SOFTMAX_REGRESSION_HPP*/

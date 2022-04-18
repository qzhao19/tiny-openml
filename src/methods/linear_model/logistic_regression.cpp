#ifndef METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

template<typename DataType>
class LogisticRegression {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    VecType W_;
    
    bool shuffle_;
    bool verbose_;
    double alpha_;
    double lambda_;
    double tol_;
    double mu_;
    std::size_t batch_size_;
    std::size_t max_iter_;
    std::string solver_;
    std::string penalty_;
    std::string update_policy_;
    std::string decay_policy_;

protected:
    /**
     * Train the logistic model with the given optimizer
     * logistic regression must have intercept term, because we want to 
     * find a decision boundary that is able to seperate 2 class data,
     * intercept term does not exist, the decision boundary no doubt 
     * pass through the origin point.
    */
    void sgd_fit_data(const MatType& X, 
        const VecType& y) {
        
        MatType X_new = X;
        VecType y_new = y;

        std::size_t num_samples = X.rows();
        VecType ones(num_samples);
        ones.fill(1.0);
        X_new = utils::hstack<MatType>(X, ones);
        
        std::size_t num_features = X_new.cols();
        VecType W(num_features);
        W.setRandom();

        loss::LogLoss<DataType> log_loss(lambda_, penalty_);
        if (update_policy_ == "vanilla") {
            optimizer::VanillaUpdate<DataType> weight_update(alpha_);
            optimizer::SGD<DataType> sgd(X_new, y_new, 
                max_iter_, 
                batch_size_, 
                alpha_, 
                tol_, 
                shuffle_, 
                verbose_);
            sgd.optimize(log_loss, weight_update, W);   
        }
        else if (update_policy_ == "momentum") {
            optimizer::MomentumUpdate<DataType> weight_update(alpha_, mu_);
            optimizer::SGD<DataType> sgd(X_new, y_new,
            max_iter_, 
            batch_size_, 
            alpha_, 
            tol_, 
            shuffle_, 
            verbose_);
            sgd.optimize(log_loss, weight_update, W);
        }
        W_ = W;
    };

    /**Predict confidence scores for samples.*/
    const VecType compute_decision_function(const MatType& X) const{

        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType decision_boundary(num_samples);

        decision_boundary = X * W_.topRows(num_features);
        VecType b(num_samples);
        b = utils::repeat<VecType>(W_.bottomRows(1), num_samples, 0);
        decision_boundary += b;

        return decision_boundary;
    }

    /** Predict class labels for samples in X.*/
    const VecType predict_label(const MatType& X) const{
    
        // calculate the desicion boundary func
        std::size_t num_samples = X.rows();
        VecType decision_boundary(num_samples);
        VecType y_pred(num_samples);

        decision_boundary = compute_decision_function(X);
        y_pred = math::sigmoid(decision_boundary);
        for(auto& value:y_pred) {
            if (value > 0.5) {
                value = 0;
            }
            else {
                value = 1;
            }
        }
        return y_pred;
    }

    const MatType predict_label_prob(const MatType& X) const {
        // calculate the desicion boundary func
        std::size_t num_samples = X.rows();
        VecType decision_boundary(num_samples);
        decision_boundary = compute_decision_function(X);

        VecType ones(num_samples);
        ones.setOnes();

        VecType y_pred(num_samples);
        y_pred = math::sigmoid(decision_boundary);

        MatType prob(num_samples, 2);
        prob = utils::hstack<MatType>(y_pred, ones - y_pred);

        return prob;
    }

public:
    /**
     * Default constructor of logistic regression. it could custom optimizer
     * parameters. 
     * 
     * @param alpha  Learning rate when update weights
     * @param lambda L2 regularization coefficient, small values specify stronger 
     *               regularization.
     * @param tol    Tolerance for stopping criteria.
     * @param mu     momentum factor
     * @param batch_size Number of samples that will be passed through the optimizer
     *                   specifies if we apply sgd method
     * @param max_iter Maximum number of iterations taken for the solvers to converge.
     * @param solver Algorithm to use in the optimization problem default "sgd"
     * @param penalty  penalty type {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
     * @param update_policy sgd weight update policies
     * @param decay_policy weight decay type
     * 
    */
    LogisticRegression(const bool shuffle, 
        const bool verbose, 
        const double alpha, 
        const double lambda,
        const double tol, 
        const double mu,
        const std::size_t batch_size, 
        const std::size_t max_iter, 
        const std::string solver,
        const std::string penalty, 
        const std::string update_policy, 
        const std::string decay_policy): shuffle_(shuffle), 
            verbose_(verbose), 
            alpha_(alpha), 
            lambda_(lambda), 
            tol_(tol), 
            mu_(mu),
            batch_size_(batch_size), 
            max_iter_(max_iter), 
            solver_(solver),
            penalty_(penalty_), 
            update_policy_(update_policy), 
            decay_policy_(decay_policy) {};

    LogisticRegression(): shuffle_(true), 
            verbose_(false), 
            alpha_(0.001), 
            lambda_(0.5), 
            tol_(0.0001), 
            mu_(0.6),
            batch_size_(32), 
            max_iter_(1000), 
            solver_("sgd"),
            penalty_("l2"), 
            update_policy_("vanilla"), 
            decay_policy_("constant") {};


    ~LogisticRegression() {};

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
        sgd_fit_data(X, y);
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

    /**
     * Probability estimates. The returned estimates for all 
     * classes are ordered by the label of classes.
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      The data matrix for which we want to get the predictions.
     * @return Returns the probability of the sample for each class in the model
    */
    const MatType predict_prob(const MatType& X) const {
        std::size_t num_samples = X.rows();
        MatType prob(num_samples, 2);
        prob = predict_label_prob(X);
        return prob;
    }

    /**override get_coef interface*/
    const VecType get_coef() const {
        return W_;
    };

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP*/

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

    VecType W;
    
    bool shuffle;
    bool verbose;
    double alpha;
    double lambda;
    double tol;
    double mu;
    std::size_t batch_size;
    std::size_t max_iter;
    std::string solver;
    std::string penalty;
    std::string update_policy;
    std::string decay_policy;

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
        VecType one_mat(num_samples);
        one_mat.fill(1.0);
        X_new = utils::hstack<MatType>(X, one_mat);
        
        std::size_t num_features = X_new.cols();
        VecType W_(num_features);
        W_.setRandom();

        loss::LogLoss<DataType> log_loss(lambda, penalty);
        if (update_policy == "vanilla") {
            optimizer::VanillaUpdate<DataType> weight_update(alpha);
            optimizer::SGD<DataType> sgd(X_new, y_new, 
                max_iter, 
                batch_size, 
                alpha, 
                tol, 
                shuffle, 
                verbose);
            sgd.optimize(log_loss, weight_update, W_);   
        }
        else if (update_policy == "momentum") {
            optimizer::MomentumUpdate<DataType> weight_update(alpha, mu);
            optimizer::SGD<DataType> sgd(X_new, y_new,
            max_iter, 
            batch_size, 
            alpha, 
            tol, 
            shuffle, 
            verbose);
            sgd.optimize(log_loss, weight_update, W_);
        }
        W = W_;
    };

    /**
     * Predict confidence scores for samples.
    */
    const VecType compute_decision_function(const MatType& X) const{

        std::size_t num_samples = X.rows(), num_features = X.cols();
        VecType decision_boundary(num_samples);

        decision_boundary = X * W.topRows(num_features);
        VecType b(num_samples);
        b = utils::repeat<VecType>(W.bottomRows(1), num_samples, 0);
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
                value = 1;
            }
            else {
                value = 0;
            }
        }
        return y_pred;
    }

    /**
     * Probability estimates. The returned estimates for all 
     * classes are ordered by the label of classes.
    */
    const MatType predict_label_prob(const MatType& X) const {
        // calculate the desicion boundary func
        std::size_t num_samples = X.rows();
        VecType decision_boundary(num_samples);
        decision_boundary = compute_decision_function(X);


        VecType ones(num_samples);
        ones.setOnes();

        VecType y_pred_class_1(num_samples);
        VecType y_pred_class_2(num_samples);

        y_pred_class_1 = math::sigmoid(decision_boundary);
        y_pred_class_2 = ones - y_pred_class_1;

        MatType prob(num_samples, 2);
        prob = utils::hstack(y_pred_class_1, y_pred_class_2);
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
    LogisticRegression(const bool shuffle_, 
        const bool verbose_, 
        const double alpha_, 
        const double lambda_,
        const double tol_, 
        const double mu_,
        const std::size_t batch_size_, 
        const std::size_t max_iter_, 
        const std::string solver_,
        const std::string penalty_, 
        const std::string update_policy_, 
        const std::string decay_policy_): shuffle(shuffle_), 
            verbose(verbose_), 
            alpha(alpha_), 
            lambda(lambda_), 
            tol(tol_), 
            mu(mu_),
            batch_size(batch_size_), 
            max_iter(max_iter_), 
            solver(solver_),
            penalty(penalty_), 
            update_policy(update_policy_), 
            decay_policy(decay_policy_) {};

    LogisticRegression(): shuffle(true), 
            verbose(false), 
            alpha(0.001), 
            lambda(0.5), 
            tol(0.0001), 
            mu(0.6),
            batch_size(32), 
            max_iter(1000), 
            solver("sgd"),
            penalty("l2"), 
            update_policy("vanilla"), 
            decay_policy("constant") {};


    ~LogisticRegression() {};

    /**public fit interface*/
    void fit(const MatType& X, 
        const VecType& y) {
        sgd_fit_data(X, y);
    }

    /**
     * Predict interface class labels for samples in X.
    */
    const VecType predict(const MatType& X) const {
        std::size_t num_samples = X.rows();
        VecType y_pred(num_samples);

        y_pred = predict_label(X);

        return y_pred;
    }

    const MatType predict_prob(const MatType& X) const {
        std::size_t num_samples = X.rows();

        MatType prob(num_samples, 2);

        prob = predict_label_prob(X);

        return prob;
    }


};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP*/

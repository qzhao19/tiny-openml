#ifndef METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace regression {

template<typename DataType>
class LogisticRegression {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    VecType W;
    
    bool shuffle;
    bool verbose;
    DataType alpha;
    DataType lambda;
    DataType tol;
    DataType mu;
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
    void fit_data(const MatType& X, 
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
            optimizer::VanillaUpdate<double> weight_update(alpha);
            optimizer::SGD<double> sgd(X_new, y_new, 
                max_iter, 
                batch_size, 
                alpha, 
                tol, 
                shuffle, 
                verbose);
            sgd.optimize(log_loss, weight_update, W_);   
        }
        else if (update_policy == "momentum") {
            optimizer::MomentumUpdate<double> weight_update(alpha, mu);
            optimizer::SGD<double> sgd(X_new, y_new,
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
        const DataType alpha_, 
        const DataType lambda_,
        const DataType tol_, 
        const DataType mu_,
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

    ~LogisticRegression() {};

    /**public fit interface*/
    void fit(const MatType& X, 
        const VecType& y) {
        fit_data(X, y);
    }

};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP*/

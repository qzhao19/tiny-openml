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
    std::string penalty;
    std::string update_policy;
    std::string decay_policy;

protected:
    /**
     * 
    */
    void fit_data(const MatType& X, 
        const VecType& y) {
        std::size_t num_features = X.cols();
        VecType weight(num_features);
        weight.setRandom();

        loss::LogLoss<DataType> log_loss(lambda, penalty);
        if (update_policy == "vanilla") {
            optimizer::VanillaUpdate<double> weight_update(alpha);
            optimizer::SGD<double> sgd(X, y, 
                max_iter, 
                batch_size, 
                alpha, 
                tol, 
                shuffle, 
                verbose);
            sgd.optimize(log_loss, weight_update, weight);   
        }
        else if (update_policy == "momentum") {
            optimizer::MomentumUpdate<double> weight_update(alpha, mu);
            optimizer::SGD<double> sgd(X, y, 
            max_iter, 
            batch_size, 
            alpha, 
            tol, 
            shuffle, 
            verbose);
            sgd.optimize(log_loss, weight_update, weight);
        }
        W = weight;
    };

public:

    LogisticRegression(const bool shuffle_, 
        const bool verbose_, 
        const DataType alpha_, 
        const DataType lambda_,
        const DataType tol_, 
        const DataType mu_,
        const std::size_t batch_size_, 
        const std::size_t max_iter_, 
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

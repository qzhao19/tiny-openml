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
    std::size_t batch_size;
    std::size_t max_iter;
    std::string penalty;
    std::string update_policy;
    std::string decay_policy;

protected:


    void fit_data(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_features = X.cols();

        loss::LogLoss<DataType> log_loss;
        
        VecType W(num_features);

        
        W = Eigen::MatrixXd::Random(num_features, 1);

    };


public:

    LogisticRegression(const bool shuffle_, 
        const bool verbose_, 
        const DataType alpha_, 
        const DataType lambda_,
        const DataType tol_, 
        const std::size_t batch_size_, 
        const std::size_t max_iter_, 
        const std::string penalty_, 
        const std::string update_policy_, 
        const std::string decay_policy_): shuffle(shuffle_), 
            verbose(verbose_), 
            alpha(alpha_), 
            lambda(lambda_), 
            tol(tol_), 
            batch_size(batch_size_), 
            max_iter(max_iter_), 
            penalty(penalty_), 
            update_policy(update_policy_), 
            decay_policy(decay_policy_) {};

    ~LogisticRegression() {};




};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP*/

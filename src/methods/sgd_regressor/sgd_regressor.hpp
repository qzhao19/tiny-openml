#ifndef METHOD_SGD_REGRESSION_SGD_REGRESSION_HPP
#define METHOD_SGD_REGRESSION_SGD_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace loss;
using namespace math;

namespace regression {

class SGDRegressor {

public:
    SGDRegressor(): loss("squared_error"),
        penalty("l2"), 
        learning_rate("optimal"),
        lambda(0.0001),
        l1_ratio(0.15),
        batch_size(32),
        max_iter(100), 
        shuffle(true) {};

    SGDRegressor(const std::string loss_, 
        const std::string penalty_ = "l2", 
        const std::string learning_rate_ = "optimal",
        const double lambda_ = 0.0001,
        const double l1_ratio_ = 0.15,
        const std::size_t batch_size_ = 32, 
        const std::size_t max_iter_ = 100, 
        const bool shuffle_ = true): 
            loss(loss_), 
            penalty(penalty_), 
            learning_rate(learning_rate_), 
            lambda(lambda_), 
            l1_ratio(l1_ratio_),
            batch_size(batch_size_), 
            max_iter(max_iter_), 
            shuffle(shuffle_){}
    
    ~SGDRegressor() {};

    void fit(const arma::mat& X, const arma::vec& y);

protected:
    template<class LossFunctionType>
    void fit(const arma::mat& X, 
        const arma::vec& y, 
        LossFunctionType& loss_fn_type);

private:
    arma::vec W;

    std::size_t batch_size;

    std::size_t max_iter;

    double lambda;

    double l1_ratio;

    std::string penalty;

    std::string loss;

    std::string learning_rate;

    bool shuffle;

};

};

#endif

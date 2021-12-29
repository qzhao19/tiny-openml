#ifndef METHOD_SGD_REGRESSION_SGD_REGRESSION_HPP
#define METHOD_SGD_REGRESSION_SGD_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace regression {

class SGDRegressor {

    SGDRegressor(): max_iters(10000), 
        alpha(0.0001),
        penalty("l2"),
        loss("squared_error"),
        learning_rate("optimal"), 
        shuffle(true) {};

    ~SGDRegressor() {};


    template<typename LossFunction>
    void fit(const arma::mat& X, const arma::vec& y);




private:
    arma::vec theta;

    int max_iters;

    double alpha;

    double l1_ratio;

    std::string penalty;

    std::string loss;

    std::string learning_rate;

    bool shuffle;

};

};

#endif

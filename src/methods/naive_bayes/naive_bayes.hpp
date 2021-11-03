#ifndef METHOD_NAIVE_BAYES_NAIVE_BAYES_HPP
#define METHOD_NAIVE_BAYES_NAIVE_BAYES_HPP
#include "../../core.hpp"
#include "../../prereqs.hpp"


namespace naive_bayes {


class NaiveBayes {
public:

    NaiveBayes(): var_smoothing(1e-9) {};

    ~NaiveBayes() {};

protected:

    /**
     * Compute the unnormalized prior log probability of y
     * I.e. ``log P(c)`` as an array-like of shape 
     * (n_classes, n_samples).
    */
    void get_log_class_prior_prob(const arma::vec& y);
    
    /**
     * Compute online update of Gaussian mean and variance.
    */
    void update_mean_variance(const arma::mat& X, 
        const arma::vec& y);


private:
    arma::vec log_class_prior_prob;

    arma::vec means;

    arma::vec vars;


    std::unordered_map<double, int> label_map;


    bool fit_prior;

    double var_smoothing;

    double binarize;

    int n_classes;

    
    

};

};

#endif

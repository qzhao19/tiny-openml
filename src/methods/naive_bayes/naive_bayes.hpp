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

    void gaussian_train(const arma::mat& X, 
        const arma::vec& y);


private:
    /**
     * @param log_class_prior_prob Vector for log prior probabilty for each class 
     * @param means Matrix of shape [n_classes, n_feature], measn of each feature for
     *              different class
     * @param vars  Matrix of shape [n_classes, n_feature], variances of each feature for
     *              different class
     * @param label_map Hash map containing each label and their count numbers 
    */
    arma::vec log_class_prior_prob;

    arma::mat means;

    arma::mat vars;

    std::unordered_map<double, double> label_map;

    bool fit_prior;

    double var_smoothing;

    double binarize;

    int n_classes;


};

};

#endif

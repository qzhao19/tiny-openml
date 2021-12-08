#ifndef METHOD_NAIVE_BAYES_NAIVE_BAYES_HPP
#define METHOD_NAIVE_BAYES_NAIVE_BAYES_HPP
#include "../../core.hpp"
#include "../../prereqs.hpp"

namespace naive_bayes {

class NaiveBayes {
public:

    NaiveBayes(): solver("gaussian"), 
        var_smoothing(1e-9), 
        alpha(0.0) {};


    NaiveBayes(const std::string solver_, 
        const double var_smoothing_, 
        const double alpha_) :
            solver(solver_),
            var_smoothing(var_smoothing_), 
            alpha(alpha_) {}
    
    
    ~NaiveBayes() {};

    void fit(const arma::mat& X, 
        const arma::vec& y);

    const arma::vec predict(const arma::mat& X);

    const arma::mat predict_prob(const arma::mat& X);

    const arma::mat predict_log_prob(const arma::mat& X);


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

    const arma::vec joint_log_likelihood(const arma::rowvec& x) const;

    const std::pair<double, arma::vec> predict_prob_label(const arma::rowvec& x) const;

    // void predict_X(const arma::mat& X);

private:
    /**
     * @param log_class_prior_prob Vector for log prior probabilty for each class 
     * @param log_prob  
     * @param means Matrix of shape [n_classes, n_feature], measn of each feature for
     *              different class
     * @param vars  Matrix of shape [n_classes, n_feature], variances of each feature for
     *              different class
     * @param label_map Hash map containing each label and their count numbers 
    */
    
    arma::mat log_joint_prob;

    arma::mat means;

    arma::mat vars;

    arma::vec log_class_prior_prob;

    arma::vec feature_count;

    arma::vec class_count;

    arma::vec feature_log_prob;

    std::map<double, double> label_map;

    std::string solver;

    bool fit_prior;

    double var_smoothing;

    double alpha;

    int n_classes;


};

};

#endif

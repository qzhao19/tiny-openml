#include "naive_bayes.hpp"

using namespace math;
using namespace naive_bayes;


void NaiveBayes::get_log_class_prior_prob(const arma::vec& y) {

    int n_samples = y.n_rows;

    std::unordered_map<double, int> counter;

    for (std::size_t i = 0; i < n_samples; i++) {
        counter[y[i]]++;
    }

    std::vector<double> class_prior_prob;

    n_classes = 0;
    for (auto cnt : counter) {
        label_map[cnt.first] = n_classes;
        n_classes++;

        class_prior_prob.push_back(double(cnt.second / n_samples));
    }

    log_class_prior_prob = arma::conv_to<arma::vec>::from(class_prior_prob);
    log_class_prior_prob = arma::log(log_class_prior_prob);

}


void NaiveBayes::update_mean_variance(const arma::mat& X, 
    const arma::vec& y) {
    
    

}


#include "naive_bayes.hpp"

using namespace math;
using namespace naive_bayes;


void NaiveBayes::get_log_class_prior_prob(const arma::vec& y) {

    int n_samples = y.n_rows;
    std::vector<double> class_prior_prob;

    for (std::size_t i = 0; i < n_samples; i++) {
        label_map[y[i]]++;
    }

    n_classes = 0;
    for (auto &label : label_map) {
        n_classes++;
        class_prior_prob.push_back(label.second / n_samples);
    }

    log_class_prior_prob = arma::conv_to<arma::vec>::from(class_prior_prob);
    log_class_prior_prob = arma::log(log_class_prior_prob);

}


void NaiveBayes::update_mean_variance(const arma::mat& X, 
    const arma::vec& y) {
    
    int n_features = X.n_cols;
    arma::mat X_y = arma::join_rows(X, y);

    std::size_t i = 0;
    for (auto &label : label_map) {
        arma::mat partial_X_y= X.rows(arma::find(X_y.col(n_features) == label.first));
        arma::rowvec partial_means = arma::mean(partial_X_y, 0);
        arma::rowvec partial_vars = arma::var(partial_X_y, 0, 0);

        means.insert_rows(i, partial_means);
        vars.insert_rows(i, partial_vars);
        i++;
    }
}













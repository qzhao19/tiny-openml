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

const arma::vec NaiveBayes::joint_log_likelihood(
    const arma::rowvec& x) const {
    std::size_t n_features = x.n_elem;

    std::vector<double> likelihoods_;

    arma::vec likelihoods;

    for (std::size_t i = 0; i < n_classes; i++) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n_features; j++) {
            double val = gaussian_fn(x(j), means(i, j), vars(i, j));
            sum += val;
        }
        likelihoods_.push_back(sum);
    }

    likelihoods = arma::conv_to<arma::vec>::from(likelihoods_);
    arma::vec joint_prob = likelihoods + log_class_prior_prob;

    return joint_prob;

}

const std::pair<double, arma::vec> NaiveBayes::predict_prob_label(
    const arma::rowvec& x) const {
    
    arma::vec joint_prob = joint_log_likelihood(x);
    std::size_t i = 0;
    std::map<double, double> joint_prob_map;
    for (auto& label : label_map) {
        joint_prob_map[label.first] = joint_prob(i);
        i++;
    }

    auto max_joint_prob_label = utils::max_element(joint_prob_map);
    std::pair<double, arma::vec> retval = {max_joint_prob_label.first, joint_prob};

    return retval;

}

const arma::vec NaiveBayes::predict(const arma::mat& X) {
    std::pair<double, arma::vec> retval;

    std::vector<double> y_pred_;

    for (std::size_t i = 0; i < X.n_rows; i++) {
        retval = predict_prob_label(X.row(i));
        y_pred_.push_back(retval.first);
        log_joint_prob.insert_cols(i, retval.second);
    }
    arma::vec y_pred = arma::conv_to<arma::vec>::from(y_pred_);

    return y_pred;
}

void NaiveBayes::fit(const arma::mat& X, 
    const arma::vec& y) {
    
    // gaussian_train(X, y);
    if (solver == "gaussian") {
        get_log_class_prior_prob(y);
        update_mean_variance(X, y);
        vars = vars + var_smoothing * vars.max();
    }
}


const arma::mat NaiveBayes::predict_log_prob(
    const arma::mat& X) {
    return arma::trans(log_joint_prob);
}


const arma::mat NaiveBayes::predict_prob(
    const arma::mat& X) {
    
    arma::mat log_prob = arma::trans(log_joint_prob);

    return arma::exp(log_prob);
}

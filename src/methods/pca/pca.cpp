#include "pca.hpp"

using namespace math;
using namespace pca;

void PCA::scale_data(arma::mat& X) const {
    arma::vec scaled_vec = arma::stddev(X, 0, 1);
    // if there are any zero, make them small
    for (std::size_t i = 0; i < scaled_vec.n_elem; i++) {
        if (scaled_vec[i] == 0) {
            scaled_vec[i] = std::numeric_limits<double>::max();;
        }
    }
    X = arma::repmat(scaled_vec, 1, X.n_cols);
}

const arma::vec PCA::score_samples(const arma::mat& X) const {
    arma::mat centered_X;
    math::center(X, centered_X);
    std::size_t n_features = centered_X.n_cols;

    arma::mat cov_X = arma::cov(centered_X);
    arma::mat precision = arma::pinv(cov_X);
    arma::vec log_like;

    log_like = -0.5 * arma::sum((centered_X * arma::trans(centered_X * precision)), 1);
    log_like -= -0.5 * (n_features * std::log(2.0 * arma::datum::pi) - math::logdet(precision));

    return log_like;
}

void PCA::fit_(const arma::mat& X) {
    std::size_t n_samples = X.n_rows;
    std::size_t n_features = X.n_cols;

    arma::vec explained_var_;

    if (solver == "svd") {
        arma::mat U;
        arma::vec s;
        arma::mat Vt;

        std::tie(U, s, Vt) = math::svd<arma::mat, arma::vec>(X);
        
        //  math::svd(X, U, s, Vt);

        // flip eigenvectors' sign to enforce deterministic output
        std::tie(U, Vt) = math::svd_flip<arma::mat>(U, Vt);
        
        explained_var_ = arma::pow(s, 2) / (static_cast<double>(n_samples) - 1.);
        components = Vt.cols(0, n_components - 1);
    }
    double total_var = arma::sum(explained_var_);
    arma::vec explained_var_ratio_ = explained_var_ / total_var;

    // get expalined variance and explained variance rotion, convert its to rowvec
    explained_var = arma::conv_to<arma::rowvec>::from(explained_var_);
    explained_var_ratio = arma::conv_to<arma::rowvec>::from(explained_var_ratio_);
    
    if (n_components < std::min(n_samples, n_features)) {
        noise_variance = arma::mean(explained_var.cols(0, n_components - 1));
    }
    else {
        noise_variance = 0.0;
    }
}

void PCA::fit(const arma::mat& X) {
    arma::mat centered_X;
    math::center(X, centered_X);
    fit_(centered_X);
}

const arma::mat PCA::transform(const arma::mat& X) {
    arma::mat centered_X;
    math::center(X, centered_X);

    return centered_X * components; 
}

const double PCA::score(const arma::mat& X) {
    arma::vec log_like;
    log_like = score_samples(X);

    return arma::mean(log_like);
}

/**
 * cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)
 * where S**2 contains the explained variances, and sigma2 contains the 
 * noise variances.
*/
arma::mat PCA::get_covariance() {

    std::size_t n_features = explained_var.n_elem;

    // exp_var is a vector, so need to convert it to a diag matrix 
    // then get the submatrix with the shape of [n_componenets, n_componenets] 
    // which contains the explained var
    arma::mat exp_var_ = arma::diagmat(explained_var);
    arma::mat exp_var = exp_var_.submat(0, 0, n_components - 1, n_components - 1);
    arma::mat exp_var_square = arma::pow(exp_var, 2);

    // calc the conv
    arma::mat cov;
    cov = components * exp_var_square * components.t();
    cov += (arma::eye(n_features, n_features) * noise_variance);

    return cov;
}

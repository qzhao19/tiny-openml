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

const arma::mat PCA::eig_train(const arma::mat& X) {
    
    arma::vec eig_val;
    arma::mat eig_vec;

    EigPolicy eig;
    eig.Apply(X, eig_val, eig_vec);

    eig_vec = arma::reverse(eig_vec, 1);

    // calc explained variance 
    std::size_t n_samples = X.n_rows;
    arma::vec explained_var_ = arma::pow(eig_val, 2) / (n_samples - 1);

    double total_var = arma::sum(explained_var_);

    arma::vec explained_var_ratio_ = explained_var_ / total_var;

    // get expalined variance and explained variance rotion, convert its to rowvec
    
    explained_var = arma::conv_to<arma::rowvec>::from(explained_var_);
    explained_var_ratio = arma::conv_to<arma::rowvec>::from(explained_var_ratio_);


    return eig_vec.cols(0, n_components - 1);

}

template<typename DecompositionPolicy>
const arma::mat PCA::svd_train(const arma::mat& X, 
        DecompositionPolicy& decomposition_policy) {
    
    arma::mat U;
    arma::vec s;
    arma::mat Vt;

    std::size_t n_samples = X_new.n_rows;

    decomposition_policy.Apply(X, U, s, Vt);

    // calc explained variance 
    arma::vec explained_var_ = arma::pow(s, 2) / (n_samples - 1);

    double total_var = arma::sum(explained_var_);

    arma::vec explained_var_ratio_ = explained_var_ / total_var;

    // get expalined variance and explained variance rotion, convert its to rowvec
    explained_var = arma::conv_to<arma::rowvec>::from(explained_var_);
    explained_var_ratio = arma::conv_to<arma::rowvec>::from(explained_var_ratio_);
    

    return Vt.cols(0, n_components - 1);
}

void PCA::fit(const arma::mat& X) {

    arma::mat retmat;
    arma::mat centered_X;
    math::center(X, centered_X);

    if (scale) {
        scale_data(centered_X);
    }

    if (solver == "eig") {
        retmat = eig_train(centered_X);
    }
    else if (solver == "full_svd") {
        ExactSvdPlicy svd_policy;
        retmat = svd_train(centered_X, svd_policy);
    }

    components = retmat;
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

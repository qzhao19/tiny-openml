#include "pca.hpp"

using namespace math;
using namespace pca;

void PCA::scale_data(arma::mat& X) {
    arma::vec scaled_vec = arma::stddev(X, 0, 1);
    // if there are any zero, make them small

    for (std::size_t i = 0; i < scaled_vec.n_elem; i++) {
        if (scaled_vec[i] == 0) {
            scaled_vec[i] = 1e-30;
        }
    }
    X = arma::repmat(scaled_vec, 1, X.n_cols);
}

const arma::mat PCA::eig_train(const arma::mat& X) const {
    
    arma::vec eig_val;
    arma::mat eig_vec;

    EigPolicy eig;
    eig.Apply(X, eig_val, eig_vec);

    eig_vec = arma::reverse(eig_vec, 1);

    return eig_vec.cols(0, n_components - 1);

}

template<typename DecompositionPolicy>
const arma::mat PCA::svd_train(const arma::mat& X, 
        DecompositionPolicy& decomposition_policy) const {
    
    arma::mat U;
    arma::vec s;
    arma::mat V;

    decomposition_policy.Apply(X, U, s, V);
    
    return V.cols(0, n_components - 1);
}

void PCA::fit(const arma::mat& X) {

    arma::mat retmat;
    arma::mat centered_X;
    math::center(X, centered_X);

    if (scale) {
        scale_data(centered_X);
    }

    if (solver == "auto") {
        retmat = eig_train(centered_X);
    }
    else if (solver == "full_svd") {
        ExactSvdPlicy svd_policy;
        retmat = svd_train(centered_X, svd_policy);
    }

}



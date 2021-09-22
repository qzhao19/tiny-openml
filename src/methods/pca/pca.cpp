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
    arma::mat centered_X;

    math::center(X, centered_X);

    arma::vec eig_val;
    arma::mat eig_vec;

    EigPolicy eig;
    eig.Apply(centered_X, eig_val, eig_vec);

    return eig_vec.cols(n_components - 1, X.n_cols - 1);

}

template<typename DecompositionPolicy>
const arma::mat PCA::svd_train(const arma::mat& X, 
        DecompositionPolicy& decomposition_policy) const {
    
    arma::mat centered_X;
    math::center(X, centered_X);

    arma::mat U;
    arma::vec s;
    arma::mat V;

    decomposition_policy.Apply(centered_X, U, s, V);

    
}



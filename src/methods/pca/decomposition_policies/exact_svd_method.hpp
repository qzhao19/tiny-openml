#ifndef METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
#define METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
#include "../../../prereqs.hpp"

namespace pca {

class ExactSvdPlicy {
public:
    /**
     * Implementaion of exact svd method using API that provided by arma
     * 
     * @param X Data matrix of shape (n_samples, n_features)
     * @param X_centered Centered data matrix, data minus the mean of X
     * @param U Matrix to put eigen vector
     * @param s Vector to put eigen values
     * @param Vt Matrix store the right singualr values
    */
    void Apply(const arma::mat& X, 
        arma::mat& U, 
        arma::vec& s, 
        arma::mat Vt) {
                
        // if ncols are much larger than nrows, need to use economical svd
        if (X.n_cols > X.n_rows) {
            // economical singular value decomposition and compute only the left
            // singular vectors
            arma::svd_econ(U, s, Vt, X, "left");
        }
        else {
             arma::svd(U, s, Vt, X);
        }   
        s %= s / (X.n_cols - 1);
    };

};

};

#endif

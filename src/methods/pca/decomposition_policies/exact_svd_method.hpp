#ifndef METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
#define METHODS_PCA_DECOMPOSITION_POLICIES_EXACT_SVD_METHOD_HPP
#include "../../../prereqs.hpp"

namespace pca {

class ExactSvdPlicy {
public:
    /**
     * Implementaion of exact svd method using API that provided by arma
     * 
     * @param X Data matrix of shape (n_samples, n_faetures)
     * @param X_centered Centered data matrix, data minus the mean of X
     * @param X_transformed Matrix to put result of pca
     * @param U Matrix to put eigen vector
     * @param s Vector to put eigen values
     * @param rank Rank of the decomposition
    */
    void Apply(const arma::mat& X, 
        const arma::mat& X_centered, 
        arma::mat& X_transformed, 
        arma::mat& U, 
        arma::vec& s, 
        const std::size_t rank) {
        
        // this matrix store the right singualr values
        arma::mat V;

        // if ncols are much larger than nrows, need to use economical svd
        if (X.n_cols > X.n_rows) {
            // economical singular value decomposition and compute only the left
            // singular vectors
            arma::svd_econ(U, s, V, X_centered, "left");
        }
        else {
             arma::svd(U, s, V, X_centered);
        }   

        s %= s / (X.n_cols - 1);

        // project the samples into the principle 
        X_transformed = arma::trans(s) * X_transformed;

    };

};

};

#endif

#ifndef METHODS_PCA_DECOMPOSITION_POLICIES_EIG_METHOD_HPP
#define METHODS_PCA_DECOMPOSITION_POLICIES_EIG_METHOD_HPP
#include "../../../prereqs.hpp"

namespace pca {

class EigPolicy {
public:
    /**
     * Implementaion of exact svd method using API that provided by arma
     * 
     * @param X Data matrix of shape (n_samples, n_faetures)
     * @param eig_vec Matrix to put eigen vector
     * @param eig_val Vector to put eigen values
     * @param rank Rank of the decomposition
    */
    void Apply(const arma::mat& X, 
        arma::vec& eig_val, 
        arma::mat& eig_vec) {
        
        arma::eig_sym(eig_val, eig_vec, X);   
    };

};

};

#endif

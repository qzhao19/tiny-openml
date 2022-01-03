#ifndef CORE_MATH_LINALG_HPP
#define CORE_MATH_LINALG_HPP
#include "../../prereqs.hpp"

namespace math {

/**
 * Eigen vector decomposition
 * @param X ndarray of shape (n_samples, n_features),
 *          matrices for which the eigenvalues and right 
 *          eigenvectors will be computed 
 * 
 * @return std::tuple(eigenvector, eigenvalue), the eigenvalues and 
 *         the eigenvectors
*/
template<typename MatType, typename VecType>
std::tuple<MatType, VecType> eig(const MatType& X) {
    MatType eig_vec;
    VecType eig_val;
    arma::eig_gen(eig_val, eig_vec, X);
    return std::make_tuple(eig_vec, eig_val);
};

/**
 * Singular Value Decomposition
 * @param X ndarray of shape (n_samples, n_features),
 *          A real or complex array
 * @param full_matrices, bool, default = false
 *        If True (default), u and vh have the shapes (..., M, M) and (..., N, N). 
 *        Otherwise, the shapes are (..., M, K) and (..., K, N), respectively, where K = min(M, N).
*/
template<typename MatType, typename VecType>
std::tuple<MatType, VecType, MatType> svd(const MatType& X, 
    bool full_matrices = false) {
    MatType U;
    VecType s; 
    MatType Vt;

    if (full_matrices) {
        arma::svd(U, s, Vt, X);
    }
    else {
        arma::svd_econ(U, s, Vt, X);
    }
    return std::make_tuple(U, s, Vt);
};

};
#endif

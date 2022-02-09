#ifndef CORE_MATH_LINALG_HPP
#define CORE_MATH_LINALG_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

/**
 * Singular Value Decomposition
 * @param X ndarray of shape (n_samples, n_features),
 *          A real or complex array
 * @param full_matrices, bool, default = false
 *        If True (default), u and vh have the shapes (M, M) and (N, N). 
 *        Otherwise, the shapes are (M, K) and (K, N), respectively, where K = min(M, N).
 * @return a tuple contains U matrix, s vector and Vt matrix. their shape are repectively 
 *      {(M, M), (M, K)}, (K), {(N, N), (K, N)}
*/
template<typename MatType, typename VecType>
std::tuple<MatType, VecType, MatType> exact_svd(const MatType& mat, 
    bool full_matrices = false) {
    MatType U;
    VecType s; 
    MatType Vt;

    std::size_t num_features = mat.cols();
    // control the switching size, default is 16
    // For small matrice (<16), it is thus preferable to directly use JacobiSVD. 
    // For larger matrice, BDCSVD
    if (num_features < 16) {
        if (full_matrices) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            s = svd.singularValues();
            Vt = svd.matrixV();
        }
        else {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            s = svd.singularValues();
            Vt = svd.matrixV();
        }
    }
    else {
        if (full_matrices) {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            s = svd.singularValues();
            Vt = svd.matrixV();
        } 
        else {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            s = svd.singularValues();
            Vt = svd.matrixV();
        }
    }

    return std::make_tuple(U, s, Vt);
};

}
}
#endif

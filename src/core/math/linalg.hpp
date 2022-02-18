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
std::tuple<MatType, VecType, MatType> exact_svd(const MatType& x, 
    bool full_matrix = false) {
    MatType U;
    VecType s; 
    MatType V;

    std::size_t num_features = x.cols();
    // control the switching size, default is 16
    // For small matrice (<16), it is thus preferable to directly use JacobiSVD. 
    // For larger matrice, BDCSVD
    if (num_features < 16) {
        if (full_matrix) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            s = svd.singularValues();
            V = svd.matrixV();
        }
        else {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            s = svd.singularValues();
            V = svd.matrixV();
        }
    }
    else {
        if (full_matrix) {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeFullU | Eigen::ComputeFullV);
            U = svd.matrixU();
            s = svd.singularValues();
            V = svd.matrixV();
        } 
        else {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU();
            s = svd.singularValues();
            V = svd.matrixV();
        }
    }

    return std::make_tuple(U, s, V);
};

/**
 * Compute the (Moore-Penrose) pseudo-inverse of a matrix. 
 * Calculate the generalized inverse of a matrix using its 
 * singular-value decomposition (SVD) and including all 
 * large singular values.
 * 
 * @param x matrix to be pseudo-inverted.
 * @param tol double cutoff for small singular values.
 * 
 * @return the pseudo-inverse of a
*/
template<typename MatType, typename VecType, 
    typename DataType = typename MatType::value_type>
MatType pinv(const MatType& x, double tol = 1.e-6) {

    MatType U;
    VecType s; 
    MatType Vt;
    std::tie(U, s, Vt) = exact_svd<MatType, VecType>(x, true);

    std::size_t num_rows = x.rows(), num_cols = x.cols();
    MatType s_inv(num_cols, num_rows);
    s_inv.setZero();

    for(std::size_t i = 0; i < s.size(); i++) {
        if (s(i) > tol) {
            s_inv(i, i) = static_cast<DataType>(1) / s(i);
        }
        else {
            s_inv(i, i) = static_cast<DataType>(0);
        }
    }

    MatType pinv_mat = Vt * s_inv * U.transpose();
    return pinv_mat;
}


/**
 * Compute log(det(A)) for A symmetric.
 * 
 * @param x ndarray of shape (num_rows, num_cols)
 *      input array, has to be a SQUARE 2d array
 * @return -Inf if det(A) is non positive or is not defined.
*/
template <typename MatType, 
    typename DataType = typename MatType::value_type>
DataType logdet(const MatType& x) {
    DataType ld = static_cast<DataType>(0);
    Eigen::PartialPivLU<MatType> lu(x);
    auto& LU = lu.matrixLU();
    DataType c = lu.permutationP().determinant(); // -1 or 1
    for (std::size_t i = 0; i < LU.rows(); ++i) {
        const auto& lii = LU(i,i);
        if (lii < static_cast<DataType>(0)) {
            c *= -1;
        }
        ld += std::log(std::abs(lii));
    }
    ld += std::log(c);
    return std::isnan(ld) ? (-std::numeric_limits<DataType>::infinity()) : ld;
}






}
}
#endif

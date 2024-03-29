#ifndef CORE_MATH_LINALG_HPP
#define CORE_MATH_LINALG_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace math {

/**
 * @brief Compute LU decomposition of a matrix.
 * @param x ndarray of shape (nrows, ncols)
 *      Matrix to decomposition
 * @param permute_l bool
 *      Perform the multiplication P*L
 * @return a tuple contains P permutation matrix, 
 *      L lower matrix, U upper matrix, if permute_l
 *      is false, otherwise returns a tuple containing
 *      P * L, lower matrix, upper matrix
 *      P is (M, M), L is (M, K), K = min(M, N), 
 *      U is (K, N) 
*/
template<typename MatType>
std::tuple<MatType, MatType, MatType> lu(const MatType& x, 
    bool permute_l = false) {

    std::size_t nrows = x.rows(), ncols = x.cols();
    std::size_t min_size = std::min(nrows, ncols);
    Eigen::FullPivLU<MatType> lu_decomposition(x);

    MatType lu_m = lu_decomposition.matrixLU();
    // L is (M, min(M, N))
    MatType L = lu_m.template triangularView<Eigen::UnitLower>().toDenseMatrix();
    L = L.leftCols(min_size).eval();

    // U is (min(M, N), K)
    MatType U = lu_m.template triangularView<Eigen::Upper>();
    U = U.topRows(min_size).eval();

    MatType P = lu_decomposition.permutationP();
    // MatType Q = lu_decomposition.permutationQ();
    MatType Q;
    if (permute_l) {
        Q = (P * L);
    }
    else {
        Q = P;
    }
    return std::make_tuple(Q, L, U);
};

/**
 * @brief Compute QR decomposition of a matrix.
 *      Calculate the decomposition X = QR where 
 *      Q is unitary/orthogonal and R upper triangular
 * 
 * @param x ndarray of shape (nrows, ncols)
 *      Matrix to decomposition
 * @param full_matrix bool, default true
 *      If full_matrices_ is true then Q is m x m and R is m x n
 *      Otherwise, Q is m x min(m, n), and R is min(m, n) x n.
*/
template<typename MatType>
std::tuple<MatType, MatType> qr(const MatType& x, 
    bool full_matrix = false) {
    const std::size_t nrows = x.rows(), ncols = x.cols();
    const int min_size = std::min(nrows, ncols);
    Eigen::HouseholderQR<MatType> qr_decomposition(x);

    MatType Q, R;
    if (full_matrix) {
        Q = qr_decomposition.householderQ();
        R = qr_decomposition.matrixQR().template triangularView<Eigen::Upper>();
    }
    else {
        MatType tmp(nrows, min_size);
        tmp.setIdentity();

        Q = qr_decomposition.householderQ() * tmp;
        auto qr_top = qr_decomposition.matrixQR().block(0, 0, min_size, ncols);
        R = qr_top.template triangularView<Eigen::Upper>();
    }

    return std::make_tuple(Q, R);
};

/**
 * @brief Singular Value Decomposition
 * @param X ndarray of shape (nrows, ncols),
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
    int options = full_matrix ? Eigen::ComputeFullU | Eigen::ComputeFullV 
                              : Eigen::ComputeThinU | Eigen::ComputeThinV;

    // control the switching size, default is 16
    // For small matrice (<16), it is thus preferable to directly use JacobiSVD. 
    // For larger matrice, BDCSVD
    if (num_features < 16) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(x, options);
        U = svd.matrixU();
        s = svd.singularValues();
        V = svd.matrixV();
    }
    else {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(x, options);
        U = svd.matrixU();
        s = svd.singularValues();
        V = svd.matrixV();
    }
    return std::make_tuple(U, s, V);
};

/**
 * @brief Compute randomized svd
 * @param X ndarray of shape (nrows, ncols),
 *      Matrix to compute decomposition
 * @param num_components std::size_t
 *      Number of singular values and vectors to extract.
 * @param num_iters std::size_t 
 *      Number of power iterations
 * @param power_iter_nomalizer string
 *      Whether the power iterations are normalized with step-by-step
 *      it has 3 value: 'QR', 'LU', 'None'
 * @return a tuple contains U matrix, s vector and Vt matrix.
*/
template<typename MatType, typename VecType, typename IdxVecType>
std::tuple<MatType, VecType, MatType> randomized_svd(const MatType& X, 
    std::size_t num_components = 3, 
    std::size_t num_oversamples = 10,
    std::size_t num_iters = 4, 
    std::string power_iter_normalizer = "LU", 
    bool flip_sign = true) {
    
    std::size_t num_samples = X.rows(), num_features = X.cols();
    std::size_t num_random = num_components + num_oversamples;

    if (num_random > std::min(num_samples, num_features)) {
        std::ostringstream err_msg;
        err_msg << "Randomized_svd requires k + p <= min(M, N) " 
                << "k: num_components, p: num_oversamples." 
                << std::endl;
        throw std::invalid_argument(err_msg.str());
    }

    // init a random matrix Q
    MatType Q = random::randn<MatType>(num_features, num_random);
    for (std::size_t i = 0; i < num_iters; ++i) {

        if (power_iter_normalizer == "None") {
            Q = (X * Q).eval();
            Q = (X.transpose() * Q).eval();
        }
        else if (power_iter_normalizer == "LU") {
            MatType L, U;
            auto XQ = (X * Q).eval();
            std::tie(Q, L, U) = math::lu<MatType>(XQ, true);
            auto Xt_Q = (X.transpose() * Q).eval();
            std::tie(Q, L, U) = math::lu<MatType>(Xt_Q, true);
        }
        else if (power_iter_normalizer == "QR") {
            MatType R;
            auto XQ = (X * Q).eval();
            std::tie(Q, R) = math::qr<MatType>(XQ, false);
            auto Xt_Q = (X.transpose() * Q).eval();
            std::tie(Q, R) = math::qr<MatType>(Xt_Q, false);
        }
        else {
            std::ostringstream err_msg;
            err_msg << "power_iter_normalizer: {LU, QR, None}, default is LU, "
                    << "but got an unknown option: " << power_iter_normalizer
                    << std::endl;
            throw std::invalid_argument(err_msg.str());
        }
    }

    MatType R;
    std::tie(Q, R) = math::qr<MatType>(X * Q, false);

    // project X to the (k + p) dimensional space
    MatType B = Q.transpose() * X;
    // compute the SVD on the thin matrix: (k + p) wide
    MatType Uhat, Vt;
    VecType s;
    std::tie(Uhat, s, Vt) = math::exact_svd<MatType, VecType>(B, true);
    Vt = Vt.transpose().eval();
    B.resize(0, 0);

    MatType U = Q * Uhat;

    if (flip_sign) {
        std::tie(U, Vt) = math::svd_flip<MatType, VecType, IdxVecType>(U, Vt);
    }

    return std::make_tuple(
        U.leftCols(num_components),
        s.topRows(num_components),
        Vt.topRows(num_components));
}

/**
 * @brief Compute the (Moore-Penrose) pseudo-inverse of a matrix. 
 *      Calculate the generalized inverse of a matrix using its 
 *      singular-value decomposition (SVD) and including all 
 *      large singular values.
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

    std::size_t nrows = x.rows(), ncols = x.cols();
    MatType s_inv(ncols, nrows);
    s_inv.setZero();

    for(std::size_t i = 0; i < s.size(); i++) {
        if (s(i) > tol) {
            s_inv(i, i) = static_cast<DataType>(1) / s(i);
        }
        else {
            s_inv(i, i) = static_cast<DataType>(0);
        }
    }
    MatType pv = Vt * s_inv * U.transpose();
    return pv;
}

/**
 * @brief Compute log(det(A)) for A symmetric.
 * 
 * @param x ndarray of shape (nrows, ncols)
 *      input array, has to be a SQUARE 2d array
 * @return -Inf if det(A) is non positive or is not defined.
*/
template <typename MatType, 
    typename DataType = typename MatType::value_type>
DataType logdet(const MatType& x) {
    DataType log_abs_det = 0;
    DataType sign = 1;
    Eigen::PartialPivLU<MatType> lu_decomposition(x);
    MatType LU = lu_decomposition.matrixLU();
    sign = lu_decomposition.permutationP().determinant(); 
    auto diag = LU.diagonal().array().eval();
    auto abs_diag = diag.cwiseAbs().eval();
    log_abs_det += abs_diag.log().sum();
    sign *= (diag / abs_diag).prod();

    return sign < 0 ? (-ConstType<DataType>::infinity()) : log_abs_det;
}

/**
 * @brief Compute the log of the sum of exponentials of input elements.
 *      logsumexp(x_1, x_2, ..., x_n) 
 *      = log(sum(exp(x_i)))
 *      = log(sum(exp(x_i - c) * exp(c)))
 *      = log(exp(c) * sum(exp(x_i - c)))
 *      = log(exp(c)) + log(sum(exp(x_i - c)))
 *      = c + log(sum(exp(x_i - c)))
 * @param x ndarray input data
 * @param axis, int
 *      Axis or axes over which the sum is taken.
*/
template<typename MatType, typename VecType>
VecType logsumexp(const MatType& x, int axis){
    std::size_t nrows = x.rows(), ncols = x.cols();
    if (axis == 1) {
        VecType c = x.rowwise().maxCoeff();
        MatType tmp = c.rowwise().replicate(ncols);
        VecType log_sum_exp = (x - tmp).array().exp().rowwise().sum().log();
        return log_sum_exp + c;
    }
    else if (axis == 0) {
        VecType c = x.colwise().maxCoeff();
        MatType tmp = c.transpose().colwise().replicate(nrows);
        VecType log_sum_exp = (x - tmp).array().exp().colwise().sum().log();
        return log_sum_exp + c;
    }
    else if (axis == -1) {
        auto c = x.maxCoeff();
        auto log_sum_exp_val = std::log((x.array() - c).exp().sum()) + c;
        VecType log_sum_exp(1);
        log_sum_exp(0, 0) = log_sum_exp_val;
        return log_sum_exp;
    }
};

/**
 * @brief Cholesky decomposition
 *      Return the Cholesky decomposition, L * L.H, of the square matrix a, 
 *      where L is lower-triangular and .H is the conjugate transpose operator.
 *      x must be Hermitian (symmetric if real-valued) and positive-definite.
 * @param x ndarray input data
*/
template<typename MatType>
MatType cholesky(const MatType& x, bool lower = true) {
    Eigen::LLT<MatType> llt_decomposition(x);

    if (llt_decomposition.info() != Eigen::Success) {
        std::ostringstream err_msg;
        err_msg << "Cholesky decomposition was not successful: " 
                << llt_decomposition.info() 
                << ". Filling lower-triangular output with NaNs." << std::endl;
        throw std::runtime_error(err_msg.str());
    }

    if (lower) {
        return llt_decomposition.matrixL();
    }
    else {
        return llt_decomposition.matrixU();
    }
};

/**
 * Compute the eigenvalues and right eigenvectors of a square array
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
std::tuple<MatType, MatType> eig(const MatType& x) {
    std::size_t num_rows = x.rows();
    if (num_rows == 0) {
        // If X is an empty matrix (0 rows, 0 col), X * X' == X.
        // Therefore, we return X.
        std::ostringstream err_msg;
        err_msg << "Got an empty input matrix." << std::endl;
        throw std::invalid_argument(err_msg.str());
    } 

    Eigen::EigenSolver<MatType> eigs(x);
    if (eigs.info() != Eigen::Success) {
        std::ostringstream err_msg;
        err_msg << "Eigen decomposition was not successful"
                << "The input might not be valid."
                << std::endl;
        throw std::runtime_error(err_msg.str());
    }

    MatType vals, vecs;
    vals = eigs.eigenvalues().template cast<DataType>();
    vecs = eigs.eigenvectors();

    return std::make_tuple(vals, vecs);
}




}
}
#endif /*CORE_MATH_LINALG_HPP*/

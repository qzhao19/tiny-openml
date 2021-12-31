#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"

namespace math {

/**
 * Sign correction to ensure deterministic output from SVD. 
 * Adjusts the columns of u and the rows of v such that the loadings in the
 * columns in u that are largest in absolute value are always positive.
 * 
 * @param U ndarray u and v are the output of `svd` with matching inner
 * dimensions so one can compute `arma::dot(u * s, v)`
 * 
 * @param u_based_decision Bool If True, use the columns of u as the basis for sign flipping. 
 * Otherwise, use the rows of v. The choice of which variable to base the 
 * decision on is generally algorithm dependent.
 * 
 * @return u_adjusted, v_adjusted : arrays with the same dimensions as the input.
*/
template<typename MatType>
std::tuple<MatType, MatType> svd_flip(const MatType& U, 
    const MatType &Vt, 
    bool u_based_decision = true) {
    
    MatType U_, Vt_;
    if (u_based_decision) {
        arma::urowvec  max_abs_idx = arma::index_max(arma::abs(U), 0); 
        arma::rowvec max_abs_cols(max_abs_idx.n_elem);
        for(std::size_t j = 0; j < max_abs_idx.n_elem; j++) {
            std::size_t i = max_abs_idx(j);
            max_abs_cols(j) = U(i, j);
        }

        arma::rowvec signs = arma::sign(max_abs_cols);
        U_ = U % arma::repmat(signs, U.n_rows, 1);
        arma::vec signs_ = arma::conv_to<arma::vec>::from(signs);
        Vt_ = Vt % arma::repmat(signs_, 1, Vt.n_cols);
    }
    else {
        arma::uvec max_abs_idx = arma::index_max(arma::abs(Vt), 1); 
        arma::rowvec max_abs_rows(max_abs_idx.n_elem);

        for (std::size_t i = 0; i < max_abs_rows.n_elem; i++) {
            std::size_t j = max_abs_idx(i);
            max_abs_rows(i) = Vt(i, j);
        }
        arma::rowvec signs = arma::sign(max_abs_rows);
        U_ = U % arma::repmat(signs, U.n_rows, 1);
        arma::vec signs_ = arma::conv_to<arma::vec>::from(signs);
        Vt_ = Vt % arma::repmat(signs_, 1, Vt.n_cols);
    }

    return std::make_tuple(U_, Vt_);
};

/**
 * Compute log(det(A)) for A symmetric.
 * Equivalent to : arma::log(arma::det(X)) but more robust. 
 * It returns -Inf if det(A) is non positive or is not defined.
 * 
 * @param X Input matrix 
*/
template<typename MatType>
double logdet(const MatType& X) {
    double val;
    double sign;

    arma::log_det(val, sign, X);

    if (!(sign > 0)) {
        return std::numeric_limits<double>::min();
    }
    return val;
};

/**
 * flatten a vector of vector to a one dimension vector
*/
template<typename Type, typename = typename Type::value_type>
Type flatten(const std::vector<Type>& v) {
    return std::accumulate(v.begin(), v.end(), Type{}, [](auto& dest, auto& src) {
        dest.insert(dest.end(), src.begin(), src.end());
        return dest;
    });
}

/**
 * Calculate centered matrix, where cerntering is done by substracting the mean 
 * over the colnum from each col of the matrix.
 * 
 * @param X Input matrix
 * @param X_centered Matrix to write centered output
*/
template<typename MatType>
void center(const MatType& X, MatType& X_centered) {
    arma::rowvec x_mean = arma::mean(X, 0);
    X_centered = X - arma::repmat(x_mean, X.n_rows, 1);
};

};
#endif

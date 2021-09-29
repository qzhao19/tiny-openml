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
void svd_flip(MatType& U, MatType& V, bool u_based_decision = true) {

    

};


/**
 * Compute log(det(A)) for A symmetric.
 * Equivalent to : arma::log(arma::det(X)) but more robust. 
 * It returns -Inf if det(A) is non positive or is not defined.
 * 
 * @param X Input matrix 
*/
template<typename MatType>
double logdet(MatType& X) {

    double val;
    double sign;

    arma::log_det(val, sign, X);

    if (sign < 0) {
        return -arma::datum::inf;
    }
    return val;
};




};
#endif

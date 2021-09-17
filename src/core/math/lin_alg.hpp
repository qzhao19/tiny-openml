#ifndef CORE_MATH_LIN_ALG_HPP
#define CORE_MATH_LIN_ALG_HPP
#include "../../prereqs.hpp"

namespace math {

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

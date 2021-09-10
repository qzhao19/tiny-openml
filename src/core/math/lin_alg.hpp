#ifndef CORE_MATH_LIN_ALG_HPP
#define CORE_MATH_LIN_ALG_HPP
#include "../../prereqs.hpp"

namespace math {

/**
 * Calculate centered matrix, where cerntering is done by substracting the mean 
 * over the colnum from each col of the matrix.
 * 
 * @param X Input matrix
 * @param centered_X Matrix to write centered output
*/
template<typename MatType>
void center(const MatType& X, MatType& centered_X) {

    arma::vec x_mean = arma::mean(X, 0);

    centered_X = X - arma::repmat(x_mean, X.n_rows, 1);

};


};
#endif

#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

// template<typename DataType>
// double sigmoid(const DataType z) {
//     return DataType(1.0) / (DataType(1.0) + exp(-z));
// }


/**
 * compute the matrix sigmoid value
 *      s(z) = 1 / (1 + exp(-z))
 * @param mat ndarray of shape [num_rows, num_cols]
 * @return sigmoid matrix 
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType sigmoid(const MatType& mat) {
    return (static_cast<DataType>(1) / 
        (static_cast<DataType>(1) + (-mat.array()).exp())).matrix();
}

/**
 * compute the data variance, if input data is a vector,
 * return is a scalar, if input data is a matrtix, return 
 * is the covariamce matrix of ndarray.
 * 
 * @param mat input data of type vector or matrix 
 * @return scalar or 2darray
*/
template<typename AnyType>
AnyType var(const AnyType& mat) {
    AnyType centered = mat.rowwise() - mat.colwise().mean();
    AnyType cov = (centered.adjoint() * centered) / static_cast<double>(mat.rows() - 1);
    return cov;
};


/**
 * Calculate centered matrix, where cerntering is done by substracting the mean 
 * over the colnum from each col of the matrix.
 * 
 * @param mat Input matrix
 * @param center_mat Matrix to write centered output
*/
template<typename MatType>
void center(const MatType& mat, MatType& center_mat) {
    std::size_t num_samples = mat.rows();
    center_mat = repeat<MatType>(mat.colwise().mean(), num_samples, 0);
};


}
}

#endif

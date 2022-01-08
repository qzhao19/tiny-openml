#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

/**
 * Stack arrays in sequence horizontally (column wise).
*/
template<typename MatType>
MatType hstack(const MatType& mat1, const MatType& mat2) {
    assert(mat1.rows() == mat2.rows() && "hstack with mismatching number of rows");
    std::size_t n_rows = mat1.rows(), n_cols1 = mat1.cols(), n_cols2 = mat2.cols();
    MatType retmat(n_rows, (n_cols1 + n_cols2));
    retmat << mat1, mat2;
    return retmat;
};

/**
 * Stack arrays in sequence vertically (row wise). This is equivalent to concatenation 
 * along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N).
 * 
 * @param mat1_mat2 ndarray of shape (n_rows, n_cols) The arrays must have the same shape along all 
 * 
 * @return stacke dndarray
*/
template<typename MatType>
MatType vstack(const MatType& mat1, const MatType& mat2) {
    assert(mat1.cols() == mat2.cols() && "vstack with mismatching number of columns");
    std::size_t n_cols = mat1.cols(), n_rows1 = mat1.rows(), n_rows2 = mat2.rows();
    MatType retmat((n_rows1 + n_rows2), n_cols);
    retmat << mat1, 
              mat2;
    return retmat;
};

/**
 * flatten a vector of vector to a one dimension vector
*/
template<typename DataType, typename = typename DataType::value_type>
DataType flatten(const std::vector<DataType>& v) {
    return std::accumulate(v.begin(), v.end(), DataType{}, [](auto& dest, auto& src) {
        dest.insert(dest.end(), src.begin(), src.end());
        return dest;
    });
};

} // namespace math
} // namespace openml

#endif /*CORE_MATH_EXTMATH_HPP*/

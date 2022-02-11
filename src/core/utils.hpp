#ifndef CORE_UTILS_HPP
#define CORE_UTILS_HPP
#include "../prereqs.hpp"

namespace openml {
namespace utils {

/**
 *find the maximum value in a std::map and return the corresponding std::pair
*/
template <class Container>
auto max_element(Container const &x)
    -> typename Container::value_type {
    
    using value_t = typename Container::value_type;
    const auto compare = [](value_t const &p1, value_t const &p2)
    {
        return p1.second < p2.second;
    };
    return *std::max_element(x.begin(), x.end(), compare);
};


/**
 * Stack arrays in sequence horizontally (column wise).
 * 
 * @param mat1_mat2 ndarray of shape (num_rows, num_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType hstack(const MatType& mat1, const MatType& mat2) {
    assert(mat1.rows() == mat2.rows() && "hstack with mismatching number of rows");
    std::size_t num_rows = mat1.rows(), num_cols1 = mat1.cols(), num_cols2 = mat2.cols();
    MatType retval(num_rows, (num_cols1 + num_cols2));
    retval << mat1, mat2;
    return retval;
};

/**
 * Stack arrays in sequence vertically (row wise). This is equivalent to concatenation 
 * along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N).
 * 
 * @param mat1_mat2 ndarray of shape (num_rows, num_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType vstack(const MatType& mat1, const MatType& mat2) {
    assert(mat1.cols() == mat2.cols() && "vstack with mismatching number of columns");
    std::size_t num_cols = mat1.cols(), num_rows1 = mat1.rows(), num_rows2 = mat2.rows();
    MatType retval((num_rows1 + num_rows2), num_cols);
    retval << mat1, 
              mat2;
    return retval;
};

/**
 * flatten a 2D vector of vector to a one dimension vector
*/
template<typename DataType, typename = typename DataType::value_type>
DataType flatten(const std::vector<DataType>& v) {
    return std::accumulate(v.begin(), v.end(), DataType{}, [](auto& dest, auto& src) {
        dest.insert(dest.end(), src.begin(), src.end());
        return dest;
    });
};

/**
 * flatten a matrix of 2d to a vector
 * 
 * @param mat Eigen matrix type ndarray 
 * @return one dim vector of Eigrn type
*/
template<typename MatType, typename VecType>
VecType flatten(const MatType& mat) {
    std::size_t num_rows = mat.rows(), num_cols = mat.cols();
    MatType trans_mat = mat.transpose();
    VecType flatten_vec(Eigen::Map<VecType>(trans_mat.data(), num_rows * num_cols));
    return flatten_vec;
};

/**
 * Repeat elements of an matrix.
 * 
 * @param mat Input array.
 * @param repeats int. The number of repetitions for each element. 
 * @param axis int. The axis along which to repeat values. 
 * 
 * @return  Output array which has the same shape as a, except along the given axis.
*/
template<typename MatType>
MatType repeat(const MatType& mat, 
    int repeats, int axis) {
    
    MatType retval;
    if (axis == 0) {
        retval = mat.colwise().replicate(repeats);
    }
    else if (axis == 1) {
        retval = mat.rowwise().replicate(repeats);
    }
    return retval;
};

/**
 * Returns the indices of the maximum values along an axis.
 *
 *  @param mat 2darray of input data3
 * @param axis int, the given specified axis.
 * 
 * @return the indices of the maximum values along an axis
*/
template<typename MatType, typename IndexType>
IndexType argmax(const MatType& mat, int axis = 0) {
    if (axis == 1) {
        std::size_t num_rows = mat.rows();
        IndexType argmax{num_rows};
        for (std::size_t i = 0; i < num_rows; i++) {
            mat.row(i).maxCoeff(&argmax[i]);
        }
        return argmax;
    }
    else if (axis == 0) {
        std::size_t num_cols = mat.cols();
        IndexType argmax{num_cols};
        for (std::size_t j = 0; j < num_cols; j++) {
            mat.col(j).maxCoeff(&argmax[j]);
        }
        return argmax;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};





}
}
#endif /**/

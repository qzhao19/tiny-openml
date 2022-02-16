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
 * @param x1_x2 ndarray of shape (num_rows, num_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType hstack(const MatType& x1, const MatType& x2) {
    assert(x1.rows() == x2.rows() && "hstack with mismatching number of rows");
    std::size_t num_rows = x1.rows(), num_cols1 = x1.cols(), num_cols2 = x2.cols();
    MatType retval(num_rows, (num_cols1 + num_cols2));
    retval << x1, x2;
    return retval;
};

/**
 * Stack arrays in sequence vertically (row wise). This is equivalent to concatenation 
 * along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N).
 * 
 * @param x1_x2 ndarray of shape (num_rows, num_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType vstack(const MatType& x1, const MatType& x2) {
    assert(x1.cols() == x2.cols() && "vstack with mismatching number of columns");
    std::size_t num_cols = x1.cols(), num_rows1 = x1.rows(), num_rows2 = x2.rows();
    MatType retval((num_rows1 + num_rows2), num_cols);
    retval << x1, 
              x2;
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
 * @param x Eigen matrix type ndarray 
 * @return one dim vector of Eigrn type
*/
template<typename MatType, typename VecType>
VecType flatten(const MatType& x) {
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    MatType trans_x = x.transpose();
    VecType flatten_vec(Eigen::Map<VecType>(trans_x.data(), num_rows * num_cols));
    return flatten_vec;
};


/**
 * Repeat elements of an matrix.
 * 
 * @param x Input array.
 * @param repeats int. The number of repetitions for each element. 
 * @param axis int. The axis along which to repeat values. 
 * 
 * @return  Output array which has the same shape as a, except along the given axis.
*/
template<typename MatType>
MatType repeat(const MatType& x, 
    int repeats, int axis) {
    
    MatType retval;
    if (axis == 0) {
        retval = x.colwise().replicate(repeats);
    }
    else if (axis == 1) {
        retval = x.rowwise().replicate(repeats);
    }
    return retval;
};


/**
 * Returns the indices of the maximum values along an axis.
 *
 * @param x 2darray of input data
 * @param axis int, the given specified axis.
 * 
 * @return the indices of the maximum values along an axis
 *      if axis = 0, return index vector of max element along horizontal
 *      if axis = 1, retunn index vector of max element along vertical
 *      if axis = -1, return the max index scalar element of 2d 
*/
template<typename MatType, typename VecType, typename IdxType>
IdxType argmax(const MatType& x, int axis = 0) {
    if (axis == 1) {
        std::size_t num_rows = x.rows();
        IdxType max_index{num_rows};
        for (std::size_t i = 0; i < num_rows; i++) {
            x.row(i).maxCoeff(&max_index[i]);
        }
        return max_index;
    }
    else if (axis == 0) {
        std::size_t num_cols = x.cols();
        IdxType max_index{num_cols};
        for (std::size_t j = 0; j < num_cols; j++) {
            x.col(j).maxCoeff(&max_index[j]);
        }
        return max_index;
    }
    else if (axis == -1) {
        // flayyen a 2d matrix into 1d verctor
        VecType flatten_vec;
        flatten_vec = flatten<MatType, VecType>(x);
        
        // get the max index of flattened vector
        IdxType max_index{1};
        flatten_vec.maxCoeff(&max_index[0]);

        return max_index;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};

/**
 * Returns the indices of the minimum values along an axis.
 * @param x 2darray of input data
 * @param axis int, the given specified axis.
 * 
 * @return the indices of the minimum values along an axis
 *      if axis = 0, return index vector of min element along horizontal
 *      if axis = 1, retunn index vector of min element along vertical
 *      if axis = -1, return the min index scalar element of 2d 
*/
template<typename MatType, typename VecType, typename IdxType>
IdxType argmin(const MatType& x, int axis = 0) {

    if (axis == 1) {
        std::size_t num_rows = x.rows();
        IdxType min_index{num_rows};
        for (std::size_t i = 0; i < num_rows; i++) {
            x.row(i).minCoeff(&min_index[i]);
        }
        return min_index;
    }
    else if (axis == 0) {
        std::size_t num_cols = x.cols();
        IdxType min_index{num_cols};
        for (std::size_t j = 0; j < num_cols; j++) {
            x.col(j).minCoeff(&min_index[j]);
        }
        return min_index;
    }
    else if (axis == -1) {
        VecType flatten_vec;
        flatten_vec = flatten<MatType, VecType>(x);
        
        // get the max index of flattened vector
        IdxType min_index{1};
        flatten_vec.minCoeff(&min_index[0]);

        return min_index;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};






}
}
#endif /**/

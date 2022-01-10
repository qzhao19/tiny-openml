#ifndef CORE_UTILITY_UTILS_HPP
#define CORE_UTILITY_UTILS_HPP
#include "../../prereqs.hpp"

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
 * @param X1_X2 ndarray of shape (n_rows, n_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType hstack(const MatType& X1, const MatType& X2) {
    assert(X1.rows() == X2.rows() && "hstack with mismatching number of rows");
    std::size_t n_rows = X1.rows(), n_cols1 = X1.cols(), n_cols2 = X2.cols();
    MatType retval(n_rows, (n_cols1 + n_cols2));
    retval << X1, X2;
    return retval;
};

/**
 * Stack arrays in sequence vertically (row wise). This is equivalent to concatenation 
 * along the first axis after 1-D arrays of shape (N,) have been reshaped to (1,N).
 * 
 * @param X1_X2 ndarray of shape (n_rows, n_cols) The arrays must have the same shape along all 
 * @return stacke dndarray
*/
template<typename MatType>
MatType vstack(const MatType& X1, const MatType& X2) {
    assert(X1.cols() == X2.cols() && "vstack with mismatching number of columns");
    std::size_t n_cols = X1.cols(), n_rows1 = X1.rows(), n_rows2 = X2.rows();
    MatType retval((n_rows1 + n_rows2), n_cols);
    retval << X1, 
              X2;
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


template<typename MatType>
MatType repeat(const MatType& array, 
    int repeats, 
    int axis = 0) {
    
    if (axis = 0) {
        return array.colwise().replicate(repeats, 1);
    }
    else if (axis = 1) {
        return array.colwise().replicate(1, repeats);
    }
};



}
}
#endif

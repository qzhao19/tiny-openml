#ifndef CORE_MATH_OPS_HPP
#define CORE_MATH_OPS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace math {

/**
 * Estimate a covariance matrix, given data.
 * Covariance indicates the level to which two variables vary together. 
 * If we examine N-dimensional samples, X = [x1, x2, .. x_n]_T , 
 * then the covariance matrix element C_ij is the covariance of x_i and x_j. 
 * The element C_ii is the variance of x_i.
 * 
 * @param x input data of type vector or matrix 
 * @return scalar or 2darray
*/
template<typename AnyType>
AnyType cov(const AnyType& x) {
    AnyType centered = x.rowwise() - x.colwise().mean();
    AnyType cov = (centered.adjoint() * centered) / static_cast<double>(x.rows() - 1);
    return cov;
};

/**
 * Axis or axes along which the variance is computed, 
 * if input data is a vector, return is a scalar, 
 * if input data is a matrtix, return is the covariamce 
 * matrix of ndarray.
 * 
 * The default is to compute the variance of the flattened array.
 * 
 * @param x input data of type vector or matrix 
 * @param axis int. default -1 The axis along which to calculate variance. 
 * @return scalar or 2darray
*/
template<typename MatType>
MatType var(const MatType& x, int axis = -1) {
    MatType copy_x = x;
    // Var(X)=E[X^2]-(E[X])^2
    if (axis == 0) {
        auto colmean = copy_x.colwise().mean();                                      
        auto x_squared_mean = copy_x.array().square().colwise().mean();
        auto colvar = x_squared_mean.array() - colmean.array().square();

        return colvar; 
    }
    else if (axis == 1) {
        auto rowmean = copy_x.rowwise().mean();
        auto x_squared_mean = copy_x.array().square().rowwise().mean();
        auto rowvar = x_squared_mean.array() - rowmean.array().square();

        return rowvar;
    }
    else if (axis == -1) {
        auto mean = copy_x.mean();
        auto x_squared_mean = copy_x.array().square().mean();
        MatType var(1, 1);
        var(0, 0) = x_squared_mean - std::pow(mean, 2);
        return var;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};

/**
 * Sum of array elements over a given axis.
 * @param x input data of type vector or matrix 
 * @param axis int. Axis or axes along which a sum is performed. 
 *      The default is -1, 
*/
template<typename MatType>
MatType sum(const MatType& x, int axis = -1) {
    MatType copy_x = x;
    if (axis == 0) {
        return copy_x.colwise().sum();
    }
    else if (axis == 1) {
        return copy_x.rowwise().sum();
    }
    else if (axis == -1) {
        MatType sum(1, 1);
        sum(0, 0) = copy_x.sum();
        return sum;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};

/**
 * Mean of array elements over a given axis.
 * @param x input data of type vector or matrix 
 * @param axis int. Axis or axes along which a mean is performed. 
 *      The default is -1, 
*/
template<typename MatType>
MatType mean(const MatType& x, int axis = -1) {
    MatType copy_x = x;
    if (axis == 0) {
        return copy_x.colwise().mean();
    }
    else if (axis == 1) {
        return copy_x.rowwise().mean();;
    }
    else if (axis == -1) {
        MatType mean(1, 1);
        mean(0, 0) = copy_x.mean();
        return mean;
    }
    else {
        throw std::invalid_argument("Got an invalid axis value.");
    }
};

/**
 * Compute the median of a matrix or vector along the specified axis.
 * @param x ndarray of shape (num_samples, num_features), Input array
 * @param axis int optional, Axis or axes along which the medians are computed.
 *      the default value is 0.
*/
template <typename MatType, 
    typename VecType, 
    typename DataType = typename VecType::value_type>
VecType median(const MatType& x, int axis = 0) {
    std::size_t nrows = x.rows(), ncols = x.cols();

    MatType X = x;
    std::size_t index;
    DataType value;
    if (axis == 0) {
        VecType col_med(ncols);
        index = nrows / 2;
        for (std::size_t j = 0; j < ncols; ++j) {
            VecType col = X.col(j);
            std::nth_element(col.data(), col.data() + index, col.data() + nrows);
            value = col(index);
            if (nrows%2 == 0) {
                std::nth_element(col.data(), col.data() + index - 1, col.data() + nrows);
                col_med(j) = (value + col(index - 1)) / 2;
            }
            else {
                col_med(j) = value;
            }
        }
        return col_med;
    }
    else if (axis == 1) {
        VecType row_med(nrows);
        index = ncols / 2;
        for (std::size_t i = 0; i < nrows; ++i) {
            VecType row = X.row(i).transpose();
            std::nth_element(row.data(), row.data() + index, row.data() + ncols);
            value = row(index);
            if (ncols%2 == 0) {
                std::nth_element(row.data(), row.data() + index - 1, row.data() + ncols);
                row_med(i) = (value + row(index - 1)) / 2;
            }
            else {
                row_med(i) = value;
            }
        }
        return row_med;
    }
    else if (axis == -1) {
        MatType trans_X = X.transpose();
        std::size_t num_elems = nrows * ncols;
        VecType flatten_X(Eigen::Map<VecType>(trans_X.data(), num_elems));
        VecType med(1);
        index = num_elems / 2;
        std::nth_element(flatten_X.data(), flatten_X.data() + index, flatten_X.data() + num_elems);
        value = flatten_X(index);
        if (num_elems%2 == 0) {
            std::nth_element(flatten_X.data(), 
                flatten_X.data() + index - 1, 
                flatten_X.data() + num_elems
            );
            med(0) = (value + flatten_X(index - 1)) / 2;
        }
        else {
            med(0) = value;
        }
        return med;
    }
};

/**
 * Calculate centered matrix, where cerntering is done by substracting the mean 
 * over the colnum from each col of the matrix.
 * 
 * @param x Input matrix
 * @return centered matrix to write centered output
*/
template<typename MatType>
MatType center(const MatType& x) {
    return x.rowwise() - x.colwise().mean();
};

/**
 * Calculate the absolute value element-wise.
 * @param x ndarray of input data
 * @return An ndarray containing the absolute value of each element in x.
*/
template<typename AnyType>
AnyType abs(const AnyType& x) {
    return x.array().abs();
};

/**
 * First array elements raised to powers from second param, element wise
 * Negative values raised to a non-integral value will return nan.
 * @param x input data of ndarray type
 * @param exponent double type 
 * 
 * @return The bases in x1 raised to the exponents
*/
template<typename AnyType>
AnyType power(const AnyType& x, double exponents) {
    return x.array().pow(exponents);
};

/**
 * Returns an element-wise indication of the sign of a number.
 * The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
 * @param x ndarray of input values
 * @return The sign of x.
*/
template<typename AnyType>
AnyType sign(const AnyType& x) {
    return x.array().sign();
};

/**
 * Matrix norm
*/
template<typename MatType, typename VecType>
VecType norm2(const MatType& x, int axis = 0) {
    if (axis == 0) {
        return x.colwise().norm();
    }
    else if (axis == 1) {
        return x.rowwise().norm();
    }
};

/**
 * Return the cumulative sum of the elements along a given axis.
 * @param x ndarray of input data
 * @param axis int optional, Axis or axes along which the 
 *      cumulative sum of are computed.
*/
template<typename MatType, typename VecType>
MatType cumsum(const MatType& x, int axis = 0) {
    std::size_t nrows = x.rows(), ncols = x.cols();
    MatType cum_sum(nrows, ncols);
    if (axis == 0) {
        for(std::size_t j = 0; j < ncols; ++j) {
            VecType colsum(nrows);
            VecType col = x.col(j);
            std::partial_sum(col.begin(), col.end(), colsum.begin());
            cum_sum.col(j) = colsum;
        }
    }
    else if (axis == 1) {
        for(std::size_t i = 0; i < nrows; ++i) {
            VecType rowsum(ncols);
            VecType row = x.row(i);
            std::partial_sum(row.begin(), row.end(), rowsum.begin());
            cum_sum.row(i) = rowsum;
        }
    }
    else if (axis == -1) {
        MatType trans_x = x.transpose();
        VecType flatten_vec(Eigen::Map<VecType>(trans_x.data(), nrows * ncols));
        VecType sum(nrows * ncols);
        std::partial_sum(flatten_vec.begin(), flatten_vec.end(), sum.begin());
        return sum;
    }
    return cum_sum;
};

/**
 * Get the modal (most common) value in the passed array. 
 * @param x the array
 *      n-dimensional array of which to find mode(s).
 * @param axis int, default: 0
 *      the axis of the input along which to compute the statistic.
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
std::tuple<MatType, MatType> mode(const MatType& x, int axis = 0) {
    MatType mode, count;
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    if (axis == 0) {
        mode.resize(1, num_cols);
        count.resize(1, num_cols);
        for (std::size_t j = 0; j < num_cols; ++j) {
            std::unordered_map<DataType, std::size_t> lookups;
            std::size_t max_freqency = 0;
            DataType most_frequent_elem;

            for (const auto& elem : x.col(j)) {
                std::size_t frequency = ++lookups[elem];
                if (frequency > max_freqency) {
                    max_freqency = frequency;
                    most_frequent_elem = elem;
                }
            }
            if (max_freqency == 1) {
                most_frequent_elem = x.col(j).minCoeff();
            }
            mode(0, j) = most_frequent_elem;
            count(0, j) = max_freqency;
        }
    }
    else if (axis == 1) {
        mode.resize(num_rows, 1);
        count.resize(num_rows, 1);
        for (std::size_t i = 0; i < num_rows; ++i) {
            std::unordered_map<DataType, std::size_t> lookups;
            std::size_t max_freqency = 0;
            DataType most_frequent_elem;

            for (const auto& elem : x.row(i)) {
                std::size_t frequency = ++lookups[elem];
                if (frequency > max_freqency) {
                    max_freqency = frequency;
                    most_frequent_elem = elem;
                }
            }
            if (max_freqency == 1) {
                most_frequent_elem = x.row(i).minCoeff();
            }
            mode(i, 0) = most_frequent_elem;
            count(i, 0) = max_freqency;
        }
    }
    return std::make_tuple(mode, count);
};

}
}

#endif /*CORE_MATH_UNARY_OPS_HPP*/

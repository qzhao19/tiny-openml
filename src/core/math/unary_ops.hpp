#ifndef CORE_MATH_UNARY_OPS_HPP
#define CORE_MATH_UNARY_OPS_HPP
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
template<typename MatType, typename VecType>
VecType var(const MatType& x, int axis = -1) {
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    // Var(X)=E[X^2]-(E[X])^2
    if (axis == 0) {
        // compute means and element-wise square along to axis 1
        VecType col_mean(num_cols);
        col_mean = x.colwise().mean();
        VecType mean_x_squared(num_cols);
        mean_x_squared = x.array().square().colwise().mean().transpose();
        VecType col_var(num_cols);
        col_var = mean_x_squared - col_mean.array().square().matrix();

        return col_var;
    }
    else if (axis == 1) {
        VecType row_mean(num_rows);
        row_mean = x.rowwise().mean();
        VecType mean_x_squared(num_rows);
        mean_x_squared = x.array().square().rowwise().mean();
        VecType row_var(num_rows);
        row_var = mean_x_squared - row_mean.array().square().matrix();

        return row_var;
    }
    else if (axis == -1) {
        MatType trans_x = x.transpose();
        VecType flatten_x(Eigen::Map<VecType>(trans_x.data(), num_rows * num_cols));
        VecType mean(1);
        mean = flatten_x.colwise().mean();
        VecType mean_x_squared(1);
        mean_x_squared = flatten_x.array().square().colwise().mean();
        VecType var(1);
        var = mean_x_squared - mean.array().square().matrix();

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
template<typename MatType, 
    typename VecType>
VecType sum(const MatType& x, int axis = -1) {
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    if (axis == 0) {
        // compute means and element-wise square along to axis 1
        VecType col_sum(num_cols);
        col_sum = x.colwise().sum();
        return col_sum;
    }
    else if (axis == 1) {
        VecType row_sum(num_rows);
        row_sum = x.rowwise().sum();
        return row_sum;
    }
    else if (axis == -1) {
        MatType trans_x = x.transpose();
        VecType flatten_x(Eigen::Map<VecType>(trans_x.data(), num_rows * num_cols));
        VecType sum(1);
        sum = flatten_x.colwise().sum();
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
template<typename MatType, 
    typename VecType>
VecType mean(const MatType& x, int axis = -1) {
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    if (axis == 0) {
        // compute means and element-wise square along to axis 1
        VecType col_mean(num_cols);
        col_mean = x.colwise().mean();
        return col_mean;
    }
    else if (axis == 1) {
        VecType row_mean(num_rows);
        row_mean = x.rowwise().mean();
        return row_mean;
    }
    else if (axis == -1) {
        MatType trans_x = x.transpose();
        VecType flatten_x(Eigen::Map<VecType>(trans_x.data(), num_rows * num_cols));
        VecType mean(1);
        mean = flatten_x.colwise().mean();
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
    std::size_t num_rows = x.rows(), num_cols = x.cols();

    MatType X = x;
    std::size_t index;
    DataType value;
    if (axis == 0) {
        VecType col_med(num_cols);
        index = num_rows / 2;
        for (std::size_t j = 0; j < num_cols; ++j) {
            VecType col = X.col(j);
            std::nth_element(col.data(), col.data() + index, col.data() + num_rows);
            value = col(index);
            if (num_rows%2 == 0) {
                std::nth_element(col.data(), col.data() + index - 1, col.data() + num_rows);
                col_med(j) = (value + col(index - 1)) / 2;
            }
            else {
                col_med(j) = value;
            }
        }
        return col_med;
    }
    else if (axis == 1) {
        VecType row_med(num_rows);
        index = num_cols / 2;
        for (std::size_t i = 0; i < num_rows; ++i) {
            VecType row = X.row(i).transpose();
            std::nth_element(row.data(), row.data() + index, row.data() + num_cols);
            value = row(index);
            if (num_cols%2 == 0) {
                std::nth_element(row.data(), row.data() + index - 1, row.data() + num_cols);
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
        std::size_t num_elems = num_rows * num_cols;
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


}
}

#endif /*CORE_MATH_UNARY_OPS_HPP*/

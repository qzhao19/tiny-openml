#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace math {

/**
 * compute the matrix sigmoid value
 *      s(z) = 1 / (1 + exp(-z))
 * @param x ndarray of shape [num_rows, num_cols]
 * @return sigmoid matrix 
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType sigmoid(const MatType& x) {
    return (static_cast<DataType>(1) / 
        (static_cast<DataType>(1) + (-x.array()).exp()));
};

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
 * transform a vector to diagonal matrix 
 * @param x vector of shape (num_rows)
 *      input data
 * @return a diagonal matrix of shape (num_rows, num_rows)
*/
template<typename MatType, typename VecType>
MatType diagmat(const VecType& x) {
    std::size_t num_rows = x.rows();
    MatType diag_mat(num_rows, num_rows);
    diag_mat = x.asDiagonal();
    return diag_mat;
}

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
 * Sign correction to ensure deterministic output from SVD. 
 * Adjusts the columns of u and the rows of v such that the loadings in the
 * columns in u that are largest in absolute value are always positive.
 * 
 * @param U ndarray u and v are the output of `svd` with matching inner
 * dimensions so one can compute `Eigen::dot(u * s, v)`
 * 
 * @param u_based_decision Bool If True, use the columns of u as the basis for sign flipping. 
 * Otherwise, use the rows of v. The choice of which variable to base the 
 * decision on is generally algorithm dependent.
 * 
 * @return u_adjusted, v_adjusted : arrays with the same dimensions as the input.
*/
template<typename MatType, typename VecType, typename IdxType>
std::tuple<MatType, MatType> svd_flip(const MatType& U, 
    const MatType &Vt, 
    bool u_based_decision = true) {
    
    MatType U_, Vt_;

    if (u_based_decision) {
        // columns of u, rows of v
        MatType abs_U = abs<MatType>(U);
        IdxType max_abs_index = utils::argmax<MatType, VecType, IdxType>(abs_U, 0);

        std::size_t num_elems = max_abs_index.rows();
        VecType max_abs_cols(num_elems);

        for(std::size_t j = 0; j < num_elems; j++) {
            std::size_t i = max_abs_index(j);
            max_abs_cols(j) = U(i, j);
        }
        VecType signs = sign<VecType>(max_abs_cols);
        U_ = U.array().rowwise() * signs.transpose().array();        
        Vt_ = Vt.array().colwise() * signs.array();
    }
    else {
        // rows of v, columns of u
        MatType abs_Vt = abs<MatType>(Vt);
        IdxType max_abs_index = utils::argmax<MatType, VecType, IdxType>(abs_Vt, 1);

        std::size_t num_elems = max_abs_index.rows();
        VecType max_abs_rows(num_elems);

        for(std::size_t i = 0; i < num_elems; i++) {
            std::size_t j = max_abs_index(i);
            max_abs_rows(i) = Vt(i, j);
        }

        VecType signs = sign<VecType>(max_abs_rows);
        U_ = U.array().rowwise() * signs.transpose().array();        
        Vt_ = Vt.array().colwise() * signs.array();
    }
    return std::make_tuple(U_, Vt_);
};


/**
 * Compute the log of the sum of exponentials of input elements.
 * @param x ndarray input data
 * @param axis, int
 *      Axis or axes over which the sum is taken.
*/
template<typename MatType, typename VecType>
VecType logsumexp(const MatType& x, int axis){
    std::size_t num_rows = x.rows(), num_cols = x.cols();
    if (axis == 1) {
        VecType c = x.rowwise().maxCoeff();
        MatType repeated_c = utils::repeat<MatType>(c, num_cols, 1);
        VecType log_sum_exp = (x - repeated_c).array().exp().rowwise().sum().log();
        return log_sum_exp + c;
    }
    else if (axis == 0) {
        VecType c = x.colwise().maxCoeff();
        MatType repeated_c = utils::repeat<MatType>(c.transpose(), num_rows, 0);
        VecType log_sum_exp = (x - repeated_c).array().exp().colwise().sum().log();
        return log_sum_exp + c;
    }
    else if (axis == -1) {
        auto c = x.maxCoeff();
        auto log_sum_exp_val = std::log((x.array() - c).exp().sum()) + c;
        VecType log_sum_exp(1);
        log_sum_exp(0, 0) = log_sum_exp_val;
        return log_sum_exp;
    }
};

/**
 * compute the entropy of a vector
 * Ent(D) = -sum(P_k * log2(P_k))
 * 
 * @param x the vector of shape (num_rows, 1)
 *    input data to compute the entropy
 * @param sample_weight input sample weight matrix
 *    it can be an empty constructor of matrix
*/
template<typename VecType, 
    typename DataType = typename VecType::value_type>
double entropy(const VecType& x, 
    const VecType& weight = VecType()) {

    VecType w = weight;
    std::size_t num_rows = x.rows();
    if (w.size() == 0) {
        w.resize(num_rows);
        w.setOnes();
    }  

    if (w.size() != 0 && w.rows() != num_rows) {
        char buffer[200];
        std::snprintf(buffer, 200, 
            "Size of sample weights must be equal to x, but got (%ld)",
            w.rows());
        std::string err_msg = static_cast<std::string>(buffer);
        throw std::invalid_argument(err_msg);
    }

    double ent = 0.0;
    std::map<DataType, std::size_t> x_count_map;
    std::map<DataType, std::vector<DataType>> w_count_map;

    for (std::size_t i = 0; i < num_rows; ++i) {
        x_count_map[x(i)]++;
        w_count_map[x(i)].push_back(w(i));
    }

    for (auto x_count = x_count_map.begin(), w_count = w_count_map.begin();
        x_count != x_count_map.end(), w_count != w_count_map.end(); 
        ++x_count, ++w_count) {
        
        double sum = 0.0;
        std::size_t num_w_counts = w_count->second.size();
        for (std::size_t i = 0; i < num_w_counts; ++i) {
            sum += static_cast<double>(w_count->second[i]);
        }

        double mean = sum / static_cast<double>(num_w_counts);
        double p_i = 1.0 * static_cast<double>(x_count->second) * mean / static_cast<double>(num_rows);

        ent += (-p_i) * std::log2(p_i);
    }
    return ent;
};


template<typename VecType, 
    typename DataType = typename VecType::value_type>
double gini(const VecType& x, 
    const VecType& weight = VecType()) {

    VecType w = weight;
    std::size_t num_rows = x.rows();
    if (w.size() == 0) {
        w.resize(num_rows);
        w.setOnes();
    }  

    if (w.size() != 0 && w.rows() != num_rows) {
        char buffer[200];
        std::snprintf(buffer, 200, 
            "Size of sample weights must be equal to x, but got (%ld)",
            w.rows());
        std::string err_msg = static_cast<std::string>(buffer);
        throw std::invalid_argument(err_msg);
    }

    double g = 0.0;
    std::map<DataType, std::size_t> x_count_map;
    std::map<DataType, std::vector<DataType>> w_count_map;

    for (std::size_t i = 0; i < num_rows; ++i) {
        x_count_map[x(i)]++;
        w_count_map[x(i)].push_back(w(i));
    }

    for (auto x_count = x_count_map.begin(), w_count = w_count_map.begin();
        x_count != x_count_map.end(), w_count != w_count_map.end(); 
        ++x_count, ++w_count) {
        double sum = 0.0;
        std::size_t num_w_counts = w_count->second.size();
        for (std::size_t i = 0; i < num_w_counts; ++i) {
            sum += static_cast<double>(w_count->second[i]);
        }

        double mean = sum / static_cast<double>(num_w_counts);
        double p_i = 1.0 * static_cast<double>(x_count->second) * mean / static_cast<double>(num_rows);

        g += std::pow(p_i, 2.0);
    }
    return 1.0 - g;
};





}
}

#endif

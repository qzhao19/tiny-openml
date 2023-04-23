#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace math {

/**
 * compute the matrix sigmoid value
 *      s(z) = 1 / (1 + exp(-z))
 * exp(fmin(X, 0)) / (1 + exp(-abs(X)))
 * @param x ndarray of shape [nrows, ncols]
 * @return sigmoid matrix 
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType sigmoid(const MatType& x) {
    MatType new_x = common::fmin<MatType>(x, static_cast<DataType>(0));
    return new_x.array().exp() / 
        (static_cast<DataType>(1) + (-x.array().abs()).exp());
};

/**
 * transform a vector to diagonal matrix 
 * @param x vector of shape (nrows)
 *      input data
 * @return a diagonal matrix of shape (nrows, nrows)
*/
template<typename MatType, typename VecType>
MatType diagmat(const VecType& x) {
    std::size_t nrows = x.rows();
    MatType diag_mat(nrows, nrows);
    diag_mat = x.asDiagonal();
    return diag_mat;
}

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
template<typename MatType, typename VecType, typename IdxVecType>
std::tuple<MatType, MatType> svd_flip(const MatType& U, 
    const MatType &Vt, 
    bool u_based_decision = true) {
    
    MatType U_, Vt_;

    if (u_based_decision) {
        // columns of u, rows of v
        MatType abs_U = abs<MatType>(U);
        IdxVecType max_abs_index = common::argmax<MatType, VecType, IdxVecType>(abs_U, 0);

        std::size_t num_elems = max_abs_index.rows();
        VecType max_abs_cols(num_elems);

        for(std::size_t j = 0; j < num_elems; j++) {
            std::size_t i = max_abs_index(j);
            max_abs_cols(j) = U(i, j);
        }
        VecType signs = math::sign<VecType>(max_abs_cols);
        U_ = U.array().rowwise() * signs.transpose().array();        
        Vt_ = Vt.array().colwise() * signs.array();
    }
    else {
        // rows of v, columns of u
        MatType abs_Vt = abs<MatType>(Vt);
        IdxVecType max_abs_index = common::argmax<MatType, VecType, IdxVecType>(abs_Vt, 1);

        std::size_t num_elems = max_abs_index.rows();
        VecType max_abs_rows(num_elems);

        for(std::size_t i = 0; i < num_elems; i++) {
            std::size_t j = max_abs_index(i);
            max_abs_rows(i) = Vt(i, j);
        }

        VecType signs = math::sign<VecType>(max_abs_rows);
        U_ = U.array().rowwise() * signs.transpose().array();        
        Vt_ = Vt.array().colwise() * signs.array();
    }
    return std::make_tuple(U_, Vt_);
};

/**
 * compute the entropy of a vector
 * Ent(D) = -sum(P_k * log2(P_k))
 * 
 * @param x the vector of shape (nrows, 1)
 *    input data to compute the entropy
 * @param sample_weight input sample weight matrix
 *    it can be an empty constructor of matrix
*/
template<typename VecType, 
    typename DataType = typename VecType::value_type>
double entropy(const VecType& x, 
    const VecType& weight = VecType()) {

    VecType w = weight;
    std::size_t nrows = x.rows();
    if (w.size() == 0) {
        w.resize(nrows);
        w.setOnes();
    }  

    if (w.size() != 0 && w.rows() != nrows) {
        std::ostringstream err_msg;
        err_msg << "Size of sample weights must be equal to x, "
                << "but got unknown " << w.rows() << std::endl;
        throw std::invalid_argument(err_msg.str());
    }

    double ent = 0.0;
    std::map<DataType, std::size_t> x_count_map;
    std::map<DataType, std::vector<DataType>> w_count_map;

    for (std::size_t i = 0; i < nrows; ++i) {
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
        double p_i = 1.0 * static_cast<double>(x_count->second) * 
            mean / static_cast<double>(nrows);

        ent += (-p_i) * std::log2(p_i);
    }
    return ent;
};

/**
 * compute gini index
 * Gini(p) = 1 - sum(p_i), i = 1 : k
 * 
 * @param x the vector of shape (nrows, 1)
 *    input data to compute the entropy
 * @param sample_weight input sample weight matrix
 *    it can be an empty constructor of matrix
*/
template<typename VecType, 
    typename DataType = typename VecType::value_type>
double gini(const VecType& x, 
    const VecType& weight = VecType()) {
    VecType w = weight;
    std::size_t nrows = x.rows();
    if (w.size() == 0) {
        w.resize(nrows);
        w.setOnes();
    }  
    if (w.size() != 0 && w.rows() != nrows) {
        std::ostringstream err_msg;
        err_msg << "Size of sample weights must be equal to x, "
                << "but got unknown " << w.rows() << std::endl;
        throw std::invalid_argument(err_msg.str());
    }
    double g = 0.0;
    std::map<DataType, std::size_t> x_count_map;
    std::map<DataType, std::vector<DataType>> w_count_map;

    for (std::size_t i = 0; i < nrows; ++i) {
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
        double p_i = 1.0 * static_cast<double>(x_count->second) * 
            mean / static_cast<double>(nrows);
        g += std::pow(p_i, 2.0);
    }
    return 1.0 - g;
};

/**
 * Row-wise (squared) Euclidean norm of X
 * Equivalent to np.sqrt((X * X).sum(axis=1))
 * @param x ndarray of shape [rows, cols]
 *    The input array data
 * @param squared bool, default false
 *    squared norms
*/
template<typename MatType, typename VecType>
MatType row_norms(const MatType& x, bool squared = false) {

    MatType x_squared = x.array() * x.array();
    if (squared) {
        return x_squared.rowwise().sum();
    }
    else {
        MatType sum_x_squared;
        sum_x_squared = x_squared.rowwise().sum();
        return sum_x_squared.array().sqrt();
    }
};

/**
 * Compute the softmax function.
 * softmax(x) = exp(x)/sum(exp(x))
 * @param x ndarray of shape [rows, cols]
 *    The input array data
 * @param axis int. 
 *    The axis along which to compute values. 
*/
template<typename MatType>
MatType softmax(const MatType& x, int axis = 0) {
    std::size_t nrows = x.rows(), ncols = x.cols();
    if (nrows == 1 || ncols == 1) {
        throw std::invalid_argument("Input ndarray should be 2 dimension");
    }

    MatType c, div, retval;
    if (axis == 0) {
        c = x.colwise().maxCoeff().colwise().replicate(nrows);
    }
    else if (axis == 1) {
        c = x.rowwise().maxCoeff().rowwise().replicate(ncols);
    }
    else if (axis == -1) {
        auto maxval = x.maxCoeff();
        c.resize(nrows, ncols);
        c.fill(maxval);
    }
    MatType exp_x = (x - c).array().exp();

    if (axis == 0) {
        div = exp_x.colwise().sum().colwise().replicate(nrows);
    }
    else if (axis == 1) {
        div = exp_x.rowwise().sum().rowwise().replicate(ncols);
    }
    else if (axis == -1) {
        auto sum = exp_x.sum();
        div.resize(nrows, ncols);
        div.fill(sum);
    }

    retval = exp_x.array() / div.array();
    return retval;

};

}
}

#endif /*CORE_MATH_EXTMATH_HPP*/

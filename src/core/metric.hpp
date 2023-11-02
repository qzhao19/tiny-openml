#ifndef CORE_METRIC_HPP
#define CORE_METRIC_HPP
#include "../prereqs.hpp"
#include "../core.hpp"

using namespace openml;

namespace openml {
namespace metric {

/**
 * The mean_squared_error function computes mean square error
 * 
 *      mse = 1 / num_samples * sum(y_i - y_hat_i)**2
 */
template<typename VecType>
double mean_absolute_error(const VecType& y_true, 
    const VecType& y_pred) {
    
    std::size_t num_samples = y_true.rows();

    double sum_abs2 = (y_true - y_pred).array().abs2().sum();

    return sum_abs2 / static_cast<double>(num_samples);
};

/**
 * The explained_variance_score computes the explained variance regression score.
 * If y_hat is the estimated target output, y the corresponding (correct) target output, and  
 * var is Variance, the square of the standard deviation.
 * 
 *      explained_variance(y_hat, y) = 1 - var(y_hat - y) / var(y)
*/
template<typename VecType>
double explained_variance_score(const VecType& y_true, 
    const VecType& y_pred) {

    VecType y_diff = y_pred - y_true;
    VecType y_diff_var = math::var<VecType>(y_diff);
    VecType y_var = math::var<VecType>(y_true);
    
    auto retval = 1.0 -  y_diff_var.array() / y_var.array();
    return static_cast<double>(retval(0, 0));
};

/**
 * Compute the L2 euclidean distances between the vectors in X and Y.
*/
template <typename DataType>
double euclidean_distance(const std::vector<DataType>& a, const std::vector<DataType>& b) {
    std::vector<DataType> aux;
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(aux),
                   [](DataType x1, DataType x2) { return std::pow((x1 - x2), 2); });
    aux.shrink_to_fit();
    return std::sqrt(std::accumulate(aux.begin(), aux.end(), 0.0));
}

/**
 * Compute the L1 manhattan distances between the vectors in X and Y.
*/
template <typename DataType>
double manhattan_distance(const std::vector<DataType>& a, const std::vector<DataType>& b) {
    std::vector<DataType> aux;
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(aux),
                   [](DataType x1, DataType x2) { return std::abs(x1 - x2); });
    aux.shrink_to_fit();
    return std::accumulate(aux.begin(), aux.end(), 0.0);
}

/**
 * Compute the minkowski metric between 2 vectors in X and Y.
 * Minkowski metric considered as a generalization of both 
 * the Euclidean distance and the Manhattan distance.
 * 
 * @param x 1darray like data
 *      input vector 1
 * @param y 1darray like data
 *      input vector 2
 * @param p int, default is 2
 *      determinate if Euclidean, Manhattan or chebyshev
 * @return a scalar, the distance of the two vectors
*/
template <typename MatType>
double minkowski_distance(const MatType& x, const MatType& y, int p) {
    std::ostringstream err_msg;
    if (x.size() != y.size()) {
        err_msg << "x number of elements " << x.size() << " != " 
                << "y number of elements " << y.size() << std::endl; 
        throw std::invalid_argument(err_msg.str());
    }

    if (x.rows() != y.rows() || x.cols() != y.cols()) {
        err_msg << "x and y did not have the same shape." << std::endl;
        throw std::invalid_argument(err_msg.str());
    }

    std::vector<int> lookup = {1, 2, Eigen::Infinity};
    std::vector<int>::iterator iter = std::find(lookup.begin(), lookup.end(), p);
    if (iter == lookup.end()) {
        err_msg << "'p' should be 2, 1 or Eigen::Infinity." << std::endl;
        throw std::invalid_argument(err_msg.str());
    }

    double dist;
    if (p == 2) {
        dist = (x - y).template lpNorm<2>();
    }
    else if (p == 1) {
        dist = (x - y).template lpNorm<1>();
    }
    else if (p == Eigen::Infinity) {
        dist = (x - y).template lpNorm<Eigen::Infinity>();
    }

    return dist;
}


}
}
#endif /*CORE_METRIC_HPP*/

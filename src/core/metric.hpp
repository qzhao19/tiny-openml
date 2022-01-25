#ifndef CORE_METRIC_HPP
#define CORE_METRIC_HPP

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

}
}
#endif /*CORE_METRIC_HPP*/

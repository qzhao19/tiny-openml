#ifndef CORE_MATH_DISTS_HPP
#define CORE_MATH_DISTS_HPP
#include "../../prereqs.hpp"

namespace openml {

namespace math {

/**
 * gaussian probability distribution
 * -(x - u) ** 2.0 / (2.0 * v) - log(sqrt(2.0 * pi * v))
 * 
 * @param x DataType, input value
 * @param u DataType, mean
 * @param v DataType, variance 
*/
// template<typename DataType>
// double gaussian_fn(const DataType& x, 
//     const DataType& u, 
//     const DataType& v) {
    
//     // retval = -(x - u) ** 2.0 / (2.0 * v) - log(sqrt(2.0 * arma::datum::pi * v));
//     double retval = -1.0 * std::pow((x - u), 2) / (2.0 * v) - std::log(std::sqrt(2.0 * arma::datum::pi * v));

//     return retval;
// };

template <typename DataType>
double euclidean_distance(const std::vector<DataType>& a, const std::vector<DataType>& b) {
    std::vector<DataType> aux;
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(aux),
                   [](DataType x1, DataType x2) { return std::pow((x1 - x2), 2); });
    aux.shrink_to_fit();
    return std::sqrt(std::accumulate(aux.begin(), aux.end(), 0.0));
}

}
}

#endif

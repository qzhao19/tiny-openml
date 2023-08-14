#ifndef CORE_MATH_DISTS_HPP
#define CORE_MATH_DISTS_HPP
#include "../../prereqs.hpp"

namespace openml {

namespace math {

/**
 * Compute the L2 euclidean distances between the vectors in X and Y.
*/
template <typename DataType>
DataType euclidean_distance(const std::vector<DataType>& a, const std::vector<DataType>& b) {
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
DataType manhattan_distance(const std::vector<DataType>& a, const std::vector<DataType>& b) {
    std::vector<DataType> aux;
    std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(aux),
                   [](DataType x1, DataType x2) { return std::abs(x1 - x2); });
    aux.shrink_to_fit();
    return std::accumulate(aux.begin(), aux.end(), 0.0)
}



}
}

#endif

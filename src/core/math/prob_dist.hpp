#ifndef CORE_MATH_PROB_DIST_HPP
#define CORE_MATH_PROB_DIST_HPP
#include "../../prereqs.hpp"

namespace math {

/**
 * gaussian probability distribution
 * 
 * @param x DataType, input value
 * @param u DataType, mean
 * @param v DataType, variance 
 * @param retval DataType, return gaussiance value 
*/
template<typename DataType>
void gaussian_fn(const DataType& x, 
    const DataType& u, 
    const DataType& v, 
    DataType& retval) {
    
    retval = std::pow(-(x - u), 2) / (2.0 * v) - std::log(std::sqrt(2.0 * arma::datum::pi * v));
};




}

#endif

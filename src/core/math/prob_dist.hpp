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
void gaussian(const DataType& X, 
    const DataType& u, 
    const DataType& v, 
    DataType& retval) {
    
    retval = -(x - u) ** 2 / (2 * v) - log(sqrt(2 * arma::datum::pi * v))
};




}

#endif

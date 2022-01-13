#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

// template<typename DataType>
// double sigmoid(const DataType z) {
//     return DataType(1.0) / (DataType(1.0) + exp(-z));
// }

double sigmoid(const double z) {
    return 1.0 / (1.0 + exp(-z));
}




}
}

#endif /*CORE_MATH_EXTMATH_HPP*/

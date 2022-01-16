#ifndef CORE_MATH_EXTMATH_HPP
#define CORE_MATH_EXTMATH_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

// template<typename DataType>
// double sigmoid(const DataType z) {
//     return DataType(1.0) / (DataType(1.0) + exp(-z));
// }

template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType sigmoid(const MatType& mat) {
    return (static_cast<DataType>(1) / 
        (static_cast<DataType>(1) + (-mat.array()).exp())).matrix();
}


}
}

#endif

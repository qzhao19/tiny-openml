#ifndef CORE_MATH_RANDOM_HPP
#define CORE_MATH_RANDOM_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

/**
 * Create an array of the given shape and populate it with random 
 * samples from a uniform distribution over
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType rand(std::size_t num_rows, std::size_t num_cols) {
    std::random_device rand;
    std::mt19937 generator(rand());  //here set the seed
    std::uniform_real_distribution<DataType> dist{0.0, 1.0};

    MatType rand_mat = MatType::NullaryExpr(
        num_rows, num_cols, [&](){
            return dist(generator);
        }
    );
    return rand_mat;
};

/**
 * with random floats sampled from a univariate 
 * Gaussian distribution of mean 0 and variance 1
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType randn(std::size_t num_rows, std::size_t num_cols) {
    std::random_device rand;
    std::mt19937 generator(rand());  //here set the seed
    std::normal_distribution<DataType> dist{0.0, 1.0};

    MatType rand_mat = MatType::NullaryExpr(
        num_rows, num_cols, [&](){
            return dist(generator);
        }
    );
    return rand_mat;
};




}
}

#endif /*CORE_MATH_RANDOM_HPP*/


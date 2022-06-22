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

template<typename IdxType>
IdxType permutation(const std::size_t size) {
    std::random_device rand;
    std::seed_seq seed{rand()};

    //create random engines with the rng seed
    std::mt19937 generator(seed);

    //create permutation Matrix with the size
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation(size);
    permutation.setIdentity();

    std::shuffle(permutation.indices().data(), 
        permutation.indices().data() + 
            permutation.indices().size(), 
        generator);

    IdxType index;
    index = permutation.indices().template cast<Eigen::Index>();
    return index;
};


}
}

#endif /*CORE_MATH_RANDOM_HPP*/


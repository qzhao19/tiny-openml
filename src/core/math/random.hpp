#ifndef CORE_MATH_RANDOM_HPP
#define CORE_MATH_RANDOM_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace random {

/**
 * Create an array of the given shape and populate it with random 
 * samples from a uniform distribution over
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
MatType rand(std::size_t nrows, std::size_t ncols, 
    DataType low = 0.0, 
    DataType high = 1.0) {
    
    std::random_device rand;
    std::mt19937 generator(rand());  //here set the seed
    std::uniform_real_distribution<DataType> dist{low, high};

    MatType rand_mat = MatType::NullaryExpr(
        nrows, ncols, [&](){
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
MatType randn(std::size_t nrows, std::size_t ncols, 
    DataType low = 0.0, 
    DataType high = 1.0) {
    
    std::random_device rand;
    std::mt19937 generator(rand());  //here set the seed
    std::normal_distribution<DataType> dist{low, high};

    MatType rand_mat = MatType::NullaryExpr(
        nrows, ncols, [&](){
            return dist(generator);
        }
    );
    return rand_mat;
};

/**
 * Randomly permute a sequence
 * @param size int  
 *    number of sequence to generate randomly permutation 
*/
template<typename IdxVecType>
IdxVecType permutation(const std::size_t size) {
    std::random_device rand;
    std::seed_seq seed{rand()};

    //create random engines with the rng seed
    std::mt19937 generator(seed);

    //create permutation Matrix with the size
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation(size);
    permutation.setIdentity();

    std::shuffle(permutation.indices().data(), 
        permutation.indices().data() + permutation.indices().size(), 
        generator);

    IdxVecType index;
    index = permutation.indices().template cast<Eigen::Index>();
    return index;
};

/**
 * Return random integers from low to high.
 * Return random integers from the discrete uniform distribution
*/
template<typename DataType, 
    typename MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>>
MatType randint(std::size_t nrows, 
    std::size_t ncols, 
    DataType low = 0, 
    DataType high = 1e+2) {
    
    std::random_device rand;     // only used once to initialise (seed) engine
    std::seed_seq seed{rand()};

    std::mt19937 generator(seed);    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<DataType> dist(low, high);

    MatType rand_mat = MatType::NullaryExpr(
        nrows, ncols, [&](){
            return dist(generator);
        }
    );
    return rand_mat;
};


}
}

#endif /*CORE_MATH_RANDOM_HPP*/

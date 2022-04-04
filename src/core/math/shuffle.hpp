#ifndef CORE_MATH_SHUFFLE_HPP
#define CORE_MATH_SHUFFLE_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace math {

/**
 * Shuffle 2d matrices along a specific axis.
 * 
 * @param X ndarray of shape (num_samples, num_features), input matrix
 * @param shuffled_X output shuffled matrix with the same dims of input matrix
 * @param axis int Axis along which a shuffle is performed, default is 1 
*/
template<typename MatType>
void shuffle_data(const MatType& X, 
    MatType& shuffled_X, 
    int axis = 1) {
    
    std::size_t permut_size;
    if (axis == 0) {
        permut_size = X.cols();
    } 
    else if (axis = 1) {
        permut_size = X.rows();
    }
    // define a index permutation
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> index_permut(permut_size);
    index_permut.setIdentity();

    // generate random seed for shuffle 
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();;
    std::shuffle(index_permut.indices().data(), 
        index_permut.indices().data() + index_permut.indices().size(), 
        std::default_random_engine(seed)
    );
    std::cout << index_permut.indices() << std::endl;
    if (axis == 0) {
        // permute columns
        shuffled_X = X * index_permut;
    } 
    else if (axis = 1) {
        // permute rows
        shuffled_X = index_permut * X;
    } 

};

template<typename MatType, typename VecType>
void shuffle_data(const MatType& X,
    const VecType& y, 
    MatType& shuffled_X, 
    VecType& shuffled_y) {
    
    std::size_t n_rows = X.rows();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> index_permut(n_rows);
    index_permut.setIdentity();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();;
    std::shuffle(index_permut.indices().data(), 
        index_permut.indices().data() + index_permut.indices().size(), 
        std::default_random_engine(seed)
    );

    shuffled_X = index_permut * X;
    shuffled_y = index_permut * y;
};

}
}

#endif /*CORE_MATH_SHUFFLE_HPP*/

#ifndef CORE_MATH_SHUFFLE_DATA_HPP
#define CORE_MATH_SHUFFLE_DATA_HPP
#include "../../prereqs.hpp"

namespace math {

/**
 * shuffle dataset and assciated labels. It is expected that input_x 
 * and input_y must have the same number of rows.
 * 
 * @param input_X The input dataset
 * @param input_y Vector of lable associated dataset
 * @param output_X The shuffled output dataset 
 * @param output_y Shuffled vector of lable associated dataset
*/
template<typename MatrixType, 
         typename VectorType>
void shuffle_data(const MatrixType& input_X, 
                  const VectorType &input_y, 
                  MatrixType &output_X, 
                  VectorType &output_y) {
    
    
    // get samples numbers
    int n_samples = input_X.n_rows;
    
    // generate shuffled index 
    arma::uvec shuffled_idx = arma::shuffle(arma::linspace<arma::uvec>(0, 
        n_samples - 1, n_samples));

    output_X = input_X.rows(shuffled_idx);
    output_y = input_y.rows(shuffled_idx);

}


};
#endif

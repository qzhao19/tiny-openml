#ifndef CORE_DATA_SPLIT_DATA_HPP
#define CORE_DATA_SPLIT_DATA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace data {

/**
 * Split arrays or matrices into random train and test subsets
 * 
 * @param X ndarray of shape (n_samples, n_features) input dataset matrix
 * @param y ndarray of shape (n_samples, ) the related labels vector
 * @param train_size float, should be between 0.0 and 1.0 and represent the 
 *                   proportion of the dataset to include in the train split
 * @param shuffle bool, Whether or not to shuffle the data before splitting.
 * 
 * @return a tuple contains X_train, X_test, y_train, y_test matrix
*/
template<typename MatType, typename VecType>
std::tuple<MatType, MatType, VecType, VecType> train_test_split(const MatType& X, 
        const VecType& y,
        double train_size = 0.75, 
        bool shuffle = false) {
    
    std::size_t n_samples = X.n_rows;

    arma::mat X_ = X;
    arma::vec y_ = y;

    if (shuffle) {
        shuffle_data(X, y, X_, y_);
    }

    int n_train_samples = n_samples * train_size;
    int n_test_samples = n_samples - n_train_samples;

    arma::mat X_train = X_.head_rows(n_train_samples);
    arma::mat X_test = X_.tail_rows(n_test_samples);
    arma::vec y_train = y_.head_rows(n_train_samples);
    arma::vec y_test = y_.tail_rows(n_test_samples);

    return std::make_tuple(X_train, X_test, y_train, y_test);

}

}
#endif

#ifndef CORE_DATA_SPLIT_HPP
#define CORE_DATA_SPLIT_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace data {

/**
 * Split arrays or matrices into random train and test subsets
 * 
 * @param X ndarray of shape (num_samples, num_features) input dataset matrix
 * @param y ndarray of shape (num_samples, ) the related labels vector
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
        bool shuffle = true) {
    
    std::size_t num_samples = X.rows(), num_features = X.cols();

    MatType new_X = X;
    VecType new_y = y;

    if (shuffle) {
        random::shuffle_data<MatType, VecType>(X, y, new_X, new_y);
    }

    int num_train_samples = num_samples * train_size;
    int num_test_samples = num_samples - num_train_samples;

    MatType X_train = new_X.topRows(num_train_samples);
    MatType X_test = new_X.bottomRows(num_test_samples);
    VecType y_train = new_y.topRows(num_train_samples);
    VecType y_test = new_y.bottomRows(num_test_samples);

    return std::make_tuple(X_train, X_test, y_train, y_test);
};


/**
 * Divide dataset based on if sample value on feature 
 * index is smaller than the given threshold
 * @param X ndarray of shape (num_samples, num_features)
 *      input matrix dataset
 * @param feature_index int
 *      the col number we want to divide
 * @param threshold DataType
 *      the given threshold value for dividing features
 * @return the result of a tuple of vectors
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
std::tuple<std::vector<std::size_t>, 
    std::vector<std::size_t>> divide_on_feature(const MatType& X, 
        std::size_t feature_index, 
        DataType threshold) {
    
    MatType keep_X, drop_X;
    std::size_t num_rows = X.rows(), num_cols = X.cols();
    std::vector<std::size_t> keep_rows;
    std::vector<std::size_t> drop_rows;

    for (std::size_t i = 0; i < num_rows; ++i) {
        if (X(i, feature_index) <= threshold) {
            keep_rows.push_back(i);
        } 
        else {
            drop_rows.push_back(i);
        }
    }
    // IdxType cols = IdxType::LinSpaced(num_cols, 0, num_cols);
    // keep_X = X(keep_rows, cols);
    // drop_X = X(drop_rows, cols);

    return std::make_tuple(keep_rows, drop_rows);
};

} // namespace split
} // namespace openml

#endif /*CORE_DATA_SPLIT_HPP*/

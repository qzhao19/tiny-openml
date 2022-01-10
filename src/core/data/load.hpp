#ifndef CORE_DATA_LOAD_HPP
#define CORE_DATA_LOAD_HPP
#include "../../prereqs.hpp"

namespace openml {
namespace data {

/**
 * load dataset from a txt file
 * @param fp String the given filepath 
 * @param X 2d-array of shape (n_samples, n_features) the output matrix 
 * @param y 2d-array of shape (n_samples, 1) the output label matrix 
*/
template<typename MatType, 
    typename VecType,
    typename DataType = typename MatType::value_type>
void loadtxt(const std::string &fp, 
    MatType& X, 
    VecType& y) {

    std::vector<std::vector<DataType>> stdmat;
    std::ifstream file_in(fp);
    for (std::string line; std::getline(file_in, line); ) {
        std::stringstream str_stream(line);
        std::vector<DataType> row;

        for (DataType elem; str_stream >> elem; ) {
            row.push_back(elem);
        }
        stdmat.push_back(row);
    }

    // conert std 2D vector into eigen matrix
    std::size_t n_rows = stdmat.size(), n_cols = stdmat[0].size();
    MatType mat(n_rows, n_cols);
    for (std::size_t i = 0; i < n_rows; i++) {
        mat.row(i) = VecType::Map(&stdmat[i][0], n_cols);
    }
    X = mat.leftCols(n_cols - 1);
    y = mat.rightCols(1);
    
};

}
}
#endif /*CORE_DATA_LOAD_HPP*/

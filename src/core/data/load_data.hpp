#ifndef CORE_DATA_LOAD_DATA_HPP
#define CORE_DATA_LOAD_DATA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace math;

namespace data {

/**
 * load dataset from a txt file
 * @param fp String the given filepath 
 * 
 * @return tuple(X, y), X ndarray of shape (n_samples, n_features)
 *         y ndarray of shape (n_sample, )
*/
template<typename DataType>
std::tuple<arma::mat, arma::vec> loadtxt(const std::string &fp) {
    std::vector<std::vector<DataType>> stdmat;

    std::ifstream file_in(fp);
    for (std::string line; std::getline(file_in, line); ) {
        std::stringstream ss(line);
        std::vector<DataType> row;

        for (DataType elem; ss >> elem; ) {
            row.push_back(elem);
        }
        stdmat.push_back(row);
    }

    std::size_t n_samples = stdmat.size(), n_features = stdmat[0].size();
    arma::mat data;
    for (std::size_t i = 0; i < n_samples; i++) {
        arma::rowvec row = arma::conv_to<arma::rowvec>::from(stdmat[i]);
        data.insert_rows(i, row);
    }

    arma::mat X = data.head_cols(n_features - 1);
    arma::vec y = data.tail_cols(1);
    
    return std::make_tuple(X, y);
    
};

}
#endif

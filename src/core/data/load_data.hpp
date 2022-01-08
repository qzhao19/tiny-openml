#ifndef CORE_DATA_LOAD_DATA_HPP
#define CORE_DATA_LOAD_DATA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace data {

/**
 * load dataset from a txt file
 * @param fp String the given filepath 
 * @param data 2d-array of shape (n_samples, n_features) the output matrix 
 * 
*/
template<typename MatType, 
    typename DataType = typename MatType::value_type>
void loadtxt(const std::string &fp, 
    MatType& data) {

    using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<DataType, Eigen::Dynamic>;

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

    std::size_t n_rows = stdmat.size(), n_cols = stdmat[0].size();
    Matrix mat(n_rows, n_cols);
    for (std::size_t i = 0; i < n_rows; i++) {
        mat.row(i) = Vector::Map(&stdmat[i][0], n_cols);
    }
    data = mat;
    
};

}
}
#endif

#ifndef CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP
#define CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace preprocessing {

template<typename DataType>
class TransactionEncoder {
private:
    using TripletType = Eigen::Triplet<DataType>;
    using SpMatType = Eigen::SparseMatrix<DataType>;
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

public:
    const SpMatType fit_transform(const std::vector<std::vector<DataType>>& X) const {
        
        std::set<DataType> cols_set;
        std::map<DataType, std::size_t> cols_map;

        for (std::size_t i = 0; i < X.size(); ++i) {
            for (std::size_t j = 0; j < X[i].size(); ++j) {
                cols_set.insert(X[i][j]);
            }
        }
        std::size_t idx = 0;
        for (auto it = cols_set.begin(); it != cols_set.end(); it++) {
            cols_map[*it] = idx;
            ++idx;
        }

        idx = 0;
        std::vector<DataType> non_sparse_values;
        std::vector<std::size_t> col_idx, row_idx;
        std::vector<std::vector<DataType>> copy_X = X;

        for (auto& row : copy_X) {
            // remove row duplicate
            std::set<DataType> seen;
            for (auto& x : row) {
                seen.insert(x);
            }
            row.assign(seen.begin(), seen.end());
            for (auto& x : row) {
                if (cols_map.find(x) != cols_map.end()) {
                    row_idx.emplace_back(idx);
                    col_idx.emplace_back(cols_map[x]);
                    non_sparse_values.emplace_back(static_cast<DataType>(1));
                }
            }
            ++idx;
        }

        std::size_t num_rows = X.size();
        std::size_t num_cols = cols_set.size();
        std::size_t num_non_sparse_values = non_sparse_values.size();
        
        SpMatType sp_mat(num_rows, num_cols);
        std::vector<TripletType> triplets;
        for (std::size_t i = 0; i < num_non_sparse_values; i++) {
            triplets.emplace_back(row_idx[i], col_idx[i], non_sparse_values[i]);
        }
        sp_mat.setFromTriplets(triplets.begin(), triplets.end());

        return sp_mat;
    }

};

}
}

#endif /*CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP*/
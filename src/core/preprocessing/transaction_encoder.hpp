#ifndef CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP
#define CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
using namespace openml;

namespace openml {
namespace preprocessing {

template<typename DataType>
class TransactionEncoder {
private:

    using SpMatType = Eigen::SparseMatrix<DataType>;
    using TripletType = Eigen::Triplet<DataType>;

    std::set<DataType> cols_;
    std::map<DataType, std::size_t> cols_map_;

    void fit_data(const std::vector<std::vector<DataType>>& X) {

        for (std::size_t i = 0; i < X.size(); ++i) {
            for (std::size_t j = 0; j < X[i].size(); ++j) {
                cols_.insert(X[i][j]);
            }
        }
        std::size_t idx = 0;
        for (auto it = cols_.begin(); it != cols_.end(); it++) {
            cols_map_[*it] = idx;
            ++idx;
        }
    }


    void transform_data(const std::vector<std::vector<DataType>>& X, bool sparse) {

        if (sparse) {

            std::vector<std::size_t> col_idx;
            std::vector<std::size_t> row_idx;

            std::vector<std::pair<std::size_t, std::size_t>> duplicate_indices;

            for (std::size_t i = 0; i < X.size(); ++i) {

                std::set<DataType> seen;

                for (std::size_t j = 0; j < X[i].size(); ++j) {
                    

                    if (cols_map_.find(X[i][j]) != cols_map_.end()) {
                        row_idx.emplace_back(i);
                        col_idx.emplace_back(cols_map_[X[i][j]]);
                    }

                    if (seen.find(X[i][j]) != seen.end()) {
                        duplicate_indices.emplace_back(std::make_pair(i, j));
                    }
                    else{
                        seen.insert(X[i][j]);
                    }
                }
                
            }

        }
        else{




        }
    }




};

}
}

#endif /*CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP*/
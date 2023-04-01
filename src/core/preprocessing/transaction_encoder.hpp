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
    std::set<DataType> cols_;
    std::map<DataType, std::size_t> cols_map_;

    void fit_data(const std::vector<std::vector<DataType>>& X) {

        for (std::size_t i = 0; i < X.size(); ++i) {
            for (std::size_t j = 0; j < X[i].size(); ++j) {
                cols_.insert(X[i][j]);
            }
        }

        std::size_t index = 0;
        for (auto it = ss.begin(); it != ss.end(); it++) {
            cols_map_[index] = it;
            ++index;
        }

    }


    void transform_data(const std::vector<std::vector<DataType>>& X, bool sparse) {

        if (sparse) {
            





        }



    }




};

}
}

#endif /*CORE_PREPPROCESSING_TRANSACTION_ENCODER_HPP*/
#ifndef METHOD_RULE_MODEL_APRIORI_HPP
#define METHOD_RULE_MODEL_APRIORI_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml {
namespace rule_model {

template<typename DataType>
class Apriori {
private:
    // define matrix and vector Eigen type
    using SpMatType = Eigen::SparseMatrix<DataType, Eigen::RowMajor>;
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    double min_support_;
    double min_confidence_;

    struct ItemsetFrequency {
        std::vector<std::vector<DataType>> itemsets_list;
        std::vector<std::size_t> count_list;

        ItemsetFrequency(std::vector<std::vector<DataType>> itemsets_list_, 
            std::vector<std::size_t> count_list_): itemsets_list(itemsets_list_), 
                count_list(count_list_) {};
        ~ItemsetFrequency() {};
    };

    std::map<std::size_t, ItemsetFrequency> all_frequency_;

protected:
    void generate_candidates(std::size_t k) {
        
        std::vector<std::vector<DataType>> candidates;
        if (k == 1) {
            candidates = common::combinations<DataType>(
                all_frequency_[1]->itemsets_list[0], k + 1
            );
        }
        else if (k == 2) {
            std::vector<DataType> frequency_1 = all_frequency_[1]->itemsets_list[0];
            std::vector<std::vector<DataType>> frequency_2 = all_frequency_[2]->itemsets_list;

            for (std::size_t i = 0; i < frequency_2.size(); ++i) {
                for (std::size_t j = 0; j < frequency_1.size(); ++j) {
                    if (std::find(frequency_2[i].begin(), frequency_2[i].end(), frequency_1[j]) == frequency_2[i].end()) {
                        frequency_2[i].emplace_back(frequency_1[j]);
                        candidates.emplace_back(frequency_2[i]);
                    }
                }
            }
        }
        else {
            std::vector<std::vector<DataType>> frequent_k = all_frequency_[k]->itemsets_list;

        }
    }


    


public:

    Apriori(): min_support_(0.04), min_confidence_(0.6) {};

    Apriori(double min_support, double min_confidence): 
        min_support_(min_support), 
        min_confidence_(min_confidence) {};
    
    ~Apriori() {};

    void fit(const std::vector<std::vector<DataType>>& X) {
        
        std::size_t support = static_cast<std::size_t>(X.size() * min_support_);

        std::cout << support << std::endl;

        preprocessing::TransactionEncoder<std::size_t> transaction_encoder;
        SpMatType sp_mat = transaction_encoder.fit_transform(X);

        // std::cout << sp_mat << std::endl;
        // std::vector<DataType> all_records;

        std::map<DataType, std::size_t> all_records;
        for (int i = 0; i < sp_mat.nonZeros(); ++i) {
            auto record = *(sp_mat.valuePtr() + i);
            all_records[record]++;
        }
        
        // for(auto record : all_records) {
        //     std::cout << "item = " << record.first << ", count = " << record.second <<std::endl;
        // }

        for(auto it = all_records.begin(); it != all_records.end(); ) {
            if(it->second < support) {
                all_records.erase(it++);
            }
            else {
                ++it;
            }
        }


    }


};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/
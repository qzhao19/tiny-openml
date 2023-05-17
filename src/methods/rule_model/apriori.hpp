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
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    double min_support_;
    double min_confidence_;

    struct ItemsetFrequency {
        std::vector<std::vector<DataType>> itemsets_list;
        std::vector<std::size_t> count_list;
    };

    std::map<std::size_t, ItemsetFrequency> all_frequency_;

    struct HashTreeNode {
        bool is_leaf;
        std::size_t index;
        std::map<std::vector<DataType>, std::size_t> bucket;
        std::map<std::size_t, std::shared_ptr<HashTreeNode>> children;
        
        HashTreeNode(): is_leaf(true), index(0) {};
        HashTreeNode(bool is_leaf_, 
            std::size_t index_ = 0): is_leaf(is_leaf_), 
                index(index_) {};

        ~HashTreeNode() {};
    };



protected:
    const std::vector<std::vector<DataType>> generate_candidates(std::size_t k) {
        
        std::vector<std::vector<DataType>> candidates;
        
        return candidates;
    }


    void sort_itemsets_counts(std::vector<std::vector<DataType>>& itemsets_list, 
        std::vector<std::size_t>& count_list) {
        
        std::vector<std::pair<std::vector<DataType>, std::size_t>> combine;
        for (int i = 0; i < itemsets_list.size(); ++i)
            combine.push_back(std::make_pair(itemsets_list[i], count_list[i]));

        std::sort(combine.begin(), combine.end(), [](const auto &left, const auto &right) {
            return left.first < right.first;
        });

        for (int i = 0; i < itemsets_list.size(); ++i) {
            itemsets_list[i] = combine[i].first;
            count_list[i] = combine[i].second;
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

        // std::unordered_map<DataType, std::size_t> all_records;
        // for (int i = 0; i < sp_mat.nonZeros(); ++i) {
        //     auto record = *(sp_mat.valuePtr() + i);
        //     all_records[record]++;
        // }
        
        // for(auto record : all_records) {
        //     std::cout << "item = " << record.first << ", count = " << record.second <<std::endl;
        // }

        std::vector<std::pair<DataType, std::size_t>> all_records;
        for (std::size_t i = 0; i < X.size(); ++i) {
            for (std::size_t j = 0; j < X[i].size(); ++j) {
                // if (std::find(all_records.begin(), all_records.end(), vvd[i][j])) {

                // }
                // int count = std::count_if(all_records.begin(), all_records.end(), [&vvd[i][j]](double &i) {
                //     return i == vvd[i][j];
                // });

                auto result1 = std::find(begin(all_records), end(all_records), X[i][j]);

                if (result1 == std::end(all_records)) {
                    all_records.emplace_back(vvd[i][j]);
                }
            }
        }   


        // for(auto it = all_records.begin(); it != all_records.end(); ) {
        //     if(it->second < support) {
        //         all_records.erase(it++);
        //     }
        //     else {
        //         ++it;
        //     }
        // }

        // read itemsets and their count numbers
        std::vector<std::vector<DataType>> itemsets_list;
        std::vector<std::size_t> count_list;
        std::vector<DataType> itemsets;
        for (auto record : all_records) {
            itemsets.emplace_back(record.first);
            count_list.emplace_back(record.second);
        }
        itemsets_list.emplace_back(itemsets);


        // for(auto c : count_list) {
        //     std::cout << "item number = " << c <<std::endl;
        // }

        // for (std::size_t i = 0; i < itemsets_list.size(); ++i) {
        //     for (std::size_t j = 0; j < itemsets_list[i].size(); ++j) {
        //         std::cout << "item = " << itemsets_list[i][j] << ", item number = "<< count_list[j] << std::endl;
        //     }
        // }


        // put them into a map with key = id, value = itemset_frequency
        // ItemsetFrequency itemset_frequency = {itemsets_list, count_list};
        // all_frequency_[1] = itemset_frequency;

        
        // std::vector<std::vector<DataType>> candidates;
        // candidates = generate_candidates(1);
        // for (std::size_t i = 0; i < candidates.size(); ++i) {
        //     for (std::size_t j = 0; j < candidates[i].size(); ++j) {
        //         std::cout << candidates[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // std::size_t k = 1;
        // while (itemsets_list.size() > 0) {
        //     std::vector<std::vector<DataType>> candidates;
        //     candidates = generate_candidates(k);

        //     std::size_t num_candidates = candidates.size();
        //     ++k;

        //     // init hash tree
        //     tree::HashTree<HashTreeNode, DataType> hash_tree(num_candidates, num_candidates);
        //     hash_tree.build_tree(candidates);

        //     for (std::size_t i = 0; i < X.size(); ++i) {
        //         if (X[i].size() < k) {
        //             break;
        //         }
        //         std::vector<DataType> pick_itemset;
        //         hash_tree.add_support(pick_itemset, X[i], k);
        //     }

        //     hash_tree.compute_frequency_itemsets(support, count_list, itemsets_list);

        //     if ((!count_list.empty()) && (!itemsets_list.empty())) {
        //         sort_itemsets_counts(itemsets_list, count_list);
        //         itemset_frequency = {itemsets_list, count_list};

        //         // for (std::size_t i = 0; i != itemsets_list.size(); ++i) {
        //         //     for (std::size_t j = 0; j != itemsets_list[i].size(); ++j) {
        //         //         std::cout << itemsets_list[i][j] << " ";
        //         //     }
        //         //     std::cout << std::endl;
        //         // }


        //         all_frequency_[k] = itemset_frequency;
        //     }
        // }
    }


};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/
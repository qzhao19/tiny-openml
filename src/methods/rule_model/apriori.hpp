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
    using FrequencyType = std::vector<std::pair<std::vector<DataType>, std::size_t>>;

    double min_support_;
    double min_confidence_;

    // FrequencyType all_frequency_;

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
    const std::vector<std::vector<DataType>> generate_k_subsets(
        const std::vector<std::vector<DataType>>& X, 
        std::size_t k) {
        
        std::vector<std::vector<DataType>> candidates;
        std::map<std::size_t, std::vector<std::vector<DataType>>> candidates_map;
        
        for (std::size_t i = 0; i < X.size(); ++i) {
            candidates_map[i] = common::combinations<DataType>(X[i], k);
        }

        for (auto candidate : candidates_map) {
            for (auto item : candidate.second) {
                candidates.emplace_back(item);
            }
        }
        
        return candidates;
    }

    bool is_prefix(const std::vector<DataType>& l1, const std::vector<DataType>& l2) {
        for (std::size_t i = 0; i < (l1.size() - 1); ++i) {
            if (l1[i] != l2[i]) {
                return false;
            }
        }
        return true;
    }

public:

    Apriori(): min_support_(0.15), min_confidence_(0.6) {};

    Apriori(double min_support, double min_confidence): 
        min_support_(min_support), 
        min_confidence_(min_confidence) {};
    
    ~Apriori() {};

    void fit(const std::vector<std::vector<DataType>>& X) {    
        std::size_t support = static_cast<std::size_t>(X.size() * min_support_);
    
        // std::cout << sp_mat << std::endl;
        std::vector<DataType> records_order;
        for (std::size_t i = 0; i < X.size(); ++i) {
            for (std::size_t j = 0; j < X[i].size(); ++j) {
                auto it = std::find(begin(records_order), end(records_order), X[i][j]);
                if (it == std::end(records_order)) {
                    records_order.emplace_back(X[i][j]);
                }
            }
        }

        preprocessing::TransactionEncoder<std::size_t> transaction_encoder;
        SpMatType sp_mat = transaction_encoder.fit_transform(X);
        std::unordered_map<DataType, std::size_t> all_records;
        for (int i = 0; i < sp_mat.nonZeros(); ++i) {
            auto record = *(sp_mat.valuePtr() + i);
            all_records[record]++;
        }

        for(auto it = all_records.begin(); it != all_records.end(); ) {
            if(it->second < support) {
                all_records.erase(it++);
            }
            else {
                ++it;
            }
        }
        
        FrequencyType all_frequency;
        std::vector<std::vector<DataType>> prev_frequency;

        for (auto record : records_order) {
            if (all_records.find(record) != all_records.end()) {
                auto count = all_records[record];

                std::vector<DataType> frequency;
                frequency.emplace_back(record);

                prev_frequency.emplace_back(frequency);

                auto tmp = std::make_pair(frequency, count);
                all_frequency.emplace_back(tmp);
            }
        }


        std::size_t length = 2;
        while (length < 5) {
            // init candidates
            std::vector<std::vector<DataType>> candidates;
            for (std::size_t i = 0; i < prev_frequency.size(); ++i) {
                std::size_t j = i + 1;
                while ((j < prev_frequency.size()) && is_prefix(prev_frequency[i], prev_frequency[j])) {
                    std::vector<DataType> tmp;
                    tmp.assign(prev_frequency[i].begin(), prev_frequency[i].end() - 1);
                    tmp.emplace_back(prev_frequency[i].back());
                    tmp.emplace_back(prev_frequency[j].back());
                    candidates.emplace_back(tmp);
                    ++j;
                    tmp.clear();
                }
            }
            
            std::size_t num_candidates = candidates.size();
            // init hash tree
            tree::HashTree<HashTreeNode, DataType> hash_tree(num_candidates, num_candidates);
            for (auto candidate : candidates) {
                // add this itemset to hashtree
                hash_tree.build_tree(candidate);
            }

            // for each transaction, find all possible subsets of size "length"
            std::vector<std::vector<DataType>> subsets;
            subsets = generate_k_subsets(X, length);
            

            for (auto subset : subsets) {
                hash_tree.add_support(subset);
            }
            
            std::vector<std::size_t> count_list;
            std::vector<std::vector<DataType>> itemsets_list;

            hash_tree.compute_frequency_itemsets(support, count_list, itemsets_list);
            FrequencyType curr_frequency;

            for (int i = 0; i < itemsets_list.size(); ++i) {
                curr_frequency.emplace_back(std::make_pair(itemsets_list[i], count_list[i]));
            }

            // 
            all_frequency.insert(all_frequency.end(), curr_frequency.begin(), curr_frequency.end());
            prev_frequency = common::sortrows<DataType>(itemsets_list);

            ++length;
        }

    }


};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/
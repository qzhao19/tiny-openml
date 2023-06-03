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
    using FrequencyType = std::vector<std::pair<std::vector<DataType>, std::size_t>>;
    using RuleType = std::tuple<std::vector<DataType>, std::vector<DataType>, double>;

    double min_support_;
    double min_confidence_;

    // create a vector to restore all frequency
    FrequencyType all_frequency_;
    std::vector<RuleType> association_rules_;

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


protected:
    void fit_data(const std::vector<std::vector<DataType>>& X) {    
        std::size_t support = static_cast<std::size_t>(X.size() * min_support_);
    
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

        if (all_records.empty()) {
            std::ostringstream err_msg;
            err_msg << "The 'min_support_ = ', it's not available" << support 
                    << " so try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        // FrequencyType all_frequency;
        std::vector<std::vector<DataType>> prev_frequency;
        for (auto record : records_order) {
            if (all_records.find(record) != all_records.end()) {
                auto count = all_records[record];

                std::vector<DataType> frequency;
                frequency.emplace_back(record);

                prev_frequency.emplace_back(frequency);

                auto tmp = std::make_pair(frequency, count);
                all_frequency_.emplace_back(tmp);
            }
        }

        std::size_t length = 2;
        while (prev_frequency.size() > 1) {
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
            

            if (subsets.size() > 0) {
                for (auto subset : subsets) {
                    hash_tree.add_support(subset);
                }
            }
            
            std::vector<std::size_t> count_list;
            std::vector<std::vector<DataType>> itemsets_list;
            
            // find frequent itemsets
            hash_tree.compute_frequency_itemsets(support, count_list, itemsets_list);
            FrequencyType curr_frequency;

            for (int i = 0; i < itemsets_list.size(); ++i) {
                curr_frequency.emplace_back(std::make_pair(itemsets_list[i], count_list[i]));
            }

            all_frequency_.insert(all_frequency_.end(), curr_frequency.begin(), curr_frequency.end());
            prev_frequency = common::sortrows<DataType>(itemsets_list);

            ++length;
        }
    }

    void transform_data() { 
        std::map<std::vector<DataType>, std::size_t> frequency_map;

        for (auto frequency : all_frequency_) {
            if (frequency.second > 0) {
                frequency_map[frequency.first] = frequency.second;
            }
        }
        
        // generates association rules with confidence greater than threshold confidence
        for (const auto& frequency : frequency_map) {
            std::size_t itemset_size = frequency.first.size();
            if (itemset_size == 1) {
                continue;
            }

            double itemset_count = static_cast<double>(frequency.second);
            for (std::size_t i = 1; i < itemset_size; ++i) {
                std::vector<std::vector<DataType>> candidate;
                candidate = common::combinations<DataType>(frequency.first, i);
                for (const auto& c : candidate) {
                    if (frequency_map.find(c) != frequency_map.end()) {
                        // compute the confidence
                        double confidence = itemset_count / static_cast<double>(frequency_map[c]);
                        if (confidence >= min_confidence_) {
                            std::vector<DataType> diff_c = common::difference<DataType>(frequency.first, c);                             
                            RuleType rule = std::make_tuple(c, diff_c, confidence);
                            association_rules_.emplace_back(rule);
                        }
                    }
                }
            }
        }
    }

public:
    Apriori(): min_support_(0.03), min_confidence_(0.5) {};

    Apriori(double min_support, double min_confidence): 
        min_support_(min_support), 
        min_confidence_(min_confidence) {};
    
    ~Apriori() {};

    void fit_transform(const std::vector<std::vector<DataType>>& X) {
        fit_data(X);
        transform_data();        
    }

    void print_rules() {
        for (auto rule : association_rules_) {
            std::vector<DataType> item1 = std::get<0>(rule);
            std::vector<DataType> item2 = std::get<1>(rule);
            double confidence = std::get<2>(rule);
            
            for (auto it1 : item1) {
                std::cout << it1 << ", ";
            }
            std::cout << " ==> ";
            for (auto it2 : item2) {
                std::cout << it2 << ", ";
            }
            std::cout << " confidence = " << confidence << std::endl;
        }
    }

    const std::vector<RuleType> get_rules() {
        return association_rules_;
    }


};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/
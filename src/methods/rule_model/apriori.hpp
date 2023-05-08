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

        // ItemsetFrequency(const std::vector<std::vector<DataType>>& itemsets_list_, 
        //     const std::vector<std::size_t>& count_list_): itemsets_list(itemsets_list_), 
        //         count_list(count_list_) {};
        
        // ~ItemsetFrequency() {};
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
        if (k == 1) {
            candidates = common::combinations<DataType>(
                all_frequency_[1].itemsets_list[0], k + 1
            );
        }
        else if (k == 2) {
            std::vector<DataType> frequency_1 = all_frequency_[1].itemsets_list[0];
            std::vector<std::vector<DataType>> frequency_2 = all_frequency_[2].itemsets_list;

            for (std::size_t i = 0; i < frequency_2.size(); ++i) {
                for (std::size_t j = 0; j < frequency_1.size(); ++j) {
                    if (std::find(frequency_2[i].begin(), frequency_2[i].end(), frequency_1[j]) == frequency_2[i].end()) {
                        frequency_2[i].emplace_back(frequency_1[j]);
                        candidates.emplace_back(frequency_2[i]);
                        frequency_2[i].pop_back();
                    }                
                }
            }
        }
        else {
            std::vector<std::vector<DataType>> frequency_k = all_frequency_[k].itemsets_list;
            std::set<std::vector<DataType>> frequency_k_reference;

            for (auto frequency : frequency_k) {
                frequency_k_reference.insert(frequency);
            }

            for (std::size_t i = 0; i < frequency_k.size(); ++i) {
                for (std::size_t j = i + 1; j < frequency_k.size(); ++j) {
                    
                    std::vector<DataType> subvector_i = {frequency_k[i].begin(), frequency_k[i].end() - 1};
                    std::vector<DataType> subvector_j = {frequency_k[j].begin(), frequency_k[j].end() - 1};
                    
                    if (common::is_equal<DataType>(subvector_i, subvector_j)) {
                        bool found = true;
                        std::vector<DataType> tmp = subvector_i;
                        tmp.emplace_back(frequency_k[i].back());
                        tmp.emplace_back(frequency_k[j].back());

                        std::vector<std::vector<DataType>> comb = combinations<DataType>(tmp, k);
                        for (const auto& c : comb) {
                            if (frequency_k_reference.find(c) == frequency_k_reference.end()) {
                                found = false;
                                break;
                            }
                        }

                        if (found) {
                            candidates.emplace_back(tmp);
                        }
                        
                    }
                }
            }
        }

        return candidates;

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
        ItemsetFrequency itemset_frequency = {itemsets_list, count_list};
        all_frequency_[1] = itemset_frequency;

        // std::vector<std::vector<DataType>> candidates;
        // candidates = generate_candidates(1);

        // for (std::size_t i = 0; i < candidates.size(); ++i) {
        //     for (std::size_t j = 0; j < candidates[i].size(); ++j) {
        //         std::cout << "candidates = " << candidates[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        std::size_t k = 1;
        while (itemsets_list.size() > 0) {
            std::vector<std::vector<DataType>> candidates;
            candidates = generate_candidates(k);

            std::size_t num_candidates = candidates.size();
            ++k;

            // init hash tree
            tree::HashTree<HashTreeNode<DataType>, DataType> hash_tree;
            hash_tree.build_tree(candidates);

            for (std::size_t i = 0; i < X; ++i) {
                if (X[i].size() < k) {
                    break;
                }
                std::vector<DataType> pick_itemset;
                hash_tree.add_support(pick_itemset, X[i], k);
            }

            hash_tree.compute_frequency_itemsets(0, count_list, itemsets_list);

        }

    }


};

} // neighbors
} // openml

#endif /*METHOD_RULE_MODEL_APRIORI_HPP*/
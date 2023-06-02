#ifndef CORE_TREE_HASH_TREE_HPP
#define CORE_TREE_HASH_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

template<typename NodeType, typename DataType>
class HashTree {
private:
    std::size_t max_leaf_size_;
    std::size_t max_child_size_;

    std::shared_ptr<NodeType> root_;
    std::vector<std::size_t> count_list_;
    std::vector<std::vector<DataType>> itemset_list_;

    inline const std::size_t hash(std::size_t num) const {
        return num % max_child_size_;
    }

protected:
    void insert(const std::shared_ptr<NodeType>& node, 
        std::vector<DataType> itemset, 
        std::size_t count, 
        std::size_t index) {
        
        std::size_t key;
        // if current itemset is the last one, just insert it
        if (index == itemset.size()) {
            // if itemset is in bucket
            if (node->bucket.find(itemset) != node->bucket.end()) {
                node->bucket[itemset] += count;
            }
            else {
                node->bucket[itemset] = count;
            }
            return ;    
        }
        
        // non-leaf node 
        if (!node->is_leaf) {
            key = hash(itemset[index]);
            // if element k of current itemset is not in
            if (node->children.find(key) == node->children.end()) {
                node->children[key] = std::make_shared<NodeType>();
            }
            insert(node->children[key], itemset, count, index + 1);
        }
        else {
            if (node->bucket.find(itemset) == node->bucket.end()) {
                node->bucket[itemset] = count;
            }
            else {
                node->bucket[itemset] += count;
            }

            if (node->bucket.size() == max_leaf_size_) {
                // bucket is a map struct and key is vector
                // bucket has reached its maximum capacity its intermediate 
                // node so split and redistribute entries
                for (auto bucket : node->bucket) {
                    key = hash(bucket.first[index]);
                    if (node->children.find(key) == node->children.end()) {
                        node->children[key] = std::make_shared<NodeType>();
                    }
                    insert(node->children[key], bucket.first, bucket.second, index + 1);
                }
                node->bucket.clear();
                node->is_leaf = false;
            }   
        }
    }

    void dfs(std::shared_ptr<NodeType>& node, std::size_t support) {
        if (node->is_leaf) {
            for (auto& bucket : node->bucket) {
                if (bucket.second >= support) {
                    itemset_list_.emplace_back(bucket.first);
                    count_list_.emplace_back(bucket.second);
                }
            }
            return ;
        }
        for (auto& child : node->children) {
            dfs(child.second, support);
        }
        
    }

public:
    HashTree(): max_leaf_size_(3), max_child_size_(3) {
        root_ = std::make_shared<NodeType>();
    };

    HashTree(std::size_t max_leaf_size, 
        std::size_t max_child_size):
            max_leaf_size_(max_leaf_size), 
            max_child_size_(max_child_size) {
        root_ = std::make_shared<NodeType>();
    };

    ~HashTree() {};

    void build_tree(const std::vector<DataType>& itemset) {
        insert(root_, itemset, 0, 0);
    }

    void compute_frequency_itemsets(std::size_t support, 
        std::vector<std::size_t>& count_list,
        std::vector<std::vector<DataType>>& itemset_list) {
    
        dfs(root_, support);
        count_list = count_list_;
        itemset_list = itemset_list_;
    }

    void add_support(const std::vector<DataType>& itemset) {
        std::size_t index = 0;
        std::shared_ptr<NodeType> node = root_;
        while (true) {
            if (node->is_leaf) {
                if (node->bucket.find(itemset) != node->bucket.end()) {
                    ++(node->bucket[itemset]);
                }
                break;
            }
            std::size_t key = hash(itemset[index]);
            if (node->children.find(key) != node->children.end()) {
                node = node->children[key];
            }
            else {
                break;
            }
            ++index;
        }
    }
    

};

}
}
#endif /*CORE_TREE_HASH_TREE_HPP*/
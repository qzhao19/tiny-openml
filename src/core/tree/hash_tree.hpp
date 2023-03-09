#ifndef CORE_TREE_HASH_TREE_HPP
#define CORE_TREE_HASH_TREE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace tree {

template<typename DataType, typename NodeType>
class HashTree {
private:
    std::size_t max_leaf_size_;
    std::size_t max_child_size_;

    std::shared_ptr<NodeType> root_;
    std::set<std::vector<DataType>> added_;
    std::vector<std::size_t> count_list_;
    std::vector<std::vector<DataType>> itemset_list_;
    
    inline const std::size_t hash(std::size_t num) const {
        return num % max_child_size_;
    }

    void dfs(std::shared_ptr<Node> node, double support) {

    };


public:
    HashTree(): max_leaf_size_(3), max_child_size_(3) {
        root_ = std::make_unique<NodeType>();
        root_->is_leaf = false;
    };

    HashTree(std::size_t max_leaf_size, 
        std::size_t max_child_size):
            max_leaf_size_(max_leaf_size), 
            max_child_size_(max_child_size) {
        root_ = std::make_unique<NodeType>();
        root_->is_leaf = false;
    };

    void insert(std::shared_ptr<Node>& node, 
        std::vector<DataType> itemset, 
        std::size_t count) {
        
        if (node->index == itemset.size()) {
            if (node->bucket.find(itemset) == node->bucket.end()) {
                node->bucket[itemset] = count;
            } 
            else {
                node->bucket[itemset] = (++count);
            }
        }



    }



};

}
}

#endif /*CORE_TREE_HASH_TREE_HPP*/
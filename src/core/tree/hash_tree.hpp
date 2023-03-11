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
    std::set<std::vector<DataType>> added_;
    std::vector<std::size_t> count_list_;
    std::vector<std::vector<DataType>> itemset_list_;

    inline const std::size_t hash(std::size_t num) const {
        return num % max_child_size_;
    }



protected:

    void insert(std::shared_ptr<NodeType>& node, 
        std::vector<DataType> itemset, 
        std::size_t count) {
        
        // if current itemset is the last one, just insert it
        if (node.index == itemset.size()) {
            node.bucket[itemset] = count;
        }
        else {
            node.bucket[itemset] = (++count);
        }

        if (!node.is_leaf) {
            std::size_t key = hash(itemset[node.index]);
            if (node.children.find(key) == node.children.end()) {
                node.children[key] = std::make_shared<NodeType>;
            }
            insert(node.children[key], itemset, count)
        }
        else {
            if (node.children.find(itemset) == node.children.end()) {
                node.bucket[itemset] = count;
            }
            else {
                node.bucket[itemset] = (++count);
            }

            
        }


    }


public:
    HashTree(): max_leaf_size_(3), max_child_size_(3) {
        // root_ = std::make_shared<NodeType>();
        // root_->is_leaf = false;
    };

    HashTree(std::size_t max_leaf_size, 
        std::size_t max_child_size):
            max_leaf_size_(max_leaf_size), 
            max_child_size_(max_child_size) {
        // root_ = std::make_shared<NodeType>();
        // root_->is_leaf = false;
    };   


};

}
}

#endif /*CORE_TREE_HASH_TREE_HPP*/
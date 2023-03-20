#include "../src/core/tree/hash_tree.hpp"   
using namespace openml;

template<typename DataType>
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

int main() {
    
    std::vector<std::vector<int>> itemsets{{1,4,5},{1,2,4},{4,5,7},{1,2,5},
            {4,5,8},{1,5,9},{1,3,6},{2,3,4},
            {5,6,7},{3,4,5},{3,5,6},{3,5,7},
            {6,8,9},{3,6,7},{3,6,8},{6,5,3}};

    HashTreeNode<int> node(false);
    tree::HashTree<HashTreeNode<int>, int> tree;
    // tree.print(node);
    tree.build_tree(itemsets);
    tree.compute_frequency_itemsets(0);

}
